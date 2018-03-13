import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

import miscellany.losses as losses
import miscellany.layers as layers


EPS = 1e-5

class RefineTrainer(object):
    """The object that manages the training of the network

    Maintains a collection of other models that first weed out pixels from
    consideration, then creates weight-map for loss function and backpropagates
    network-in-training against pixels weighted accordingly
    """

    def __init__(self, input_placeholder, labels_placeholder, kp_placeholder,
                 net, priors, server, learning_rate, sess,
                 tversky_alpha, tester):
        """Constructs the RefineTrainer Object

        Defines the net's loss function based on computation graph involving
        the priors and iteratively optimizes it.
        """
        # Feeding here allows for querying values computed by any prior or net
        self.input_placeholder = input_placeholder
        self.labels_placeholder = labels_placeholder
        self.kp_placeholder = kp_placeholder

        # Alpha for tversky score (precision weight in denominator)
        self.tversky_alpha = tversky_alpha

        # For running test step
        self.tester = tester

        self.intensities = tf.slice(
            self.input_placeholder,
            [0,0,0,0],
            [-1,-1,-1,1]
        )

        # Models involved in training
        self.priors = priors
        self.net = net

        # Keep prob is a float value for training - NOT A TENSOR PLACEHOLDER
        self.server = server

        # Save tensor of ones where data is not padding and zeros otherwise
        self.considering = tf.cast(
            tf.less(
                self.labels_placeholder,
                2
            ),
            tf.int64
        )
        self.labels = tf.multiply(
            self.labels_placeholder,
            self.considering
        )


        # Get prior precision to be used in computing class weights
        #  and define computation graph that brings input volume to
        #  hot only where not yet removed from contention
        self.inarrowed = tf.multiply(
            self._run_priors(priors),
            self.considering
        )
        self.narrowed = tf.cast(self.inarrowed, tf.float32)
        self.vinarrowed = tf.multiply(
            self._run_priors(priors, master_threshold=0.5),
            self.considering
        )
        self.vnarrowed = tf.cast(self.vinarrowed, tf.float32)

        self.prior_precision = self.priors[-1].precision
        print("Prior precision:", self.prior_precision)




        # define computation graph with computes loss weights
        self.weights = self._get_weights_sampling(
            self.narrowed, self.prior_precision, self.labels,
            self.tversky_alpha, self.considering
        )
        # define compute for validation weights for computing cost only,
        #   not for training
        self.val_weights = self._get_weights_sampling(
            self.vnarrowed, self.prior_precision, self.labels,
            self.tversky_alpha, self.considering
        )

        # define computation graph that computes the model objective value
        self.loss, self.flat_loss_map = losses.weighted_pixelwise_softmax(
            self.labels, self.net.logits, self.weights,
            self.server.adapter.out_channels
        )
        self.val_loss, self.val_flat_loss_map = losses.weighted_pixelwise_softmax(
            self.labels, self.net.logits, self.val_weights,
            self.server.adapter.out_channels
        )

        # reshape loss map and rare probabilities for visualization
        self.loss_map = tf.reshape(
            self.flat_loss_map,
            tf.shape(self.labels)
        )
        self.val_loss_map = tf.reshape(
            self.val_flat_loss_map,
            tf.shape(self.labels)
        )
        self.unrefined_probs = tf.slice(
            tf.nn.softmax(
                self.net.logits,
                axis=3
            ),
            [0,0,0,1],
            [-1,-1,-1,1]
        )
        self.refined_probs = tf.multiply(
            self.narrowed,
            self.unrefined_probs
        )
        self.val_refined_probs = tf.multiply(
            self.vnarrowed,
            self.unrefined_probs
        )

        # define the optimization operations
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.opt.minimize(self.loss)

        self.prediction_threshold = tf.placeholder_with_default(0.5, ())

        # define final predictions
        self.predictions = tf.cast(
            tf.greater(
                self.refined_probs,
                self.prediction_threshold
            ),
            tf.int64
        )
        # define final predictions
        self.val_predictions = tf.cast(
            tf.greater(
                self.val_refined_probs,
                0.5
            ),
            tf.int64
        )

        # define model performance
        (
            self.true_positives, self.false_positives, self.false_negatives,
            self.accuracy, self.precision, self.recall, self.f1, self.tversky,
            (self.tp_probs, self.tp_intensities),
            (self.fp_probs, self.fp_intensities),
            (self.fn_probs, self.fn_intensities)
        ) = layers.get_performance(
            self.predictions, self.labels, self.intensities,
            self.refined_probs, self.tversky_alpha
        )

        # define model performance on validation set
        (
            self.val_true_positives, self.val_false_positives,
            self.val_false_negatives, self.val_accuracy, self.val_precision,
            self.val_recall, self.val_f1, self.val_tversky,
            (self.val_tp_probs, self.val_tp_intensities),
            (self.val_fp_probs, self.val_fp_intensities),
            (self.val_fn_probs, self.val_fn_intensities)
        ) = layers.get_performance(
            self.val_predictions, self.labels, self.intensities,
            self.val_refined_probs, self.tversky_alpha
        )


    def _logits_to_prior_predictions(self, logits, threshold):
        """Take logits from model to true/false values as predictions from
        priors

        Assumes input takes 4d tensor shape
        """
        sm = tf.nn.softmax(logits, axis=3)
        return tf.greater(
            tf.slice(
                sm,
                [0,0,0,1],
                [-1,-1,-1,1]
            ),
            threshold
        )

    def _run_priors(self, priors, master_threshold=None):
        """Run input volume through each prior model

        Returns the aggregate predictions and the precision of the most recent
        prior as int64
        """
        if master_threshold is not None:
            thresh = master_threshold
        else:
            thresh = priors[0].prior_threshold

        predictions = self._logits_to_prior_predictions(
            priors[0].logits,
            thresh
        )
        for prior in priors[1:]:
            if master_threshold is not None:
                thresh = master_threshold
            else:
                thresh = prior.prior_threshold

            predictions = tf.logical_and(
                predictions,
                self._logits_to_prior_predictions(
                    prior.logits,
                    thresh
                )
            )
        return tf.cast(predictions, tf.int64)


    def _get_weights(self, narrowed, precision, labels, alpha, considering):
        """Get the weights for each pixel for the loss in this training batch

        Needs precision of most recent model and narrowed tensor (1s where
        rare class might still live, 0s elsewhere)
        """
        labels = tf.cast(labels, tf.float32)
        rare_mfweight = 0.5/(2*(1-alpha)*precision)
        comm_mfweight = 0.5/(1-2*(1-alpha)*precision)
        broad = tf.multiply(
            tf.add(
                tf.multiply(
                    labels,
                    rare_mfweight
                ),
                tf.multiply(
                    tf.subtract(1.0, labels),
                    comm_mfweight
                )
            ),
            tf.cast(considering, tf.float32)
        )
        return tf.multiply(
            narrowed, broad
        )

    def _get_weights_sampling(self, narrowed, precision, labels, alpha, considering):
        """Get the weights for each pixel for the loss in this training batch

        Needs precision of most recent model and narrowed tensor (1s where
        rare class might still live, 0s elsewhere)
        """
        labels = tf.cast(labels, tf.float32)
        rare_mfweight = 0.5/(2*(1-alpha)*precision)
        comm_mfweight = 0.5/(1-2*(1-alpha)*precision)

        rare_prob = rare_mfweight/(rare_mfweight + comm_mfweight)
        comm_prob = comm_mfweight/(rare_mfweight + comm_mfweight)

        rare_prob = rare_prob / max(rare_prob, comm_prob)
        comm_prob = comm_prob / max(rare_prob, comm_prob)

        print("RARE PROB:", rare_prob)
        print("COMM PROB:", comm_prob)


        probs = tf.multiply(
            tf.multiply(
                tf.add(
                    tf.multiply(
                        labels,
                        rare_prob
                    ),
                    tf.multiply(
                        tf.subtract(1.0, labels),
                        comm_prob
                    )
                ),
                tf.cast(considering, tf.float32)
            ),
            narrowed
        )

        return tf.cast(tf.less(
            tf.random_uniform(tf.shape(probs)), probs
        ), tf.float32)


    def create_tensorboard_summaries(self, logdir, sess, name):
        """Define and organize the things to be included in the tensorboard
        summary when merged is run

        Creates new placeholders for aggregated values, everything else is
        fed with the visualization set
        """

        self.ag_loss = tf.placeholder(tf.float32, shape=())
        self.ag_prec = tf.placeholder(tf.float32, shape=())
        self.ag_reca = tf.placeholder(tf.float32, shape=())
        self.ag_f1 = tf.placeholder(tf.float32, shape=())
        self.ag_tversky = tf.placeholder(tf.float32, shape=())

        self.tag_loss = tf.placeholder(tf.float32, shape=())
        self.tag_prec = tf.placeholder(tf.float32, shape=())
        self.tag_reca = tf.placeholder(tf.float32, shape=())
        self.tag_f1 = tf.placeholder(tf.float32, shape=())
        self.tag_tversky = tf.placeholder(tf.float32, shape=())

        # self.ag_tp_probs = tf.placeholder(tf.float32, shape=(None,))
        # self.ag_tp_intensities = tf.placeholder(tf.float32, shape=(None,))
        # self.ag_fp_probs = tf.placeholder(tf.float32, shape=(None,))
        # self.ag_fp_intensities = tf.placeholder(tf.float32, shape=(None,))
        # self.ag_fn_probs = tf.placeholder(tf.float32, shape=(None,))
        # self.ag_fn_intensities = tf.placeholder(tf.float32, shape=(None,))

        etime = time.time()
        with tf.name_scope('primary_measures'):
            tf.summary.scalar('ag_loss', self.ag_loss)
            tf.summary.scalar('tag_loss', self.tag_loss)
            tf.summary.scalar('ag_tversky', self.ag_tversky)
            tf.summary.scalar('tag_tversky', self.tag_tversky)
            tf.summary.scalar('ag_recall', self.ag_reca)
            tf.summary.scalar('tag_recall', self.tag_reca)
            tf.summary.scalar('ag_precision', self.ag_prec)
            tf.summary.scalar('tag_precision', self.tag_prec)
            tf.summary.scalar('ag_f1', self.ag_f1)
            tf.summary.scalar('tag_f1', self.tag_f1)
        with tf.name_scope('viz_set_measures'):
            tf.summary.scalar('loss', self.val_loss)
            tf.summary.scalar('f1', self.val_f1)
            tf.summary.scalar('tversky', self.val_tversky)
            tf.summary.scalar('accuracy', self.val_accuracy)
            tf.summary.scalar('precision', self.val_precision)
            tf.summary.scalar('recall', self.val_recall)
        with tf.name_scope('variable_distributions'):
            for key, val in self.net.variables.items():
                tf.summary.histogram(key, val)
        # with tf.name_scope('predicted_probabilities'):
        #     tf.summary.histogram("tp_probs", self.ag_tp_probs)
        #     tf.summary.histogram("fp_probs", self.ag_fp_probs)
        #     tf.summary.histogram("fn_probs", self.ag_fn_probs)
        # with tf.name_scope('intensities_by_prediction'):
        #     tf.summary.histogram("tp_intensities", self.ag_tp_intensities)
        #     tf.summary.histogram("fp_intensities", self.ag_fp_intensities)
        #     tf.summary.histogram("fn_intensities", self.ag_fn_intensities)
        # with tf.name_scope('gradient_distributions'):
        #     for key, val in self.gradients.iteritems():
        #         tf.summary.histogram(key, val)
        #         tf.summary.scalar(
        #             key + '_avg_magnitude',
        #             tf.reduce_mean(tf.abs(val))
        #         )
        with tf.name_scope('pixel-wise_objectives'):
            tf.summary.image('loss_map',
                self.val_loss_map,
                max_outputs=20)
        with tf.name_scope('pixel-wise_predictions'):
            tf.summary.image('prediction_map',
                self.val_refined_probs,
                max_outputs=20)
        with tf.name_scope('prior_narrowed'):
            tf.summary.image('vnarrowed',
                self.vnarrowed,
                max_outputs=20)

        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(
            logdir + "/" + name + str(etime),
            sess.graph
        )


    def train_step(self, sess, X, Y):
        """A single iteration of the training algorithm

        Feed-forward through all networks, calculate appropriate cost for the
        ensemble, and modify the trainable parameters
        """
        sess.run((self.train_op), feed_dict={
            self.input_placeholder: X,
            self.labels_placeholder: Y
        })


    def _new_best(self, j, ag_f1, sess, meta):
        """Compute necessary information about model for using it as a prior

        Save ag_precision (validation set)
        """
        self.net.save(j, ag_f1, sess, meta)

    def test_step(self, j, sess, batch_size, save_threshold):

        (
            ag_loss, ag_prec, ag_reca, ag_f1, ag_tversky
            # , tot_tpp, tot_tpi, tot_fpp, tot_fpi, tot_fnp, tot_fni
        ) = self.tester.val_step(
            self.input_placeholder, self.labels_placeholder,
            self.net.keep_prob, self.tversky_alpha, (
                self.val_true_positives, self.val_false_positives,
                self.val_false_negatives, self.val_loss
                # , self.val_tp_probs, self.val_tp_intensities,
                # self.val_fp_probs, self.val_fp_intensities,
                # self.val_fn_probs, self.val_fn_intensities
            ), sess, batch_size
        )


        (
            tag_loss, tag_prec, tag_reca, tag_f1, tag_tversky
            # , tot_tpp, tot_tpi, tot_fpp, tot_fpi, tot_fnp, tot_fni
        ) = self.tester.test_step(
            self.input_placeholder, self.labels_placeholder,
            self.net.keep_prob, self.tversky_alpha, (
                self.val_true_positives, self.val_false_positives,
                self.val_false_negatives, self.val_loss
                # , self.val_tp_probs, self.val_tp_intensities,
                # self.val_fp_probs, self.val_fp_intensities,
                # self.val_fn_probs, self.val_fn_intensities
            ), sess, batch_size
        )

        # Run visualization data and record event
        X, Y = self.server.get_viz_batch()
        self._tensorboard_record_event(sess, j, (), {
                self.input_placeholder: X,
                self.labels_placeholder: Y,
                self.net.keep_prob: 1.0,
                self.ag_loss: ag_loss,
                self.ag_prec: ag_prec,
                self.ag_reca: ag_reca,
                self.ag_f1: ag_f1,
                self.ag_tversky: ag_tversky,
                self.tag_loss: tag_loss,
                self.tag_prec: tag_prec,
                self.tag_reca: tag_reca,
                self.tag_f1: tag_f1,
                self.tag_tversky: tag_tversky
                # , self.ag_tp_probs: tot_tpp,
                # self.ag_tp_intensities: tot_tpi,
                # self.ag_fp_probs: tot_fpp,
                # self.ag_fp_intensities: tot_fpi,
                # self.ag_fn_probs: tot_fnp,
                # self.ag_fn_intensities: tot_fni
            }
        )
        if (ag_tversky > save_threshold):
            print("saving model")
            save_threshold = ag_tversky
            meta = {}
            meta["precision"] = ag_prec
            self._new_best(j, ag_tversky, sess, meta)

        return ag_loss, ag_f1, ag_tversky, save_threshold



    def train(self, sess, epochs, batch_size, display_step, save_threshold,
              init_i):
        """Iteratively calls train_step and occasionally reports validation
        performance

        Communicates with server to get the next mini-batch for training.
        """
        # self.test_step(
        #     init_i, sess, batch_size, save_threshold
        # )
        # print(init_i, l, f)
        for j in range(init_i+1, init_i+epochs+1):
            X, Y = self.server.next_training_batch(j-1, batch_size)
            self.train_step(sess, X, Y)
            print(j)
            if (j % display_step == 0):
                l, f, t, save_threshold = self.test_step(
                    j, sess, batch_size, save_threshold
                )
                print(j, l, f, t)

    def _tensorboard_record_event(self, sess, step, run, feed_dict={}):
        """Run the "merged_summary" tensor to dump the information into
        tensorboard

        Takes aggregates and visualization data in the feed_dict
        """
        summary, rest = sess.run(
            (self.merged_summary, run), feed_dict=feed_dict
        )
        self.summary_writer.add_summary(summary, step)
        return rest
