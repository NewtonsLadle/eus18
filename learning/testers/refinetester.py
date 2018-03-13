import numpy as np
import tensorflow as tf

import miscellany.layers as layers

EPS = 1e-5


class RefineTester(object):
    """Manages validation and testing set prediction during training, along with
    threshold and new precision computation when preparing a prior"""

    def __init__(self, server):
        """Build the tester"""
        self.server = server


    def _get_precision(self, probs, threshold, truth):
        """Compute precision of given probabilities with given threshold and
        ground truth"""
        tp = np.sum(
            np.greater(
                np.add(
                    probs,
                    truth
                ),
                1.0+threshold
            ).astype(np.float32)
        )
        fp = np.sum(
            np.logical_and(
                np.greater(
                    probs,
                    threshold
                ),
                np.less(
                    truth,
                    0.5
                )
            ).astype(np.float32)
        )

        return tp/(tp + fp)


    def _get_threshold(self, tot_probs, tot_y, precision):
        """Get new threshold for prior to use on training set during training
        of the next model"""
        limit = 20
        lb = 0.0
        ub = 1.0
        this_thresh = 0.5
        residual = self._get_precision(tot_probs, this_thresh, tot_y) - precision
        i = 0
        print("Target: ", precision)
        while ((np.abs(residual) > 1e-3) and (i < limit)):
            this_thresh = (ub + lb)/2
            print("Off by %f at threshold = %f" % (residual, this_thresh))
            if residual > 0:
                ub = this_thresh
            else:
                lb = this_thresh

            residual = self._get_precision(tot_probs, this_thresh, tot_y) - precision
            i = i + 1

        if (i >= limit):
            threshold = lb
        else:
            threshold = this_thresh

        new_precision = self._get_precision(tot_probs, lb, tot_y)
        return threshold, new_precision


    def _remove_twos(self, Y):
        """If anything in the ground thruth is greater than 1.5,
        set it to zero"""
        return np.multiply(
            np.less(Y, 1.5).astype(np.float32),
            Y
        )


    def get_prior_precision(self, sess, num_bundles, input_placeholder,
                            labels_placeholder, keep_prob_placeholder,
                            threshold_placeholder, numtp, numfp, target):
        """Computes new precision and threshold when using this model as a
        prior"""
        print("Target is %f" % target)
        inc = 0.5
        residual = 1.0
        threshold = 0.5
        totprec = 0
        numiters = 25
        for i in range(0, numiters):
            threshold = min(threshold/(1.0+inc*(-1)**(1+int(residual > 0))), 0.5)
            inc = inc / 1.25
            ag_tp = 0
            ag_fp = 0
            for i in range(0, num_bundles):
                X, Y = self.server.random_training_bundle()
                tp, fp = sess.run((numtp, numfp),
                    feed_dict={
                        input_placeholder: X,
                        labels_placeholder: Y,
                        keep_prob_placeholder: 1.0,
                        threshold_placeholder: threshold
                    }
                )
                ag_tp = ag_tp + tp
                ag_fp = ag_fp + fp

            precision = ag_tp/(ag_tp + ag_fp)
            residual = precision - target
            totprec = totprec + precision
            print("Precision at %f is %f" % (threshold, precision))


        newprecision = totprec/numiters
        print("New precision:", newprecision)

        return threshold, newprecision



    def val_step(self, inputs, labels, keep_prob, tversky_alpha,
                  to_run, sess, batch_size):
        """A function to get an unbiased measure of the performance of the
        model on a hold-out set

        Whole hold-out set is not fed at once for memory reasons, so things are
        fed iteratively and aggregated
        """
        prop_of_triv = 0.1
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        tot_loss = 0
        tot_batches = 0
        divisor = 0.0
        # tot_tpp = np.zeros((0,), np.float32)
        # tot_tpi = np.zeros((0,), np.float32)
        # tot_fpp = np.zeros((0,), np.float32)
        # tot_fpi = np.zeros((0,), np.float32)
        # tot_fnp = np.zeros((0,), np.float32)
        # tot_fni = np.zeros((0,), np.float32)
        more_exists = True
        while more_exists:
            tot_batches = tot_batches + 1
            if (tot_batches*batch_size > self.server.ntl_val_slices):
                multiplier = 1.0/prop_of_triv
            else:
                multiplier = 1.0
            X, Y, more_exists = self.server.next_validation_batch(batch_size, prop_of_triv)
            tp, fp, fn, loss = sess.run(to_run,
                feed_dict={
                    inputs: X,
                    labels: Y,
                    keep_prob: 1.0
                }
            )
            tot_tp = tot_tp + multiplier*tp
            tot_fp = tot_fp + multiplier*fp
            tot_fn = tot_fn + multiplier*fn
            tot_loss = tot_loss + multiplier*loss
            # tot_tpp = np.concatenate((tot_tpp, layers.remove_wherenot(tpp, 1e-5)))
            # tot_tpi = np.concatenate((tot_tpi, layers.remove_wherenot(tpi)))
            # tot_fpp = np.concatenate((tot_fpp, layers.remove_wherenot(fpp, 1e-5)))
            # tot_fpi = np.concatenate((tot_fpi, layers.remove_wherenot(fpi)))
            # tot_fnp = np.concatenate((tot_fnp, layers.remove_wherenot(fnp, 1e-5)))
            # tot_fni = np.concatenate((tot_fni, layers.remove_wherenot(fni)))
            divisor = divisor + multiplier

        ag_loss = tot_loss/divisor
        ag_prec = tot_tp/(tot_tp + tot_fp + EPS)
        ag_reca = tot_tp/(tot_tp + tot_fn + EPS)
        ag_f1 = 2.0*ag_prec*ag_reca/(ag_prec + ag_reca + EPS)
        ag_tversky = (
            ag_prec*ag_reca
            /(
                tversky_alpha*ag_prec
                + (1.0 - tversky_alpha)*ag_reca
                + EPS
            )
        )


        return (
            ag_loss, ag_prec, ag_reca, ag_f1, ag_tversky
            #, tot_tpp, tot_tpi, tot_fpp, tot_fpi, tot_fnp, tot_fni
        )


    def test_step(self, inputs, labels, keep_prob, tversky_alpha,
                  to_run, sess, batch_size):
        """A function to get an unbiased measure of the performance of the
        model on a hold-out set

        Whole hold-out set is not fed at once for memory reasons, so things are
        fed iteratively and aggregated
        """
        prop_of_triv = 0.1
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        tot_loss = 0
        tot_batches = 0
        divisor = 0.0
        # tot_tpp = np.zeros((0,), np.float32)
        # tot_tpi = np.zeros((0,), np.float32)
        # tot_fpp = np.zeros((0,), np.float32)
        # tot_fpi = np.zeros((0,), np.float32)
        # tot_fnp = np.zeros((0,), np.float32)
        # tot_fni = np.zeros((0,), np.float32)
        more_exists = True
        while more_exists:
            tot_batches = tot_batches + 1
            if (tot_batches*batch_size > self.server.ntl_tst_slices):
                multiplier = 1.0/prop_of_triv
            else:
                multiplier = 1.0
            X, Y, more_exists = self.server.next_testing_batch(batch_size, prop_of_triv)
            tp, fp, fn, loss = sess.run(to_run,
                feed_dict={
                    inputs: X,
                    labels: Y,
                    keep_prob: 1.0
                }
            )
            tot_tp = tot_tp + multiplier*tp
            tot_fp = tot_fp + multiplier*fp
            tot_fn = tot_fn + multiplier*fn
            tot_loss = tot_loss + multiplier*loss
            # tot_tpp = np.concatenate((tot_tpp, layers.remove_wherenot(tpp, 1e-5)))
            # tot_tpi = np.concatenate((tot_tpi, layers.remove_wherenot(tpi)))
            # tot_fpp = np.concatenate((tot_fpp, layers.remove_wherenot(fpp, 1e-5)))
            # tot_fpi = np.concatenate((tot_fpi, layers.remove_wherenot(fpi)))
            # tot_fnp = np.concatenate((tot_fnp, layers.remove_wherenot(fnp, 1e-5)))
            # tot_fni = np.concatenate((tot_fni, layers.remove_wherenot(fni)))
            divisor = divisor + multiplier

        ag_loss = tot_loss/divisor
        ag_prec = tot_tp/(tot_tp + tot_fp + EPS)
        ag_reca = tot_tp/(tot_tp + tot_fn + EPS)
        ag_f1 = 2.0*ag_prec*ag_reca/(ag_prec + ag_reca + EPS)
        ag_tversky = (
            ag_prec*ag_reca
            /(
                tversky_alpha*ag_prec
                + (1.0 - tversky_alpha)*ag_reca
                + EPS
            )
        )


        return (
            ag_loss, ag_prec, ag_reca, ag_f1, ag_tversky
            #, tot_tpp, tot_tpi, tot_fpp, tot_fpi, tot_fnp, tot_fni
        )
