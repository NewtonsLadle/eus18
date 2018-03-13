import time
import numpy as np
import tensorflow as tf
from pathlib import Path

import miscellany.layers as layers
from miscellany.netmanager import NetManager

def _restore_assignment(key, rest1, rest2, rest3, default, trainable=True):
    """Assign an initial value and some properties to a tf variable

    First checks primary dict for the key, if not found checks secondary dict,
    if still not found, initializes with given default value. All types/shapes
    must agree
    """
    if (key in rest1):
        print("PRIMARY", key)
        return tf.Variable(
            tf.constant(rest1[key], dtype=tf.float32),
            name=key,
            trainable=trainable
        )
    elif (key in rest2):
        print("SECONDARY", key)
        return tf.Variable(
            tf.constant(rest2[key], dtype=tf.float32),
            name=key,
            trainable=trainable
        )
    elif (key in rest3):
        print("TERTIARY", key)
        return tf.Variable(
            tf.constant(rest3[key], dtype=tf.float32),
            name=key,
            trainable=trainable
        )
    else:
        print("DEFAULT", key)
        return default


class UNet(object):
    """Encapsulates the feed-forward network

    This architecture is meant to closely resemble the 2015 UNet paper by
    Ronneberg et al.

    This object defines the computations to take input data to logits. Interpreting
    these logits as predictions needs to be done by another managing environment.

    This object is meant to be part of a training pipeline. The trainable argument
    to the constructor defines whether the parameters of each instance are meant
    to be trained or not.
    """


    # Input comes as 4d tensor
    def __init__(self, in_channels, out_channels, levels, csize, psize,
                 initial_filters, keep_prob, input_placeholder, trainable,
                 save_root, primary_restore_path, secondary_restore_path,
                 prepping_prior=False):
        """ Construct the UNet object

        Input data (probably placeholders) must be defined elsewhere and fed as
        an argument.
        """
        # This is a placeholder specific to the net and none of the priors
        #  (priors will all have keep_prob=1.0)
        self.keep_prob = keep_prob

        # Network hyperparameters
        self.levels = levels
        self.csize = csize
        self.psize = psize
        self.initial_filters = initial_filters

        # Details about input data/ground truth that affects architecture
        self.in_channels = in_channels
        self.out_channels = out_channels

        print("Constructing UNet with %d input channels" % self.in_channels)


        # Whether these parameters should be optimized
        self.trainable = trainable

        # Operating system interface for saving and restoring
        self.net_manager = NetManager(
            primary_restore_path,
            secondary_restore_path,
            save_root
        )

        # Initial values for the net's parameters
        self.primary_restore_dict = self.net_manager.primary_restore_dict
        self.secondary_restore_dict = self.net_manager.secondary_restore_dict
        self.init_dict = self._get_init_dict()

        # Define the computation graph taking input to logits
        self.logits, self.variables = self._build_ff_graph(
            input_placeholder,
            keep_prob,
            self.primary_restore_dict,
            self.secondary_restore_dict,
            self.init_dict
        )

        if (("newprecision" in self.primary_restore_dict) and (not prepping_prior)):
            self.precision = float(self.primary_restore_dict["newprecision"])
        elif "precision" in self.primary_restore_dict:
            self.precision = float(self.primary_restore_dict["precision"])
        else:
            self.precision = None

        if "threshold" in self.primary_restore_dict:
            self.prior_threshold = float(self.primary_restore_dict["threshold"])
        else:
            self.prior_threshold = None
        # self.prior_threshold = 0.125



    def _get_ups_init(self, fltrs):
        """Returns hand-picked upsampling filter to undo pooling

        Might make these untrainable in the future
        """
        doub = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32
        )
        ups = np.stack([doub for j in range(0,fltrs)], axis=2)
        ups = np.stack([ups for j in range(0,fltrs*2)], axis=3)
        return ups


    def _get_init_dict(self):
        """Returns dictionary with hand-picked upsampling filters where appropriate

        In other models, this might contain other "good ideas" to start out with
        """
        dicti = {}
        for i in np.arange(self.levels-1, 0, -1):
            fltrs = self.initial_filters*(2**(i-1))
            ups_key = 'w_l' + str(i) + 'u_d'
            dicti[ups_key] = self._get_ups_init(fltrs)
        return dicti


    def feed_forward(self):
        """Simply returns logits

        More of a sanity preserver than anything else. Allows computation graph
        to be "run" in a more natural way in a management environment
        """
        return self.logits


    def _build_ff_graph(self, input_volume, keep_prob, rest1, rest2, rest3):
        """Defines computation graph taking input data to logits

        Initializes variables from restore dicts.

        Returns logits (tf tensor) as well as a dict of variables (also tf
        tensors)
        """
        variables = {}
        bridge_activations = {}

        input_features = self.in_channels
        n = self.levels

        for i in range(1, n+1):
            print(i)
            fltrs = self.initial_filters*(2**(i-1))
            print(fltrs)

            # variable shapes
            w1_shp = [self.csize, self.csize, input_features, fltrs]
            b1_shp = [fltrs]
            g1_shp = [fltrs]
            e1_shp = [fltrs]
            w2_shp = [self.csize, self.csize, fltrs, fltrs]
            b2_shp = [fltrs]
            g2_shp = [fltrs]
            e2_shp = [fltrs]

            # variables
            w1 = _restore_assignment(
                'w_l' + str(i) + 'd_1',
                rest1,
                rest2,
                rest3,
                layers.tn_variable(w1_shp),
                trainable=self.trainable
            )
            b1 = _restore_assignment(
                'b_l' + str(i) + 'd_1',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=b1_shp)),
                trainable=self.trainable
            )
            g1 = _restore_assignment(
                'g_l' + str(i) + 'd_1',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=g1_shp)),
                trainable=self.trainable
            )
            e1 = _restore_assignment(
                'e_l' + str(i) + 'd_1',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=e1_shp)),
                trainable=self.trainable
            )
            w2 = _restore_assignment(
                'w_l' + str(i) + 'd_2',
                rest1,
                rest2,
                rest3,
                layers.tn_variable(w2_shp),
                trainable=self.trainable
            )
            b2 = _restore_assignment(
                'b_l' + str(i) + 'd_2',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=b2_shp)),
                trainable=self.trainable
            )
            g2 = _restore_assignment(
                'g_l' + str(i) + 'd_2',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=g2_shp)),
                trainable=self.trainable
            )
            e2 = _restore_assignment(
                'e_l' + str(i) + 'd_2',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=e2_shp)),
                trainable=self.trainable
            )
            # Set input features for next time to fltrs this time
            input_features = fltrs

            # Add to dict
            variables['w_l' + str(i) + 'd_1'] = w1
            variables['b_l' + str(i) + 'd_1'] = b1
            variables['g_l' + str(i) + 'd_1'] = g1
            variables['e_l' + str(i) + 'd_1'] = e1
            variables['w_l' + str(i) + 'd_2'] = w2
            variables['b_l' + str(i) + 'd_2'] = b2
            variables['g_l' + str(i) + 'd_2'] = g2
            variables['e_l' + str(i) + 'd_2'] = e2

            # Computations
            h1 = layers.conv_relu_bn(input_volume, w1, b1, g1, e1, self.keep_prob)
            h2 = layers.conv_relu_bn(h1, w2, b2, g2, e2, self.keep_prob)
            if (i != n):
                bridge_activations[str(i)] = h2
                input_volume = layers.pool(h2)
            else:
                input_volume = h2


        for i in np.arange(n-1, 0, -1):
            fltrs = self.initial_filters*(2**(i-1))

            # variable shapes
            wd_shp = [self.psize, self.psize, fltrs, input_features]
            bd_shp = [fltrs]
            gd_shp = [fltrs]
            ed_shp = [fltrs]
            w1_shp = [self.csize, self.csize, 2*fltrs, fltrs]
            b1_shp = [fltrs]
            g1_shp = [fltrs]
            e1_shp = [fltrs]
            w2_shp = [self.csize, self.csize, fltrs, fltrs]
            b2_shp = [fltrs]
            g2_shp = [fltrs]
            e2_shp = [fltrs]

            # variables
            wd = _restore_assignment(
                'w_l' + str(i) + 'u_d',
                rest1,
                rest2,
                rest3,
                layers.tn_variable(wd_shp),
                trainable=self.trainable
            )
            bd = _restore_assignment(
                'b_l' + str(i) + 'u_d',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=bd_shp)),
                trainable=self.trainable
            )
            gd = _restore_assignment(
                'g_l' + str(i) + 'u_d',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=gd_shp)),
                trainable=self.trainable
            )
            ed = _restore_assignment(
                'e_l' + str(i) + 'u_d',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=ed_shp)),
                trainable=self.trainable
            )
            w1 = _restore_assignment(
                'w_l' + str(i) + 'u_1',
                rest1,
                rest2,
                rest3,
                layers.tn_variable(w1_shp),
                trainable=self.trainable
            )
            b1 = _restore_assignment(
                'b_l' + str(i) + 'u_1',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=b1_shp)),
                trainable=self.trainable
            )
            g1 = _restore_assignment(
                'g_l' + str(i) + 'u_1',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=g1_shp)),
                trainable=self.trainable
            )
            e1 = _restore_assignment(
                'e_l' + str(i) + 'u_1',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=e1_shp)),
                trainable=self.trainable
            )
            w2 = _restore_assignment(
                'w_l' + str(i) + 'u_2',
                rest1,
                rest2,
                rest3,
                layers.tn_variable(w2_shp),
                trainable=self.trainable
            )
            b2 = _restore_assignment(
                'b_l' + str(i) + 'u_2',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=b2_shp)),
                trainable=self.trainable
            )
            g2 = _restore_assignment(
                'g_l' + str(i) + 'u_2',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=g2_shp)),
                trainable=self.trainable
            )
            e2 = _restore_assignment(
                'e_l' + str(i) + 'u_2',
                rest1,
                rest2,
                rest3,
                tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=e2_shp)),
                trainable=self.trainable
            )

            # Set input features for next time to fltrs this time
            input_features = fltrs

            # Add to dict
            variables['w_l' + str(i) + 'u_d'] = wd
            variables['b_l' + str(i) + 'u_d'] = bd
            variables['g_l' + str(i) + 'u_d'] = gd
            variables['e_l' + str(i) + 'u_d'] = ed
            variables['w_l' + str(i) + 'u_1'] = w1
            variables['b_l' + str(i) + 'u_1'] = b1
            variables['g_l' + str(i) + 'u_1'] = g1
            variables['e_l' + str(i) + 'u_1'] = e1
            variables['w_l' + str(i) + 'u_2'] = w2
            variables['b_l' + str(i) + 'u_2'] = b2
            variables['g_l' + str(i) + 'u_2'] = g2
            variables['e_l' + str(i) + 'u_2'] = e2

            # Computations
            hd = layers.convt_relu_bn(input_volume, wd, bd, gd, ed, self.keep_prob, tf.shape(bridge_activations[str(i)]))
            c = layers.concat(bridge_activations[str(i)], hd)
            h1 = layers.conv_relu_bn(c, w1, b1, g1, e1, self.keep_prob)
            h2 = layers.conv_relu_bn(h1, w2, b2, g2, e2, self.keep_prob)
            input_volume = h2

        # Output layer
        wo_shp = [1, 1, input_features, self.out_channels]
        bo_shp = [self.out_channels]

        wo = _restore_assignment(
            'wo',
            rest1,
            rest2,
            rest3,
            layers.tn_variable(wo_shp),
            trainable=self.trainable
        )
        bo = _restore_assignment(
            'bo',
            rest1,
            rest2,
            rest3,
            tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=bo_shp)),
            trainable=self.trainable
        )

        variables['wo'] = wo
        variables['bo'] = bo

        h_out = layers.conv(input_volume, wo) + bo

        return h_out, variables

    # save model to disk
    # takes location as a path object, must exist
    def save(self, j, measure, sess, meta_dict={}):
        """Dump variables into individual .npy files on disk

        Particulary useful for partial restoration
        """
        vardict = sess.run((self.variables))
        self.net_manager.save_model(vardict, meta_dict, j, measure)
