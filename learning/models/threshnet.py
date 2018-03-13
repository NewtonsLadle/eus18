import time
import numpy as np
import tensorflow as tf
from pathlib import Path

import miscellany.layers as layers
from miscellany.netmanager import NetManager

class ThreshNet(object):
    """Encapsulates the feed-forward network

    Predicts positive for all voxels above floor
    """


    # Input comes as 4d tensor
    def __init__(self, input_placeholder, floor, ceiling, kernel_size, server):
        """ Construct the ThreshNet object

        Predicts positive for all voxels above floor
        """
        self.threshold = tf.Variable(floor, trainable=False)
        #TODO handle ceiling
        self.precision = (1-server.trivial_prob)*server.adapter.frequencies[1]
        self.prior_threshold = 0.5

        self.ksize = kernel_size
        self.floor = floor

        self.logits = self._get_thresh_logits(input_placeholder)


    def _get_thresh_logits(self, input_placeholder):
        """Run blur and threshold to remove impossible voxels from consideration
        """
        kernel = tf.constant((1/(self.ksize**2))*np.ones((self.ksize, self.ksize, 1, 1)), tf.float32)

        probs = tf.where(
            tf.greater(
                tf.nn.conv2d(
                    input_placeholder,
                    kernel,
                    strides=[1,1,1,1],
                    padding='SAME'
                ),
                self.floor
            ),
            x=tf.ones(tf.shape(input_placeholder)),
            y=tf.zeros(tf.shape(input_placeholder))
        )

        return tf.cast(
            tf.concat(
                (
                    tf.subtract(tf.ones(tf.shape(input_placeholder)), probs),
                    probs
                ),
                axis=3
            ),
            tf.float32
        )
