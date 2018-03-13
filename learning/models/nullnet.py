import time
import numpy as np
import tensorflow as tf
from pathlib import Path

import miscellany.layers as layers
from miscellany.netmanager import NetManager

class NullNet(object):
    """Encapsulates the feed-forward network

    Predicts the rare class everytime. For code clarity later on in pipeline
    """


    # Input comes as 4d tensor
    def __init__(self, input_placeholder, server, trivial_prob):
        """ Construct the NullNet object

        Input data (probably placeholders) must be defined elsewhere and fed as
        an argument.
        Always predicts rare class and has adapter defined precision
        """
        self.logits = self._get_null_logits(input_placeholder)
        self.precision = (1-trivial_prob)*server.adapter.frequencies[1]
        self.prior_threshold = 0.5

    def _get_null_logits(self, input_placeholder):
        intensities = tf.slice(
            input_placeholder,
            [0,0,0,0],
            [-1,-1,-1,1]
        )
        return tf.concat(
            (
                tf.zeros(tf.shape(intensities)),
                tf.ones(tf.shape(intensities))
            ),
            axis=3
        )
