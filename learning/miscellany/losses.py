import numpy as np
import tensorflow as tf

EPS = 1e-5

def weighted_pixelwise_softmax(y, out, weights, num_classes):
    """Pixel-wise softmax weighted by given tensor

    Typically used for median-frequency weighting along with ignoring pixels
    already eliminated from contention
    """

    flat_sparse_labels = tf.reshape(y, [-1])
    flat_logits = tf.reshape(out, [-1, num_classes])
    flat_weights = tf.reshape(weights, [-1,1])

    unweighted_costs = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = flat_sparse_labels,
        logits = flat_logits
    ), [-1, 1])

    costs = tf.multiply(flat_weights, unweighted_costs)

    return tf.reduce_mean(costs), unweighted_costs
