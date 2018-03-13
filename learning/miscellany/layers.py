from __future__ import division
import tensorflow as tf
import numpy as np

EPS = 1e-5

def _get_output_shape(in_shp, strides, krnl_shp):
    """Get output shape after convolution transpose

    Assumes padding=SAME during convolutions
    """
    spatial_shp = [
        strides[1]*in_shp[1],
        strides[2]*in_shp[2],
    ]
    return tf.cast(
        (
            in_shp[0],
            spatial_shp[0],
            spatial_shp[1],
            krnl_shp[2]
        ),
        tf.int32
    )


def tn_variable(shape=(), stddev=0.1):
    """Get a truncated normal variable with given shape and standard deviation

    https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def conv(input_tensor, w):
    """A 2d Convolution of stride 1

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    """
    return tf.nn.conv2d(
        input_tensor, w, strides=[1,1,1,1], padding='SAME'
    )


def convt(input_tensor, w, out_shape):
    """A 2d Convolution transpose of stride 2 (undoes pooling)

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose
    """
    strides = [1,2,2,1]
    # out_shape = _get_output_shape(
    #     tf.shape(input_tensor), strides, tf.shape(w)
    # )

    return tf.nn.conv2d_transpose(
        input_tensor, w,
        output_shape=out_shape, strides=strides,
        padding='VALID'
    )


def conv_relu_bn(input_tensor, w, b, g, e, keep_prob=1.0):
    """A 2d convolution and a rectified linear unit with batch normalization and
    dropout

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    https://www.tensorflow.org/api_docs/python/tf/nn/relu
    https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
    https://www.tensorflow.org/api_docs/python/tf/nn/dropout
    """
    activations = conv(input_tensor, w) + b
    mean, variance = tf.nn.moments(activations, [0, 1, 2])

    return tf.nn.dropout(
        tf.nn.relu(
            tf.nn.batch_normalization(
                activations,
                mean,
                variance,
                e,
                g,
                EPS
            )
        ),
        keep_prob
    )


def convt_relu_bn(input_tensor, w, b, g, e, keep_prob=1.0, output_shape=None):
    """A 2d convolution transpose and a rectified linear unit with
    batch normalization and dropout

    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose
    https://www.tensorflow.org/api_docs/python/tf/nn/relu
    https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
    https://www.tensorflow.org/api_docs/python/tf/nn/dropout
    """
    activations = convt(input_tensor, w, output_shape) + b
    mean, variance = tf.nn.moments(activations, [0, 1, 2])

    return tf.nn.dropout(
        tf.nn.relu(
            tf.nn.batch_normalization(
                activations,
                mean,
                variance,
                e,
                g,
                EPS
            )
        ),
        keep_prob
    )


def pool(input_tensor):
    """Standard max_pooling in two dimension and no overlap

    https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    """
    return tf.nn.max_pool(input_tensor, [1,2,2,1], [1,2,2,1], padding='VALID')


def concat(left_side, right_side):
    """Concatenate the left side to the right size along the 3rd axis

    https://www.tensorflow.org/api_docs/python/tf/concat
    """
    return tf.concat((left_side, right_side), axis=3)


def remove_wherenot(arr, thresh=-9999.0):
    """To be computed on cpu, not in tf computation graph

    Return flattened array of values that came from x in below tf.where() call
    """
    arr = np.reshape(arr, (-1,))
    return arr[np.greater(arr, thresh)]


def get_performance(predicted, truth, intensities, probabilities, alpha=0.5):
    """Compute performance measures for these predictions

    alpa corresponds to weighting precision in tversky score
    """
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                predicted,
                truth
            ),
            tf.float32
        )
    )

    zeros = -10000.0*tf.ones(tf.shape(intensities), dtype=tf.float32)

    true_positives = tf.logical_and(
        tf.equal(predicted, 1),
        tf.equal(truth, 1)
    )
    tp_probs = tf.where(true_positives, x=probabilities, y=zeros)
    tp_intensities = tf.where(true_positives, x=intensities, y=zeros)
    num_true_positives = tf.reduce_sum(
        tf.cast(
            true_positives,
            tf.float32
        )
    )

    false_positives = tf.logical_and(
        tf.equal(predicted, 1),
        tf.equal(truth, 0)
    )
    fp_probs = tf.where(false_positives, x=probabilities, y=zeros)
    fp_intensities = tf.where(false_positives, x=intensities, y=zeros)
    num_false_positives = tf.reduce_sum(
        tf.cast(
            false_positives,
            tf.float32
        )
    )

    false_negatives = tf.logical_and(
        tf.equal(predicted, 0),
        tf.equal(truth, 1)
    )
    fn_probs = tf.where(false_negatives, x=probabilities, y=zeros)
    fn_intensities = tf.where(false_negatives, x=intensities, y=zeros)
    num_false_negatives = tf.reduce_sum(
        tf.cast(
            false_negatives,
            tf.float32
        )
    )

    precision = num_true_positives/(num_true_positives+num_false_positives+EPS)
    recall = num_true_positives/(num_true_positives+num_false_negatives+EPS)
    f1 = 2*precision*recall/(precision + recall + EPS)
    tversky = precision*recall/(alpha*precision + (1.0-alpha)*recall + EPS)

    return (
        num_true_positives,
        num_false_positives,
        num_false_negatives,
        accuracy,
        precision,
        recall,
        f1,
        tversky,
        (tp_probs, tp_intensities),
        (fp_probs, fp_intensities),
        (fn_probs, fn_intensities)
    )
