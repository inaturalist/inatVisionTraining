import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


# most of this is from the keras implementation of
# sparse_categorical_accuracy, just modified to measure
# accuracy on the parent label.


def make_sparse_parent_accuracy_metric(parent_labels):
    lookup_table = tf.constant(parent_labels)

    def sparse_parent_accuracy_metric(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        y_pred_rank = y_pred.shape.ndims
        y_true_rank = y_true.shape.ndims
        if (
            (y_true_rank is not None)
            and (y_pred_rank is not None)
            and (len(K.int_shape(y_true)) == len(K.int_shape(y_pred)))
        ):
            y_true = tf.squeeze(y_true, [-1])
        y_pred = tf.compat.v1.argmax(y_pred, axis=-1)

        if K.dtype(y_pred) != K.dtype(y_true):
            y_pred = tf.cast(y_pred, K.dtype(y_true))

        y_parents = tf.gather(lookup_table, tf.cast(y_true, np.int32))
        yhat_parents = tf.gather(lookup_table, tf.cast(y_pred, np.int32))

        return tf.cast(tf.equal(y_parents, yhat_parents), K.floatx())

    return sparse_parent_accuracy_metric


# most of this is from the keras implementation of
# sparse_categorical_accuracy, just modified to measure
# accuracy on the parent label.


def make_parent_accuracy_metric(parent_labels):
    lookup_table = tf.constant(parent_labels)

    def parent_accuracy_metric(y_true, y_pred):
        # convert from dense to sparse encoding
        # TODO - this implementation won't work with label smoothing
        y_true = tf.argmax(y_true, 1)

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true)
        y_pred_rank = y_pred.shape.ndims
        y_true_rank = y_true.shape.ndims
        if (
            (y_true_rank is not None)
            and (y_pred_rank is not None)
            and (len(K.int_shape(y_true)) == len(K.int_shape(y_pred)))
        ):
            y_true = tf.squeeze(y_true, [-1])
        y_pred = tf.compat.v1.argmax(y_pred, axis=-1)

        y_parents = tf.gather(lookup_table, tf.cast(y_true, np.int32))
        yhat_parents = tf.gather(lookup_table, tf.cast(y_pred, np.int32))

        return tf.cast(tf.equal(y_parents, yhat_parents), K.floatx())

    return parent_accuracy_metric
