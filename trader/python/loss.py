#!python3
import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class SparseCategoricalHighestConfidence(SparseCategoricalAccuracy):
    """
    Calculates per batch accuracy based on the following

        1 if argmax(highest confidence result) == y_true
        0 otherwise
    """

    def __init__(self, name='highest_confidence', dtype=None):
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):

        index = tf.argmax(tf.math.reduce_max(y_pred, axis=-1))
        y_true = tf.gather(y_true, index)
        y_pred = tf.gather(y_pred, index)

        return super().update_state(y_true, y_pred, sample_weight)
