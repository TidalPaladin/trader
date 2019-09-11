#!python3
import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy

class SparseCategoricalMostConfident(SparseCategoricalAccuracy):
    """
    Calculates accuracy as the accuracy of the most confident
    predicition in the batch.

    Example:
        predicted               true
        ---                     ---
        [(0.1, 0.2, 0.3, 0.4)]  0
        [(0.5, 0.2, 0.3, 0.0)]  0

        result:
        1.00 because P_max = 0.5 in example 2 and
        argmax(y_pred_2) == 0 == y_true_2
    """

    def __init__(self, name='most_confident_acc', dtype=None):
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        index = tf.argmax(tf.math.reduce_max(y_pred, axis=-1))
        y_true = tf.gather(y_true, index)
        y_pred = tf.gather(y_pred, index)

        return super().update_state(y_true, y_pred, sample_weight)

class SparseCategoricalTopKConfident(SparseCategoricalAccuracy):
    """
    Calculates per batch accuracy like with
    SparseCategoricalMostConfident, but now consider the top k most
    confident predictions rather than top 1
    """

    def __init__(self, name='top_k_confident_acc', k=1, dtype=None):
        super().__init__(name, dtype)
        self.k = k

    def update_state(self, y_true, y_pred, sample_weight=None):

        probabilities = tf.math.reduce_max(y_pred, axis=-1)
        values, indices = tf.math.top_k(probabilities, k=self.k)

        y_true = tf.gather(y_true, indices)
        y_pred = tf.gather(y_pred, indices)

        return super().update_state(y_true, y_pred, sample_weight)

class CategoricalMostConfident(CategoricalAccuracy):
    """
    Calculates accuracy as the accuracy of the most confident
    predicition in the batch.

    Example:
        predicted               true
        ---                     ---
        [(0.1, 0.2, 0.3, 0.4)]  [0, 0, 1, 0]
        [(0.5, 0.2, 0.3, 0.0)]  [1, 0, 0, 0]

        result:
        1.00 because P_max = 0.5 in example 2 and
        argmax(y_pred_2) == argmax(y_true_2)
    """

    def __init__(self, name='most_confident_acc', dtype=None):
        super().__init__(name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        index = tf.argmax(tf.math.reduce_max(y_pred, axis=-1))
        y_true = tf.gather(y_true, index)
        y_pred = tf.gather(y_pred, index)

        return super().update_state(y_true, y_pred, sample_weight)

class CategoricalTopKConfident(CategoricalAccuracy):
    """
    Calculates per batch accuracy like with
    CategoricalMostConfident, but now consider the top k most
    confident predictions rather than top 1
    """

    def __init__(self, name='top_k_confident_acc', k=1, dtype=None):
        super().__init__(name, dtype)
        self.k = k

    def update_state(self, y_true, y_pred, sample_weight=None):

        probabilities = tf.math.reduce_max(y_pred, axis=-1)
        values, indices = tf.math.top_k(probabilities, k=self.k)

        y_true = tf.gather(y_true, indices)
        y_pred = tf.gather(y_pred, indices)

        return super().update_state(y_true, y_pred, sample_weight)

class SparseCategoricalAccuracyAboveConfidence(SparseCategoricalAccuracy):
    """
    Calculates accuracy as the accuracy of predicitions in the batch where
    max(y_pred_i) >= threshold

    Example:
        thresh = 0.4

        predicted               true
        ---                     ---
        [(0.1, 0.2, 0.3, 0.4)]  0
        [(0.5, 0.2, 0.3, 0.0)]  0

        result = 0.5:
            P_max_1 = 0.4 >= thresh and argmax(y_pred_1) != true
            P_max_2 = 0.5 >= thresh and argmax(y_pred_2) == true

    Example 2:
        thresh = 0.3

        predicted               true
        ---                     ---
        [(0.1, 0.2, 0.3, 0.4)]  0
        [(0.5, 0.2, 0.3, 0.0)]  0

        result = 1.0:
            P_max_1 = 0.4 < thresh and argmax(y_pred_1) != true
            P_max_2 = 0.5 >= thresh and argmax(y_pred_2) == true

            The incorrect y_pred_1 is ignored because max(y_pred_1) < thresh
    """

    def __init__(self, name='top_k_confident_acc', thresh=0.5, dtype=None):
        super().__init__(name, dtype)
        self.thresh = thresh

    def update_state(self, y_true, y_pred, sample_weight=None):

        probabilities = tf.math.reduce_max(y_pred, axis=-1)

        is_above_thresh = tf.math.greater_equal(
            probabilities,
            tf.constant(self.thresh),
        )

        indices = tf.where(is_above_thresh)
        y_true = tf.gather(y_true, indices)
        y_pred = tf.gather(y_pred, indices)

        return super().update_state(y_true, y_pred, sample_weight)
