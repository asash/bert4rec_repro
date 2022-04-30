from aprec.losses.loss import Loss
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K


class BCELoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__name__ = "BCE"
        self.less_is_better = True
        self.eps = tf.constant(1e-16, 'float32')

    def __call__(self, y_true_raw, y_pred):
        y_true = tf.cast(y_true_raw, 'float32')
        is_target = tf.cast((y_true >= -self.eps), 'float32')
        trues = y_true*is_target
        pos = -trues*tf.math.log((tf.sigmoid(y_pred) + self.eps)) * is_target
        neg = -(1.0 - trues)*tf.math.log((1.0 - tf.sigmoid(y_pred)) + self.eps) * is_target
        num_targets = tf.reduce_sum(is_target)
        ce_sum = tf.reduce_sum(pos + neg)
        res_sum = tf.math.divide_no_nan(ce_sum, num_targets)
        return res_sum