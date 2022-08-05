import tensorflow as tf
import tensorflow.keras.backend as K

class KerasSuccess(object):
   def __init__(self, k):
        self.k = k
        self.__name__ = f"Success_at_{k}"
        self.less_is_better = False

   def __call__(self, y_true, y_pred):
        top_k = tf.nn.top_k(y_pred, self.k)
        gains = tf.gather(y_true, top_k.indices, batch_dims=1)
        user_success = K.sum(gains, axis=-1)
        return K.mean(user_success)

