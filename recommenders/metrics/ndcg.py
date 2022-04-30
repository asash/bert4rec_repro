import math

import tensorflow as tf

class KerasNDCG(object):
    def __init__(self, k):
        self.k = k
        discounts = []
        for i in range(1, k+1):
            discounts.append(1 / math.log2(i + 1))
        self.discounts = tf.cast(tf.constant(tf.expand_dims(discounts, 1)), 'float32')
        self.__name__ = f"ndcg_at_{k}"
        self.less_is_better = False

    def dcg(self, scores):
       gain = tf.pow(2.0, scores) - 1
       return gain @ self.discounts

    def __call__(self, y_true, y_pred):
        eps = 0.000001
        top_k = tf.nn.top_k(y_pred, self.k)
        gains = tf.gather(y_true, top_k.indices, batch_dims=1)
        dcg_val = self.dcg(tf.cast(gains, 'float32'))

        ideal_top_k = tf.nn.top_k(y_true, self.k)
        ideal_gains = tf.gather(y_true, ideal_top_k.indices, batch_dims=1)
        idcg_val = self.dcg(tf.cast(ideal_gains, 'float32'))
        return float(tf.reduce_mean(dcg_val / (idcg_val + eps)))

