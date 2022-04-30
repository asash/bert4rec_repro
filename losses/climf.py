#CLIMF Loss Implementation
#See paper:
#https://dl.acm.org/doi/10.1145/2365952.2365981

import tensorflow as tf

from aprec.losses.loss_utils import my_map
from aprec.losses.loss import Loss


class CLIMFLoss(Loss):
    def __init__(self, num_items=None, batch_size=None, max_positives=10):
        super().__init__(num_items, batch_size)
        self.max_positives = max_positives


    def get_pairwise_diffs_matrix(self, x, y):
        a, b = tf.meshgrid(tf.transpose(y), x)
        return tf.subtract(b, a)

    def get_pairwise_diffs_matrices(self, a, b):
        result = my_map(self.get_pairwise_diffs_matrix, (a, b))
        return result

    #equation (9) from the paper
    def __call__(self, y_true, y_pred):
        EPS=1e-6
        top_true = tf.math.top_k(y_true, self.max_positives)
        true_values = top_true.values
        pred_ordered = tf.gather(y_pred, top_true.indices, batch_dims=1)
        values = self.get_pairwise_diffs_matrices(pred_ordered, y_pred)
        values_sigmoid = tf.math.sigmoid(values)
        tiled_values = tf.tile(true_values, [1, y_pred.shape[-1]])
        mask = tf.reshape(tiled_values, (self.batch_size, self.num_items, true_values.shape[1]))
        mask = tf.transpose(mask, perm=[0, 2, 1])
        second_climf_term = tf.math.reduce_sum(tf.math.log(1 - mask*values_sigmoid + EPS), axis=1)
        first_climf_term = tf.math.log_sigmoid(y_pred)
        result = -tf.reduce_sum(y_true*(second_climf_term + first_climf_term))
        return result