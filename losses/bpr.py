from math import perm
import tensorflow as tf

from aprec.losses.loss_utils import get_pairwise_diff_batch, get_truncated, masked_softmax
from aprec.losses.loss import Loss

#BPR Loss as  described in orignial paper. 
#https://arxiv.org/abs/1205.2618
#This loss doesn't include regularization term as in tensorflow it should be done on the model side (e.g. include l2 regularization in embeddings)
#Setting softmax_weighted into True will turn this loss into BPR-max loss, as described in the GRU4Rec+ paper
##https://dl.acm.org/doi/abs/10.1145/3269206.3271761
class BPRLoss(Loss):
    def __init__(self, num_items=None, batch_size=None, max_positives=10, pred_truncate=None, softmax_weighted=False):
        super().__init__(num_items, batch_size)
        self.max_positives = max_positives
        self.softmax_weighted=softmax_weighted
        self.pred_truncate = pred_truncate

    def __call__(self, y_true, y_pred):
        top_true = tf.math.top_k(y_true, self.max_positives)
        pred_ordered_by_true = tf.gather(y_pred, top_true.indices, batch_dims=1)

        pred, true_ordered_by_pred = get_truncated(y_true, y_pred, self.pred_truncate) 
        pred_size = tf.shape(pred)[-1]

        mask = tf.cast((get_pairwise_diff_batch(top_true.values, true_ordered_by_pred, self.max_positives, pred_size) > 0), tf.float32)
        values = get_pairwise_diff_batch(pred_ordered_by_true, pred, self.max_positives, pred_size)
        sigmoid =  -tf.math.log_sigmoid(values) * mask
        if self.softmax_weighted:
            pred_tile = tf.tile(tf.expand_dims(pred, 1), [1, self.max_positives, 1])
            mask_transposed = tf.transpose(mask, perm=[0, 2, 1])
            pred_softmax = tf.transpose(masked_softmax(pred_tile, mask_transposed), perm=[0, 2, 1])
            sigmoid *= pred_softmax
        result = tf.reduce_sum(sigmoid, axis=[1, 2]) / tf.reduce_sum(mask, axis=[1, 2])
        return tf.reduce_mean(result)

