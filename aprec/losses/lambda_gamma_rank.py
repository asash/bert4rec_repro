import tensorflow as tf

from aprec.losses.loss import Loss

class LambdaGammaRankLoss(Loss):
    #num items and batch size need to be specified before usage.
    #some models can do it automatically, therefore they are set to None by default
    def __init__(self, num_items=None, batch_size=None, sigma=1.0, ndcg_at=50, dtype=tf.float32, lambda_normalization=True,
                 pred_truncate_at=None, bce_grad_weight=0.0, remove_batch_dim=False):
        super().__init__(num_items, batch_size)

        self.__name__ = 'lambdarank'
        self.sigma = sigma
        self.dtype = dtype
        self.bce_grad_weight = bce_grad_weight
        self.remove_batch_dim = remove_batch_dim
        self.params_truncate_at = pred_truncate_at

        self.params_ndcg_at = ndcg_at
        self.lambda_normalization = lambda_normalization
        self.setup()

    def set_num_items(self, num_items):
        super().set_num_items(num_items)
        self.setup()

    def set_batch_size(self, batch_size):
        super().set_batch_size(batch_size)
        self.setup()

    def get_pairwise_diffs_for_vector(self, x):
        a, b = tf.meshgrid(x[:self.ndcg_at], tf.transpose(x))
        result = tf.subtract(b, a)
        return result

    def get_pairwise_diff_batch(self, x):
        x_top_tile = tf.tile(tf.expand_dims(x[:, :self.ndcg_at], 1), [1, self.pred_truncate_at, 1])
        x_tile = tf.tile(tf.expand_dims(x, 2), [1, 1, self.ndcg_at])
        result = x_tile - x_top_tile
        return result

    def setup(self):
        if self.batch_size is None or self.num_items is None:
            return

        if self.params_truncate_at == None:
            self.pred_truncate_at = self.num_items
        else:
            self.pred_truncate_at = self.params_truncate_at

        self.ndcg_at = min(self.params_ndcg_at, self.num_items)
        self.dcg_position_discounts = 1. / tf.experimental.numpy.log2(
                        tf.cast(tf.range(self.pred_truncate_at) + 2, self.dtype))
        self.top_position_discounts = tf.reshape(self.dcg_position_discounts[:self.ndcg_at], (self.ndcg_at, 1))
        self.swap_importance = tf.abs(self.get_pairwise_diffs_for_vector(self.dcg_position_discounts))
        self.batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(self.batch_size), 1), [1, self.pred_truncate_at]),
                                        (self.pred_truncate_at * self.batch_size, 1))
        self.mask = tf.cast(tf.reshape(1 - tf.pad(tf.ones(self.ndcg_at),
                                                  [[0, self.pred_truncate_at - self.ndcg_at]]),
                                       (1, self.pred_truncate_at)), self.dtype)


    @tf.custom_gradient
    def __call__(self, y_true_raw, y_pred_raw):
        if self.remove_batch_dim:
            y_true = tf.reshape(y_true_raw, (self.batch_size, self.num_items))
            y_pred = tf.reshape(y_pred_raw, (self.batch_size, self.num_items))
        else:
            y_true = y_true_raw
            y_pred = y_pred_raw
        result = tf.reduce_mean(tf.abs(y_pred))

        def grad(dy):
            lambdarank_lambdas = self.get_lambdas(y_true, y_pred)
            bce_lambdas = self.get_bce_lambdas(y_true, y_pred)
            return 0 * dy, ((1 - self.bce_grad_weight) * lambdarank_lambdas + (bce_lambdas * self.bce_grad_weight)) * dy


        return result, grad

    def get_bce_lambdas(self, y_true, y_pred):
        with tf.GradientTape() as g:
            g.watch(y_pred)
            bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
            logits_loss_lambdas = g.gradient(bce_loss, y_pred) / self.num_items
        return  logits_loss_lambdas

    def bce_lambdas_len(self, y_true, y_pred):
        bce_lambdas = self.get_bce_lambdas(y_true, y_pred)
        norms = tf.norm(bce_lambdas , axis=1)
        return self.bce_grad_weight * tf.reduce_mean(norms)

    @tf.function
    def get_lambdas(self, y_true, y_pred):
        sorted_by_score = tf.nn.top_k(tf.cast(y_pred, self.dtype), self.pred_truncate_at)
        col_indices_reshaped = tf.reshape(sorted_by_score.indices, (self.pred_truncate_at * self.batch_size, 1))
        pred_ordered = sorted_by_score.values
        true_ordered = tf.gather(tf.cast(y_true, self.dtype), sorted_by_score.indices, batch_dims=1)
        inverse_idcg = self.get_inverse_idcg(true_ordered)
        true_gains = 2 ** true_ordered - 1
        true_gains_diff = self.get_pairwise_diff_batch(true_gains)
        S = tf.sign(true_gains_diff)
        delta_ndcg = true_gains_diff * self.swap_importance * inverse_idcg
        pairwise_diffs = self.get_pairwise_diff_batch(pred_ordered) * S

        #normalize dcg gaps - inspired by lightbm
        if self.lambda_normalization:
            best_score = pred_ordered[:, 0]
            worst_score = pred_ordered[:, -1]

            range_is_zero = tf.reshape(tf.cast(tf.math.equal(best_score, worst_score), self.dtype), (self.batch_size, 1, 1))
            norms = (1 - range_is_zero) * (tf.abs(pairwise_diffs) + 0.01) + (range_is_zero)
            delta_ndcg = tf.math.divide_no_nan(delta_ndcg, norms)

        sigmoid = -self.sigma / (1 + tf.exp(self.sigma * (pairwise_diffs)))
        lambda_matrix =  delta_ndcg * sigmoid

        #calculate sum of lambdas by rows. For top items - calculate as sum by columns.
        lambda_sum_raw = tf.reduce_sum(lambda_matrix, axis=2)
        top_lambda_sum = tf.pad(-tf.reduce_sum(lambda_matrix, axis=1), [[0, 0],
                                                                        [0, self.pred_truncate_at - self.ndcg_at]])
        lambda_sum_raw_top_masked = lambda_sum_raw * self.mask
        lambda_sum_result = lambda_sum_raw_top_masked + top_lambda_sum

        if self.lambda_normalization:
            #normalize results - inspired by lightbm
            all_lambdas_sum = tf.reshape(tf.reduce_sum(tf.abs(lambda_sum_result), axis=(1)), (self.batch_size, 1))
            norm_factor = tf.math.divide_no_nan(tf.experimental.numpy.log2(all_lambdas_sum + 1), all_lambdas_sum )

            lambda_sum = lambda_sum_result * norm_factor
        else:
            lambda_sum = lambda_sum_result

        indices = tf.concat([self.batch_indices, col_indices_reshaped], axis=1)
        result_lambdas = tf.scatter_nd(indices, tf.reshape(lambda_sum, [self.pred_truncate_at * self.batch_size]),
                                       tf.constant([self.batch_size, self.num_items]))
        return tf.cast(result_lambdas, tf.float32)

    def get_inverse_idcg(self, true_ordered):
        top_k_values = tf.nn.top_k(true_ordered, self.ndcg_at).values
        top_k_discounted = tf.linalg.matmul(top_k_values, self.top_position_discounts)
        return tf.reshape(tf.math.divide_no_nan(tf.cast(1.0, self.dtype), top_k_discounted), (self.batch_size, 1, 1))


class LambdarankLambdasSum(object):
    def __init__(self, lambdarank_loss):
        self.lambdarank_loss = lambdarank_loss
        self.name = "lambdarank_lambdas_len"

    def __call__(self, y_true, y_pred):
        lambdas = self.lambdarank_loss.get_lambdas(y_true, y_pred)
        return (1 - self.lambdarank_loss.bce_grad_weight) * tf.reduce_sum(tf.abs(lambdas))

class BCELambdasSum(object):
    def __init__(self, lambdarank_loss):
        self.lambdarank_loss = lambdarank_loss
        self.name = "bce_lambdas_len"

    def __call__(self, y_true, y_pred):
        lambdas = self.lambdarank_loss.get_bce_lambdas(y_true, y_pred)
        norms = tf.reduce_sum(lambdas, axis=1)
        return (self.lambdarank_loss.bce_grad_weight) * tf.reduce_sum(tf.abs(lambdas))

