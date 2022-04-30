import unittest

class TestLambdaranTime(unittest.TestCase):
   def test_get_lambdas(self):
       import random
       import numpy as np
       from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
       import tensorflow as tf
       from tqdm import tqdm


       random.seed(31337)
       np.random.seed(31337)
       batch_size = 128
       dataset_size = 128 * 1024
       positives_per_sample = 100
       n_items = 50000
       pred_truncate_at = 500

       y_true = np.zeros((batch_size, n_items))
       for sample_num in range(batch_size):
            positives = np.random.choice((range(n_items)), positives_per_sample, replace=False)
            for positive in positives:
                y_true[sample_num][positive] = random.random()
       y_true = tf.constant(y_true)

       loss = LambdaGammaRankLoss(n_items, batch_size, 1, ndcg_at=40, dtype=tf.float32,
                                  pred_truncate_at=pred_truncate_at, bce_grad_weight=0.1)
       for i in tqdm(range(dataset_size // batch_size)):
           y_pred =  tf.random.uniform((batch_size, n_items))
           #tf.keras.losses.binary_crossentropy(y_true, y_pred)
           loss.get_lambdas(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()
