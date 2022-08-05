from aprec.recommenders.metrics.ndcg import KerasNDCG
import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import unittest

class TestKerasNDCG(unittest.TestCase):
    def setUp(cls):
        tf.keras.backend.clear_session()

    def tearDown(cls):
        tf.keras.backend.clear_session()


    def test_keras_ndcg(self):
        EPS=1e-5
        y_true = K.constant(np.array([[0, 1, 0], [1, 1, 0]]))
        y_pred = K.constant(np.array([[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]]))
        keras_ndcg = KerasNDCG(2)
        res = keras_ndcg(y_true, y_pred)
        assert abs(res - K.constant(0.815464854)) < EPS

if __name__ == "__main__":
    unittest.main()
