import unittest

from aprec.losses.xendcg import XENDCGLoss


class TestXENDCGLoss(unittest.TestCase):
    def test_xendcg(self):
        import tensorflow as tf

        true = tf.constant([[0.0, 1.0, 0.0]])
        pred = tf.constant([[0.0, 0.5, 0]])
        xendcg = XENDCGLoss(true.shape[1], true.shape[0])
        result = xendcg(true, pred)
        self.assertAlmostEqual(float(result), 1.0122824907302856)

    @unittest.skip  # TODO - this test sometimes fails, as model training is stochastic.
    def test_model_xendcg(self):
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        X = tf.constant([[0.0, 0], [1, 0]])
        Y = tf.constant([[1.0, 0], [0, 1]])
        model = Sequential()
        model.add(Dense(2, activation="sigmoid"))
        model.add(Dense(2, activation="sigmoid"))
        model.add(Dense(2, activation="sigmoid"))
        model.add(Dense(2, activation="linear"))
        xendcg = XENDCGLoss(X.shape[1], X.shape[0])
        model.compile(optimizer="adam", loss=xendcg)

        model.fit(X, Y, epochs=2000, verbose=False)
        result = model.predict(X)
        tf.print(result)
        assert result[0, 0] > result[0, 1]
        assert result[1, 0] < result[1, 1]


if __name__ == "__main__":
    unittest.main()
