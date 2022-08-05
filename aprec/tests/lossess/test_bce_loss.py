import unittest

class TestBCELoss(unittest.TestCase):
    def test_bce_loss(self):
        from tensorflow.keras.losses import BinaryCrossentropy
        import tensorflow as tf
        from aprec.tests.lossess.bce_bad_sample import y_true as bad_y_true
        from aprec.tests.lossess.bce_bad_sample import y_pred as bad_y_pred
        from aprec.losses.bce import BCELoss
        loss = float(BCELoss()(bad_y_true, bad_y_pred))
        print(loss)

        y_true = tf.constant([-1., -1, -1, -1])
        y_pred = [-50.0, -50, -50, -50]
        loss = float(BCELoss()(y_true, y_pred))
        self.assertAlmostEqual(loss, 0.0)

        y_true = tf.constant([1, 0, 1, 0])
        y_pred = [0.1, 0.2, 0.3, 0.4]
        loss = float(BCELoss()(y_true, y_pred))
        keras_loss = float(BinaryCrossentropy(from_logits=True)(y_true, y_pred))
        self.assertAlmostEqual(loss, keras_loss, 5)

        y_true = tf.constant([1., 0, 1, 0])
        y_pred = [-50.0, -50, -50, -50]
        loss = float(BCELoss()(y_true, y_pred))
        keras_loss = float(BinaryCrossentropy(from_logits=True)(y_true, y_pred))
        self.assertAlmostEqual(loss, 18.420679092407227)


if __name__ == "__main__":
    unittest.main()

