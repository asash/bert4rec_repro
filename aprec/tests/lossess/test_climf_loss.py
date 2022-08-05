import unittest



class TestCLIMFLoss(unittest.TestCase):
    def test_climf_loss(self):
        from aprec.losses.climf import CLIMFLoss
        import tensorflow.keras.backend as K
        climf_loss = CLIMFLoss(4, 2, 3)
        val = climf_loss(K.constant([[0, 0, 1, 1],
                                   [0, 0, 1, 1]]),
                       K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1]]))
        self.assertAlmostEqual(float(val), 7.418338775634766, places=4)
        climf_loss = CLIMFLoss(4, 1, 3)
        poor_pred_loss = climf_loss(K.constant([[0, 0, 1, 1]]), K.constant([[1, 0.5, 0, 0]]))
        avg_pred_loss = climf_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0.1, 0.3, 1, 0]]))
        good_pred_loss = climf_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0, 0, 1, 1]]))
        assert (poor_pred_loss > avg_pred_loss)
        assert (good_pred_loss < avg_pred_loss)

if __name__ == "__main__":
    unittest.main()
