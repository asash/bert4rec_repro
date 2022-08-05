import unittest

def sigmoid(x):
    import math 
    return 1 / (1 + math.exp(-x))

def naive_bpr_impl(y_true, y_pred, softmax_weighted=False):
    import math
    n_pairs = 0
    loss = 0.0
    for i in range(len(y_true)):
        exp_sum = 0.0
        if y_true[i] < 1e-5:
            continue
        for j in range(len(y_true)):
            if y_true[i] > 0.5 and y_true[j] < 0.5:
                exp_sum += math.exp(y_pred[j])
        for j in range(len(y_true)):
            if y_true[i] > 0.5 and y_true[j] < 0.5:
                n_pairs += 1
                positive = y_pred[i]
                negative = y_pred[j]
                diff = positive - negative
                sigm = sigmoid(diff)
                weight = 1
                if softmax_weighted:
                    weight = math.exp(y_pred[j]) / exp_sum
                loss -= math.log(sigm) * weight
    return loss/n_pairs


    

class TestBPRLoss(unittest.TestCase):
        def compare_with_naive(self, a, b, ordered=False, weighted=False):
            from aprec.losses.bpr import BPRLoss
            import tensorflow as tf

            if not ordered:
                bpr_loss = BPRLoss(max_positives=len(a), num_items=len(a), batch_size=1, softmax_weighted=weighted)
            else:
                bpr_loss = BPRLoss(max_positives=len(a), pred_truncate=len(a), num_items=len(a), batch_size=1, softmax_weighted=weighted)
            naive_bpr_los_val = naive_bpr_impl(a, b, softmax_weighted=weighted)
            computed_loss_val = float(bpr_loss(tf.constant([a]), tf.constant([b])))
            self.assertAlmostEquals(computed_loss_val, naive_bpr_los_val, places=4)
            
        def test_compare_with_naive(self):
                import random
                import tensorflow.keras.backend as K


                self.compare_with_naive([0.0, 1.], [0.1, 0])
                random.seed(6)
                for i in range(100):
                    ordered = bool(random.randint(0, 1))
                    weighted = bool(random.randint(0, 1))
                    sample_len = random.randint(10, 100)
                    y_true = []
                    y_pred = []
                    for j in range(sample_len):
                        y_true.append(random.randint(0, 1) * 1.0)
                        y_pred.append(random.random())
                    self.compare_with_naive(y_true, y_pred, ordered, weighted)


        def test_bpr_loss(self):
            import tensorflow.keras.backend as K 
            from aprec.losses.bpr import BPRLoss
            bpr_loss = BPRLoss(max_positives=3, batch_size=2, num_items=4)

            val = bpr_loss(K.constant([[0, 0, 1, 1],
                                 [0, 0, 1, 1]]),
                     K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1]]))
            self.assertAlmostEqual(float(val), 0.4495173692703247, places=4)
            poor_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[1, 0.5, 0, 0]]))
            avg_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0.1, 0.3, 1, 0]]))
            good_pred_loss = bpr_loss(K.constant([[0, 0, 1, 1]]), K.constant([[0, 0, 1, 1]]))
            self.assertGreater (poor_pred_loss, avg_pred_loss)
            self.assertLess (good_pred_loss, avg_pred_loss)

        def test_bpr_loss_with_softmax(self):
            import tensorflow.keras.backend as K 
            from aprec.losses.bpr import BPRLoss
            bpr_loss = BPRLoss(max_positives=3, batch_size=2, num_items=4, softmax_weighted=True)
            val = bpr_loss(K.constant([[0, 0, 1, 1],
                                 [0, 0, 1, 1]]),
                     K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1]]))
            self.assertAlmostEqual(float(val), 0.2258300483226776, places=4)

        def test_bpr_truncate(self):
            from aprec.losses.bpr import BPRLoss
            import tensorflow as tf 

            bpr_loss = BPRLoss(max_positives=3, pred_truncate=1, num_items=4, batch_size=1)
            val = float(bpr_loss(tf.constant([[0, 0, 0, 1]]), tf.constant([[0.1, 0.3, 0, 0]])))
            self.assertAlmostEqual(val, 0.8543552444685271)
 


if __name__ == "__main__":
   unittest.main()
