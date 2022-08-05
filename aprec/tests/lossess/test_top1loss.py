import unittest

def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

def naive_top1_impl(y_true, y_pred, softmax_weighted=False):
    import math
    n_pairs = 0
    loss = 0.0
    exp_sum = 0.0
    for i in range(len(y_true)):
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
                diff = negative - positive
                sigm = sigmoid(diff)
                weight = 1
                if softmax_weighted:
                    weight = math.exp(y_pred[j]) / exp_sum
                term = (sigm + sigmoid(negative ** 2))*weight
                loss += term 
    return loss/n_pairs


class TestTOP1Loss(unittest.TestCase):
        def compare_with_naive(self, a, b, ordered=False, weighted=False):
            import tensorflow as tf
            from aprec.losses.top1 import TOP1Loss


            if not ordered:
                top1_loss = TOP1Loss(softmax_weighted=weighted)
            else:
                top1_loss = TOP1Loss(pred_truncate=len(a), softmax_weighted=weighted)
            naive_loss_val = naive_top1_impl(a, b, softmax_weighted=weighted)
            computed_loss_val = float(top1_loss(tf.constant([a]), tf.constant([b])))
            self.assertAlmostEquals(computed_loss_val, naive_loss_val, places=4)
            
        def test_compare_with_naive(self):
                import random
                self.compare_with_naive([0.0, 1.], [0.2, 0.1])
                random.seed(1)
                for i in range(100):
                    ordered = bool(random.randint(0, 1))
                    weighted = bool(random.randint(0, 1))
                    sample_len = random.randint(2, 500)
                    y_pred = []
                    y_true = [0] * sample_len
                    y_true[random.randint(0, sample_len - 1)] = 1.0
                    for j in range(sample_len):
                        y_pred.append(random.random())
                    self.compare_with_naive(y_true, y_pred, ordered, weighted)
                        

        def test_top1_loss(self):
            from aprec.losses.top1 import TOP1Loss
            import tensorflow.keras.backend as K

            top1_loss = TOP1Loss() 
            val = top1_loss(K.constant([[0, 0, 0, 1.0],
                                 [0, 0, 1., 0]]),
                     K.constant([[0.1, 0.3, 1, 0], [0, 0, 1, 1.0]]))
            self.assertAlmostEqual(float(val),1.059244155883789, places=4)

        def test_top1_truncate(self):
            from aprec.losses.top1 import TOP1Loss
            import tensorflow as tf 

            top1_loss = TOP1Loss(pred_truncate=1) 
            val = float(top1_loss(tf.constant([[0, 0, 0, 1]]), tf.constant([[0.1, 0.3, 0, 0]])))
            self.assertAlmostEqual(val, 1.0969274044036865)
 


if __name__ == "__main__":
   unittest.main()
