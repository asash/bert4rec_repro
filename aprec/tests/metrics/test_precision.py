import unittest

class TestPrecision(unittest.TestCase):
    def test_precsion(self):
        from aprec.evaluation.metrics.precision import Precision
        from aprec.api.action import Action

        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [Action(user_id = 1, item_id = 1, timestamp=1), 
                  Action(user_id = 1, item_id = 3, timestamp=2)]
        precision_1 = Precision(1)
        precision_2 = Precision(2)
        precision_3 = Precision(3)
        self.assertEqual(precision_1(recommended, actual), 1)
        self.assertEqual(precision_2(recommended, actual), 0.5)
        self.assertEqual(precision_3(recommended, actual), 2/3)

if __name__ == "__main__":
    unittest.main()
