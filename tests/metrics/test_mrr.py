
import unittest

class TestMRR(unittest.TestCase):
    def test_map(self):
        from aprec.evaluation.metrics.mrr import MRR
        from aprec.api.action import Action
        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [Action(user_id = 1, item_id = 4, timestamp=1),
                  Action(user_id = 1, item_id = 3, timestamp=2)]
        mrr = MRR()
        self.assertEqual(mrr(recommended, actual), 1/3)

if __name__ == "__main__":
    unittest.main()
