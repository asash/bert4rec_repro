import unittest

class TestNDCG(unittest.TestCase):
    def test_ndcg(self):
        from aprec.evaluation.metrics.ndcg import NDCG
        from aprec.api.action import Action
        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [Action(user_id = 1, item_id = 4, timestamp=1), 
                  Action(user_id = 1, item_id = 3, timestamp=2)]
        ndcg = NDCG(3)
        self.assertEqual(ndcg(recommended, actual), 0.5)

if __name__ == "__main__":
    unittest.main()
