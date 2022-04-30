import unittest

class TestPairwiseCosSim(unittest.TestCase):
    def test_pairwise_cos_sim(self):
        from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
        from aprec.api.action import Action
        actions = [Action(user_id=1, item_id=1, timestamp=1),
                   Action(user_id=1, item_id=3, timestamp=2),

                   Action(user_id=2, item_id=1, timestamp=2),
                   Action(user_id=2, item_id=2, timestamp=2),
                   Action(user_id=2, item_id=3, timestamp=2)]

        pairwise_cos_sim = PairwiseCosSim(actions, 10)

        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [Action(user_id = 1, item_id = 1, timestamp=1),
                  Action(user_id = 1, item_id = 2, timestamp=2)]
        self.assertEqual(pairwise_cos_sim(recommended, actual), 2/3)

if __name__ == "__main__":
    unittest.main()
