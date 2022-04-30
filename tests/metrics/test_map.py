
import unittest

class TestMAP(unittest.TestCase):
    def test_map(self):
        from aprec.evaluation.metrics.map import MAP
        from aprec.api.action import Action
        recommended = [(6, 0.9), (3, 0.85), (5, 0.71), (0, 0.63), (4, 0.47), (2, 0.36), (1, 0.24), (7, 0.16)]
        actual = [Action(user_id = 1, item_id = 6, timestamp=1),
                  Action(user_id = 1, item_id = 5, timestamp=2),
                  Action(user_id = 1, item_id = 0, timestamp=3),
                  Action(user_id = 1, item_id = 2, timestamp=4),
                  ]
        map = MAP(8)
        self.assertEqual(map(recommended, actual), 0.7708333333333333)

if __name__ == "__main__":
    unittest.main()
