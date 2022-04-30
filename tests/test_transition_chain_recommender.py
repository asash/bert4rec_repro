import unittest

class TestTransitionsChainRecommender(unittest.TestCase):
    def test_transitions_chain_recommender(self):
        from typing import List
        from aprec.api.action import Action
        from aprec.recommenders.transition_chain_recommender import TransitionsChainRecommender
        recommender = TransitionsChainRecommender()
        actions: List[Action] = [
            Action(user_id=0, item_id=0, timestamp=0, data={'utrip_id': 100}),
            Action(user_id=0, item_id=1, timestamp=10, data={'utrip_id': 100}),
            Action(user_id=0, item_id=2, timestamp=20, data={'utrip_id': 100}),

            Action(user_id=2, item_id=3, timestamp=0, data={'utrip_id': 200}),
            Action(user_id=2, item_id=1, timestamp=10, data={'utrip_id': 200}),
            Action(user_id=2, item_id=2, timestamp=20, data={'utrip_id': 200}),

            Action(user_id=3, item_id=1, timestamp=10, data={'utrip_id': 2000}),
            Action(user_id=3, item_id=3, timestamp=20, data={'utrip_id': 2000}),

            Action(user_id=4, item_id=1, timestamp=0, data={'utrip_id': 300}),
        ]
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.recommend(4, 2)
        self.assertEqual(recommendations, [(2, 2), (3, 1)])

if __name__ == "__main__":
    unittest.main()
