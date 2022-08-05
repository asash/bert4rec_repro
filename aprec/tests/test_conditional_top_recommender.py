import unittest

class TestConditionalTopRecommender(unittest.TestCase):
    def test_conditional_top_recommender(self):
        from typing import List
        from aprec.api.action import Action
        from aprec.recommenders.conditional_top_recommender import ConditionalTopRecommender

        recommender = ConditionalTopRecommender(conditional_field='country_id')
        actions: List[Action] = [
            Action(user_id=0, item_id=0, timestamp=0, data={'country_id': 100}),
            Action(user_id=0, item_id=0, timestamp=10, data={'country_id': 100}),
            Action(user_id=0, item_id=1, timestamp=20, data={'country_id': 100}),
        ]
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.recommend(0, 1)
        self.assertEqual(recommendations, [(0, 2)])
    
if __name__ == "__main__":
    unittest.main()
