

import unittest

class TestFilterSeenRecommender(unittest.TestCase):
    def test_constant_recommender(self):
        from aprec.recommenders.constant_recommender import ConstantRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.api.action import Action

        constant_recommender = ConstantRecommender(((1, 1),(2, 0.5), (3, 0.4)))
        recommender = FilterSeenRecommender(constant_recommender)
        recommender.add_action(Action(user_id=1, item_id=2, timestamp=1))
        self.assertEqual(recommender.recommend(1, 2), [(1, 1), (3, 0.4)])

    def test_filte_seen_sampled_rankings(self):
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.recommenders.top_recommender import TopRecommender
        from aprec.utils.generator_limit import generator_limit
        from aprec.api.items_ranking_request import ItemsRankingRequest
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

        recommender = FilterSeenRecommender(TopRecommender())
        ranking_request = ItemsRankingRequest(user_id='1', item_ids=['1196', '589'])
        recommender.add_test_items_ranking_request(ranking_request)
        for action in generator_limit(get_movielens20m_actions(), 1000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.get_item_rankings()
        self.assertEqual(recommendations,{'1': [('589', 9), ('1196', -float('inf'))]})

if __name__ == "__main__":
    unittest.main()