from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
from aprec.recommenders.top_recommender import TopRecommender
from aprec.utils.generator_limit import generator_limit

import unittest
class TestTopRecommender(unittest.TestCase):
    def test_top_recommender(self):
        recommender = TopRecommender()
        catalog = get_movies_catalog()
        for action in generator_limit(get_movielens20m_actions(), 1000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.recommend(1, 5)
        self.assertEqual(recommendations, [('260', 10), ('589', 9), ('1', 8), ('356', 8), ('480', 8)])

    def test_recent_top(self):
        recommender = TopRecommender(recency=0.5)
        for action in generator_limit(get_movielens20m_actions(), 1000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.recommend(1, 5)
        self.assertEquals(recommendations,[('2959', 3), ('2762', 3), ('1196', 3), ('260', 3), ('587', 2)]
 )
        print(recommendations)


    def test_top_recommender_ranking_request(self):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        recommender = TopRecommender()
        ranking_request = ItemsRankingRequest(user_id='1', item_ids=['1196', '589'])
        recommender.add_test_items_ranking_request(ranking_request)
        actions = list(generator_limit(get_movielens20m_actions(), 1000))
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.get_item_rankings()
        self.assertEqual(recommendations, {'1': [('589', 9), ('1196', 8)]})

if __name__ == "__main__":
    unittest.main()
