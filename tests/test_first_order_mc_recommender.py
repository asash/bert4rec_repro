
import unittest

USER_ID = '120'

class TestFirstOrderMCRecommender(unittest.TestCase):
    def test_first_order_mc_recommender(self):
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.first_order_mc import FirstOrderMarkovChainRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        recommender = FilterSeenRecommender(FirstOrderMarkovChainRecommender())
        for action in generator_limit(get_movielens20m_actions(), 100000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

    def test_sampled_rankings(self):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.first_order_mc import FirstOrderMarkovChainRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit

        recommender = FilterSeenRecommender(FirstOrderMarkovChainRecommender())
        for action in generator_limit(get_movielens20m_actions(), 100000):
            recommender.add_action(action)
        ranking_request = ItemsRankingRequest('120', ['608', '294', '648'])
        recommender.add_test_items_ranking_request(ranking_request)
        recommender.rebuild_model()
        sampled_scores = recommender.get_item_rankings()
        self.assertEqual(len(sampled_scores), 1)
        predicted_scores = sampled_scores['120']
        unseen_item = '294'
        for item, score in predicted_scores:
            if item == unseen_item:
                self.assertEqual(score, 0)
            else:
                self.assertGreater(score, 0)



if __name__ == "__main__":
    unittest.main()
