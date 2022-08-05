import unittest

class TestB4rVaeBert4rec(unittest.TestCase):
    def test_b4rvae(self):
        from aprec.datasets.dataset_utils import filter_cold_users
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit

        USER_ID = '120'
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        recommender = B4rVaeBert4Rec(epochs=5)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in filter_cold_users(generator_limit(get_movielens20m_actions(), 10000), 5):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

    def test_sampled_rankings(self):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        from aprec.datasets.dataset_utils import filter_cold_users
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit


        recommender = FilterSeenRecommender(B4rVaeBert4Rec(epochs=5))
        for action in filter_cold_users(generator_limit(get_movielens20m_actions(), 100000), 5):
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
                self.assertEqual(score, float('-inf'))
            else:
                self.assertGreater(score, float('-inf'))



if __name__ == "__main__":
    unittest.main()