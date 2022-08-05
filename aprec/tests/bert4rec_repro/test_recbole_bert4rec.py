import unittest



class TestRecboleBert4rec(unittest.TestCase):
    def test_sampled_rankings(self):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.recommenders.bert4recrepro.recbole_bert4rec import RecboleBERT4RecRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit

        recommender = FilterSeenRecommender(RecboleBERT4RecRecommender(epochs=5, max_sequence_len=10))
        for action in generator_limit(get_movielens20m_actions(), 10000):
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

    def test_recbole_bert4rec(self):
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.bert4recrepro.recbole_bert4rec import RecboleBERT4RecRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit

        USER_ID = '120'
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        recommender = RecboleBERT4RecRecommender(epochs=5, max_sequence_len=10)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

    @unittest.skip
    def test_default_recbole(self):
        from recbole.quick_start import run_recbole
        parameter_dict = {
            'load_col': {'inter':  ['user_id', 'item_id', 'rating', 'timestamp']},
            'neg_sampling': None,
        }
        run_recbole(model='BERT4Rec', dataset='ml-1m', config_dict=parameter_dict)



if __name__ == "__main__":
    unittest.main()