import unittest


def get_actions():
    from aprec.utils.generator_limit import generator_limit
    from aprec.datasets.movielens20m import get_movielens20m_actions
    return [action for action in generator_limit(get_movielens20m_actions(), 100000)]

def get_recommender_and_add_actions():
        recommender = get_recommender()
        for action in get_actions():
            recommender.add_action(action)
        return recommender

def get_recommender():
    from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
    return  VanillaBERT4Rec(training_time_limit=5)

class TestVanillaBert4rec(unittest.TestCase):
    def test_vanilla_bert4rec(self):
        recommender = get_recommender_and_add_actions()
        recommender.rebuild_model()
        print(recommender.recommend('120', 10))
        recs = recommender.recommend('cold-start-user', 10)
        self.assertEqual(recs, [])

    def test_sampled_rankings(self):
        from aprec.api.items_ranking_request import ItemsRankingRequest
        recommender = get_recommender_and_add_actions()
        predict_items = ['260', '294', '296']
        ranking_request = ItemsRankingRequest('120', ['260', '294', '296'])
        recommender.add_test_items_ranking_request(ranking_request)
        recommender.rebuild_model()
        sampled_scores = recommender.get_item_rankings()
        self.assertEqual(len(sampled_scores), 1)
        predicted_scores = sampled_scores['120']
        unseen_item = '294'
        for item, score in predicted_scores:
            self.assertTrue(item in predict_items)
            if item == unseen_item:
                self.assertEqual(score, -float('inf'))
            else:
                self.assertGreater(score, -float('inf'))
                self.assertLess(score, float('inf'))




if __name__ == "__main__":
    unittest.main()
