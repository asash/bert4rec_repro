import unittest

class TestLightFMRecommender(unittest.TestCase):
    def test_lightfm_recommender(self):
        USER_ID = '120'
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.recommenders.lightfm import LightFMRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit

        lightfm_recommender = LightFMRecommender(30, 'bpr')
        recommender = FilterSeenRecommender(lightfm_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()
