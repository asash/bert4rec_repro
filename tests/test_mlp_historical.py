from aprec.recommenders.mlp_historical import GreedyMLPHistorical
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
import tensorflow as tf
import unittest

USER_ID = '120'

class TestMLPRecommender(unittest.TestCase):
    def setUp(cls):
        tf.keras.backend.clear_session()

    def tearDown(cls):
        tf.keras.backend.clear_session()


    def test_mlp_recommender(self):
        mlp_recommender = GreedyMLPHistorical(train_epochs=10, n_val_users=10, batch_size=5)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()
