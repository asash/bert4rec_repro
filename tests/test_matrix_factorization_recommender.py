from aprec.losses import top1
from aprec.recommenders.matrix_factorization import MatrixFactorizationRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
import tensorflow as tf
import unittest

USER_ID = '120'

class TestMatrixFactorizationRecommender(unittest.TestCase):
    def setUp(cls):
        tf.keras.backend.clear_session()

    def tearDown(cls):
        tf.keras.backend.clear_session()



    def test_matrix_factorization_recommender_recommender(self):
        losses = ['bce', 'bpr', 'lambdarank', 'xendcg', 'climf', 'top1']
        for loss in losses: 
            print(f"testing matrix factorization model with {loss} loss")
            matrix_factorization_recommender = MatrixFactorizationRecommender(32, 5, loss, batch_size=10)
            recommender = FilterSeenRecommender(matrix_factorization_recommender)
            for action in generator_limit(get_movielens20m_actions(), 10000):
                recommender.add_action(action)
            recommender.rebuild_model()
            recs = recommender.recommend(USER_ID, 10)
            print(recs)

    def test_recommend_batch(self):
        matrix_factorization_recommender = MatrixFactorizationRecommender(32, 5, 'bce', batch_size=10)
        recommender = FilterSeenRecommender(matrix_factorization_recommender)
        user_ids = set()
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
            user_ids.add(action.user_id)
        recommender.rebuild_model()
        requests = [(user_id, None) for user_id in ['142', '111', '57', '37', '136', '88']]
        batch_recommendations = recommender.recommend_batch(requests, 10)
        print(batch_recommendations)


if __name__ == "__main__":
    unittest.main()
