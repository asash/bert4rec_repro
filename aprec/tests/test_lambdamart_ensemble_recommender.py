
import unittest

class TestLambdaMartEnsembleRecommender(unittest.TestCase):
    def test_lambdamart_ensemble_recommender(self):
        import json
        import os
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.recommenders.top_recommender import TopRecommender
        from aprec.recommenders.svd import SvdRecommender
        from aprec.recommenders.lambdamart_ensemble_recommender import LambdaMARTEnsembleRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        import tempfile
        from aprec.utils.generator_limit import generator_limit
        import pandas as pd

        tempdir = tempfile.mkdtemp("lambdamart_recommender_test")
        candidates_selection = FilterSeenRecommender(TopRecommender())
        other_recommenders = {
                                "svd_recommender": SvdRecommender(128)
                             }
        recommender = LambdaMARTEnsembleRecommender(
                            candidates_selection_recommender=candidates_selection, 
                            other_recommenders=other_recommenders,
                            n_ensemble_users=200, 
                            n_ensemble_val_users=20, 
                            log_dir=tempdir
        ) 
        
        USER_ID = '120'

        for action in generator_limit(get_movielens20m_actions(), 100000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        recs = recommender.recommend('121', 10)
        print(recs)
        print(json.dumps(recommender.get_metadata()))
        recommender.predictions_file.close()
        train_csv = pd.read_csv(os.path.join(tempdir, 'ensemble_train.csv.gz'), compression='gzip', delimiter=';')
        val_csv = pd.read_csv(os.path.join(tempdir, 'ensemble_train.csv.gz'), compression='gzip', delimiter=';')
        prediction_csv = pd.read_csv(os.path.join(tempdir, 'ensemble_prediction_features.csv.gz'), compression='gzip', delimiter=';')
        print(train_csv)
        print(val_csv)
        print(prediction_csv)

if __name__ == "__main__":
    unittest.main()



