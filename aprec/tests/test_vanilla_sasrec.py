import unittest

class TestVanillaSasrec(unittest.TestCase):
    def test_vanilla_sasrec(self):
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.losses.bce import BCELoss
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
        from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit

        USER_ID = '120'
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = SASRec(embedding_size=32, vanilla=True)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, 
                                               debug=True, sequence_splitter=ShiftedSequenceSplitter, 
                                               targets_builder=NegativePerPositiveTargetBuilder,
                                               metric=BCELoss()
                                               )
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

if __name__ == "__main__":
    unittest.main()