import unittest

class TestMixer(unittest.TestCase):
    def test_mixer_recommender(self):
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.losses.mean_ypred_ploss import MeanPredLoss
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.models.mixer import MixerModel, RecsysMixer
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder

        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.tests.test_dnn_sequential import USER_ID
        from aprec.utils.generator_limit import generator_limit


        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = RecsysMixer(embedding_size=32)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, 
                                               loss = MeanPredLoss(),
                                               debug=True, sequence_splitter=lambda: ItemsMasking(recency_importance=exponential_importance(0.8)), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=True),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
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