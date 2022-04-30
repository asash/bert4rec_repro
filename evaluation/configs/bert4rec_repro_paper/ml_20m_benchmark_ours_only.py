
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.losses.mean_ypred_ploss import MeanPredLoss
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def our_bert4rec(relative_position_encoding=False, sequence_len=50, rss = lambda n, k: 1, layers=2, arch=BERT4Rec, masking_prob=0.2, max_predictions_per_seq=20):
        model = arch( max_history_len=sequence_len)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=64,
                                               training_time_limit=3600000, 
                                               loss = MeanPredLoss(),
                                               debug=True, sequence_splitter=lambda: ItemsMasking(masking_prob=masking_prob, max_predictions_per_seq=max_predictions_per_seq, recency_importance=rss), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=relative_position_encoding),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               )
        return recommender

recommenders = {
     "our_bert4rec":  our_bert4rec
}

TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)
METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

def get_recommenders(filter_seen: bool, filter_recommenders = set()):
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
                continue
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result



DATASET = "ml-20m"
N_VAL_USERS=1024
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)