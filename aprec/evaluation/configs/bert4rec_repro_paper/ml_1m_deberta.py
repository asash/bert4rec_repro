from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.losses.mean_ypred_ploss import MeanPredLoss
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import (
    DNNSequentialRecommender,
)
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import (
    AddMaskHistoryVectorizer,
)
from aprec.recommenders.dnn_sequential_recommender.models.deberta4rec.deberta4rec import (
    Deberta4Rec,
)
from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import (
    ItemsMaskingTargetsBuilder,
)
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import (
    ItemsMasking,
)
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]


def top_recommender():
    return TopRecommender()


def svd_recommender(k):
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    return LightFMRecommender(k, loss, num_threads=32)


def deberta4rec(relative_position_encoding, sequence_len):
    model = Deberta4Rec(
        embedding_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        max_history_len=sequence_len,
    )
    recommender = DNNSequentialRecommender(
        model,
        train_epochs=10000,
        early_stop_epochs=200,
        batch_size=256,
        training_time_limit=3600000,
        loss=MeanPredLoss(),
        debug=False,
        sequence_splitter=lambda: ItemsMasking(masking_prob=0.2),
        targets_builder=lambda: ItemsMaskingTargetsBuilder(
            relative_positions_encoding=relative_position_encoding
        ),
        val_sequence_splitter=lambda: ItemsMasking(force_last=True),
        metric=MeanPredLoss(),
        pred_history_vectorizer=AddMaskHistoryVectorizer(),
    )
    return recommender


recommenders = {
    "deberta4rec_static-200": lambda: deberta4rec(False, 200),
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)


def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[
                recommender_name
            ] = lambda recommender_name=recommender_name: FilterSeenRecommender(
                recommenders[recommender_name]()
            )
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result


DATASET = "BERT4rec.ml-1m"
N_VAL_USERS = 2048
MAX_TEST_USERS = 6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)
