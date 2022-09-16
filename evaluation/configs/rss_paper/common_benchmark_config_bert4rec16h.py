from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
from aprec.evaluation.samplers.random_sampler import RandomTargetItemSampler
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from tensorflow.keras.optimizers import Adam

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def top_recommender():
    return TopRecommender()


def svd_recommender(k):
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    return LightFMRecommender(k, loss, num_threads=32)


def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter=SequenceContinuation, 
                 target_builder=FullMatrixTargetsBuilder,
                optimizer=Adam(),
                training_time_limit=3600, metric=KerasNDCG(40), 
                max_epochs=10000
                ):
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          debug=False
                                                          )

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

HISTORY_LEN=50

recommenders = {
    "bert4rec-16h": lambda: vanilla_bert4rec(3600 * 16) 
}

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
