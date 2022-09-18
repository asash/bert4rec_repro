import os
import random

from tqdm import tqdm

from aprec.datasets.mts_kion import get_mts_kion_dataset, get_submission_user_ids, get_users, get_items
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.random_fraction_splitter import RandomFractionSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.kion_challenge_featurizer import KionChallengeFeaturizer
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.conditional_top_recommender import ConditionalTopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.hit import HIT
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.recommenders.deep_mf import DeepMFRecommender
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.transition_chain_recommender import TransitionsChainRecommender
from aprec.recommenders.lambdamart_ensemble_recommender import LambdaMARTEnsembleRecommender
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.featurizers.hashing_featurizer import HashingFeaturizer
from aprec.recommenders.svd import SvdRecommender
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = "mts_kion"
USERS = get_users
ITEMS = get_items

GENERATE_SUBMIT_THRESHOLD =  0.0

def generate_submit(recommender, recommender_name, evaluation_result, config):
    submit_user_ids = get_submission_user_ids()
    if evaluation_result["MAP@10"] <= config.GENERATE_SUBMIT_THRESHOLD:
        print("MAP@10 is less than threshold, not generating the submit")
        return

    print("generating submit...")
    with open(os.path.join(config.out_dir, recommender_name + "_submit_" + ".csv"), 'w') as out_file:
        out_file.write("user_id,item_id\n")
        for user_id in tqdm(submit_user_ids, ascii=True):
            recommendations = recommender.recommend(user_id, limit=10)
            content_ids = [recommendation[0] for recommendation in recommendations]
            line = user_id + ",\"["  +  ", ".join(content_ids) + "]\"\n"
            out_file.write(line)

CALLBACKS = (generate_submit, )

def deepmf(users_per_sample, items_per_sample, loss, truncation_level=None, bce_weight=0.0):
    if loss == 'lambdarank':
        loss = LambdaGammaRankLoss(items_per_sample, users_per_sample,
                                   ndcg_at=50, pred_truncate_at=truncation_level,
                                   bce_grad_weight=bce_weight, remove_batch_dim=True)
    return FilterSeenRecommender(DeepMFRecommender(users_per_sample, items_per_sample, loss, steps=1500))

USERS_FRACTIONS = [1.]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter=SequenceContinuation, 
                 target_builder=FullMatrixTargetsBuilder,
                optimizer=Adam(),
                training_time_limit=3600, metric=KerasNDCG(40), 
                max_epochs=10000,
                user_hasher=None
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
                                                          debug=False, 
                                                          users_featurizer=user_hasher
                                                          )


caser_default = dnn(Caser(requires_user_id=False, user_extra_features=True),
                                                                         loss=LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975),
                                                                         sequence_splitter=SequenceContinuation,
                                                                         user_hasher=HashingFeaturizer())

caser_random_fraction = dnn(Caser(requires_user_id=False, user_extra_features=True),
                                                                         loss=LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975),
                                                                         sequence_splitter=RandomFractionSplitter,
                                                                         user_hasher=HashingFeaturizer(), training_time_limit=4*3600)
sasrec = dnn(SASRec(max_history_len=50, 
                            dropout_rate=0.2,
                            num_heads=1,
                            num_blocks=2,
                            vanilla=True, 
                            embedding_size=50),
            BCELoss(),
            ShiftedSequenceSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=lambda: NegativePerPositiveTargetBuilder(50), 
            metric=BCELoss())

transitions_chain = TransitionsChainRecommender()

sasrec_biased = dnn(SASRec(max_history_len=50, 
                            dropout_rate=0.2,
                            num_heads=1,
                            num_blocks=2,
                            embedding_size=50),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            sequence_splitter = lambda: RecencySequenceSampling(max_pct=0.2, bias=0.8),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40))

 
recommenders_raw = {

     "Ensemble":  lambda: FilterSeenRecommender(LambdaMARTEnsembleRecommender(
                                                        candidates_selection_recommender=caser_random_fraction, 
                                                        other_recommenders = { 
                                                             "lightfm_bpr": LightFMRecommender(128, loss='bpr', num_threads=32), 
                                                             "lightfm_warp": LightFMRecommender(128, loss='warp', num_threads=32), 
                                                             "transitions_chain": transitions_chain,
                                                             "top_age":        ConditionalTopRecommender(conditional_field='age'), 
                                                             "top_sex":        ConditionalTopRecommender(conditional_field='sex'), 
                                                             "top_income":     ConditionalTopRecommender(conditional_field='income'), 
                                                             "top_kids":       ConditionalTopRecommender(conditional_field='kids_flg'), 
                                                             "svd":        SvdRecommender(128),
                                                             "top_full": TopRecommender(),
                                                             "top_recent_1pct": TopRecommender(0.01),
                                                             "top_recent_5pct": TopRecommender(0.05),
                                                             "top_recent_10pct": TopRecommender(0.1),
                                                             "top_recent_20pct": TopRecommender(0.2),
                                                             "top_recent_40pct": TopRecommender(0.4),
                                                             "top_recent_80pct": TopRecommender(0.8),
                                                             "sasrec_biased": sasrec_biased,
                                                             "sasrec":sasrec,
                                                             "caser_default": caser_default, 
                                                        }, 
                                                        featurizer=KionChallengeFeaturizer(),
                                                        n_ensemble_users=15000, 
                                                        lambda_l2=0.1,
                                                        recently_interacted_hours=7*24,
                                                        n_ensemble_val_users=2048,
                                                        ))
}


all_recommenders = list(recommenders_raw.keys())


RECOMMENDERS = {
        "top_recommender": lambda: TopRecommender(0.01),
    }
for model in all_recommenders:
    RECOMMENDERS[model] = recommenders_raw[model]

print(f"evaluating {len(RECOMMENDERS)} models")

N_VAL_USERS=1024
MAX_TEST_USERS=4096

METRICS = [MAP(10), NDCG(10), NDCG(2), NDCG(5), NDCG(20), NDCG(40), Precision(10), Recall(10), HIT(1), HIT(10), MRR()]


SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS, remove_single_action=False, recently_interacted_hours=7*24)
FILTER_COLD_START=False
