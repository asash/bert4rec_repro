
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def b4rvae_bert4rec(epochs=None):
    return B4rVaeBert4Rec(epochs=epochs)


recommenders = {
      "b4vae_bert4rec": b4rvae_bert4rec,
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