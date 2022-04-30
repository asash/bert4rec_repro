from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.matrix_factorization import MatrixFactorizationRecommender
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.sampled_proxy_metric import SampledProxy
from aprec.evaluation.split_actions import LeaveOneOut
import numpy as np


DATASET = "BERT4rec.ml-1m"

USERS_FRACTIONS = [1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def top_recommender_nofilter():
    return TopRecommender()

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def mf_recommender(embedding_size, num_epochs, loss, batch_size, regularization, learning_rate):
    return FilterSeenRecommender(MatrixFactorizationRecommender(embedding_size,
                     num_epochs, loss, batch_size, regularization, learning_rate))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

RECOMMENDERS = {
    "top_recommender": top_recommender,
    #"top_recommender_nofilter": top_recommender_nofilter,
    #"lightfm_recommender_30_bpr": lambda: lightfm_recommender(30, 'bpr'),
}

for i in range(0):
    all_losses = ['binary_crossentropy', 'xendcg', 'lambdarank', 'bpr', 'climf', 'mse']
    loss = all_losses[i % len(all_losses)]
    regularization = float(np.random.choice([0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 4]))
    embedding_size = int(np.random.choice([16, 32, 64, 128, 256, 512, 1024]))
    num_epochs = int(np.random.choice([16, 32, 64, 128, 256, 512, 1024]))
    batch_size = int(np.random.choice([64, 128, 256, 512]))
    learning_rate = float(np.random.choice([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,]))
    recommender = lambda embedding_size=embedding_size, num_epochs=num_epochs, loss=loss, \
                         batch_size=batch_size, learning_rate=learning_rate, regularization=regularization: \
        mf_recommender(embedding_size, num_epochs, loss, batch_size, regularization, learning_rate)
    name = f"mf_epochs:{num_epochs}_emb:{embedding_size}_reg:{regularization}_loss:{loss}_bs:{batch_size}_lr:{learning_rate}"
    RECOMMENDERS[name] = recommender


TEST_FRACTION = 0.25
MAX_TEST_USERS=6040

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR()]

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

