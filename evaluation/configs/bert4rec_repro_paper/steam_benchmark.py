from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_repro_paper.common_benchmark_config import *

DATASET = "BERT4rec.steam"
N_VAL_USERS=2048
MAX_TEST_USERS=281428
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)