
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_repro_paper.common_benchmark_config import *

DATASET = "ml-20m"
N_VAL_USERS=1024
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)