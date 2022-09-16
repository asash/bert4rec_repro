from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.rss_paper.common_benchmark_config import *

DATASET = "gowalla_warm5"
N_VAL_USERS=1024
MAX_TEST_USERS=86168
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=False, filter_recommenders=set(["bert4rec-1h", "bert4rec-16h"]))