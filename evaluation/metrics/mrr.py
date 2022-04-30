import math
from .metric import Metric

class MRR(Metric):
    def __init__(self):
        self.name = "MRR"

    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0):
            return 0
        actual_set = set([action.item_id for action in actual_actions])
        for i in range(len(recommendations)):
            if recommendations[i][0] in actual_set:
                return 1/(i + 1)
        return 0
