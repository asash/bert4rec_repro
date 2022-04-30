from .metric import Metric

class Recall(Metric):
    def __init__(self, k):
        self.name = "recall@{}".format(k)
        self.k = k
        
    def __call__(self, recommendations, actual_actions):
        if len(recommendations) == 0:
            return 0
        actual_set = set([action.item_id for action in actual_actions])
        recommended = set([recommendation[0] for recommendation in recommendations[:self.k]])
        cool = recommended.intersection(actual_set)
        return len(cool) / len(actual_set)
