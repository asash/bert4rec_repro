import math
from .metric import Metric

class MAP(Metric):
    def __init__(self, k):
        self.name = f"MAP@{k}"
        self.k = k

    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0 or len(actual_actions) == 0):
            return 0
        actual_set = set([action.item_id for action in actual_actions])
        correct_predictions = 0
        running_sum = 0
        for i in range(len(recommendations[:self.k])):
            pos = i + 1
            predicted = recommendations[i][0]
            if predicted in actual_set:
                correct_predictions += 1
                running_sum += correct_predictions/pos
                pass
        return running_sum / len(actual_actions)


