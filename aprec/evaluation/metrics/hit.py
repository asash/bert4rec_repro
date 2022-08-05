from .metric import Metric

class HIT(Metric):
    """
        Short-Term Prediction Success
        Equals 1 if recommender system was able to predict next item in sequence, 0 otherwise.  
    """
    def __init__(self, k):
        self.name = "HIT@{}".format(k)
        self.k = k
        
    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0):
            return 0
        action_to_check = actual_actions[0] 
        for action in actual_actions[1:]:
            if action.timestamp < action_to_check.timestamp:
                action_to_check = action
        recommended = set([recommendation[0] for recommendation in recommendations[:self.k]])
        return 1 if action_to_check.item_id in recommended else 0
