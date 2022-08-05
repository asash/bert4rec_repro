import math
from .metric import Metric

class NDCG(Metric):
    def __init__(self, k):
        self.name = "ndcg@{}".format(k)
        self.k = k
        
    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0):
            return 0
        actual_set = set([action.item_id for action in actual_actions])
        recommended = [recommendation[0] for recommendation in recommendations[:self.k]]
        cool = set(recommended).intersection(actual_set)
        if len(cool) == 0:
            return 0
        ideal_rec = sorted(recommended, key = lambda x: not(x in actual_set))
        return NDCG.dcg(recommended, actual_set)/NDCG.dcg(ideal_rec, actual_set)
         

    @staticmethod
    def dcg(id_list, relevant_id_set):
        result = 0.0
        for idx in range(len(id_list)):
            i = idx + 1
            if (id_list[idx]) in relevant_id_set:
                result += 1 / math.log2(i+1)
        return result




