from .metric import Metric
from collections import Counter

class AveragePopularityRank(Metric):
    def __init__(self, k, actions):
        self.name = "apr@{}".format(k)
        self.k = k
        cnt = Counter()
        for action in actions:
            cnt[action.item_id] += 1

        self.pop_rank = {}
        rank = 0
        for item, cnt in cnt.most_common():
            rank += 1
            self.pop_rank[item] = rank

        
    def __call__(self, recommendations, actual_actions):
        cnt = 0
        s =  0
        for recommendation in recommendations[:self.k]:
            item_id = recommendation[0]
            if item_id in self.pop_rank:
                s += self.pop_rank[item_id]
                cnt += 1
        if cnt == 0:
            return 0
        return s/cnt
