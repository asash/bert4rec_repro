import random
from collections import defaultdict, Counter

import numpy as np

from .metric import Metric

#this proxy is used to match BERT4rec evaluation strategy,
# in order to be able to compare our models evaluation to what they report in the paper
# In their code they randomly sample 100 items out of full items list, add relevant items and then calculate metrics.
# for the items outside of returned recommendations,
# we assume, that score is equal to min_score of recommended items - random value


class SampledProxy(Metric):
    def __init__(self, item_ids, probs, n_negatives, metric):
        self.item_ids = list(item_ids)
        self.n_negatives = n_negatives
        self.metric = metric
        self.name = f"{metric.name}_sampled@{self.n_negatives}"
        self.probs = probs

    def __call__(self, recommendations, actual_actions):
        rec_dict = {}
        min_score = float('inf')
        for item, score in recommendations:
            rec_dict[item] = score
            min_score = min(min_score, score)

        recs = []
        recommended = set()
        for action in actual_actions:
            recs.append((action.item_id, self.get_item_score(action.item_id, min_score, rec_dict)))
            recommended.add(action.item_id)

        target_size = len(actual_actions) + self.n_negatives
        while(len(recommended) < target_size):
            item_ids = np.random.choice(self.item_ids,  target_size - len(recommended), p=self.probs, replace=False)
            for item_id in item_ids:
                if item_id not in recommended:
                    recs.append((item_id, self.get_item_score(item_id, min_score, rec_dict)))
                    recommended.add(item_id)
        recs.sort(key=lambda x: -x[1])
        return self.metric(recs, actual_actions)

    @staticmethod
    def all_item_ids_probs(actions):
        counter = Counter()
        cnt = 0
        for action in actions:
            counter[action.item_id] += 1
            cnt += 1

        items, probs = [], []
        for item, item_cnt in counter.most_common():
            items.append(item)
            probs.append(item_cnt / cnt)
        return items, probs



    @staticmethod
    def get_item_score(item_id, min_score, rec_dict):
        if item_id not in rec_dict:
            return min_score - random.random()
        else:
            return rec_dict[item_id]
