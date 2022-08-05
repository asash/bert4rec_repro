from collections import Counter, defaultdict
import random

from aprec.evaluation.metrics.metric import Metric
import numpy as np
from tqdm import tqdm


class PairwiseCosSim(Metric):
    def __init__(self, actions, k):
        print("init pairwise_cos_sim...")
        self.name = "pairwise_cos_sim@{}".format(k)
        self.k = k
        self.max_actions_per_user = 500
        self.max_users = 500
        self.item_cnt = Counter()
        self.pair_cnt = Counter()

        user_sets = defaultdict(list)

        for action in actions:
            user_sets[action.user_id].append(action.item_id)

        for user_id in np.random.choice(list(user_sets.keys()), min(self.max_users, len(user_sets)), replace=False):
            random.shuffle(user_sets[user_id])
            for item1 in user_sets[user_id][:self.max_actions_per_user]:
                self.item_cnt[item1] += 1
                for item2 in user_sets[user_id][:self.max_actions_per_user]:
                    if item1 != item2:
                        self.pair_cnt[(item1, item2)] += 1
        self.item_cnt = dict(self.item_cnt)
        self.pair_cnt = dict(self.pair_cnt)
        print("init done...")

    def __call__(self, recommendations, actual_actions):
        items = [recommendation[0] for recommendation in recommendations[:self.k]]
        pairs = 0
        s = 0
        for item1 in items:
            for item2 in items:
                if (item1 != item2):
                    pairs += 1
                    if (item1, item2) in self.pair_cnt:
                        s += self.pair_cnt[(item1, item2)] ** 2 / (self.item_cnt[item1] * self.item_cnt[(item2)])
        if pairs == 0: return 0
        return s/pairs



