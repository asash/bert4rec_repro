import random

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder


class SampledMatrixBuilder(TargetBuilder):
    def __init__(self, max_target_label=1.0, target_decay=1.0, min_target_val=0.1, n_samples=101):
        self.max_target_label = max_target_label
        self.target_decay = target_decay
        self.min_target_val = min_target_val
        self.n_samples = n_samples

    def build(self, user_targets):
        all_items = list(range(self.n_items))
        self.target_matrix = []
        self.target_ids = []
        for i in range(len(user_targets)): 
            targets = []
            target_ids =  []
            sampled = set()
            cur_val = self.max_target_label 
            for action_num in range(len(user_targets[i])):
                action = user_targets[i][action_num]           
                targets.append(cur_val)
                target_ids.append(action[1])
                sampled.add(action[1])
                cur_val *= self.target_decay
                if cur_val < self.min_target_val:
                    cur_val = self.min_target_val
                sampled.add(action[1])
            while(len(targets) < self.n_samples):
                negatives = np.random.choice(all_items, self.n_samples - len(targets))
                for item_id in negatives:
                    if item_id not in sampled:
                        sampled.add(item_id)
                        target_ids.append(item_id)
                        targets.append(0.0)
            targets_with_ids = list(zip(targets, target_ids))
            random.shuffle(targets_with_ids)
            targets, target_ids = zip(*targets_with_ids)
            self.target_matrix.append(targets)
            self.target_ids.append(target_ids)
        self.target_matrix = np.array(self.target_matrix)
        self.target_ids = np.array(self.target_ids)

    def get_targets(self, start, end):
        target_inputs = [self.target_ids[start:end]]
        target_outputs = self.target_matrix[start:end]
        return target_inputs, target_outputs


