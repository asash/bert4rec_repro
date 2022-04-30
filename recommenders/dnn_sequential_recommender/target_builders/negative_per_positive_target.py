from random import Random

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder


class NegativePerPositiveTargetBuilder(TargetBuilder):
    def __init__(self, sequence_len=64, random_seed=31337):
        self.random = Random()
        self.random.seed(random_seed) 
        self.sequence_len = sequence_len

    def build(self, user_targets):
        self.inputs = []
        self.targets = []
        for i in range(len(user_targets)):
            user_inputs = []
            targets_for_user = [] 
            seq = user_targets[i]
            if len(seq) < self.sequence_len:
                user_inputs += [[self.n_items, self.n_items]] * (self.sequence_len - len(seq))
                targets_for_user += [[-1.0, -1.0]] * (self.sequence_len - len(seq))
            for target in seq[-self.sequence_len:]:
                positive = target[1]
                negative = self.random.randint(0, self.n_items - 1)
                while negative == positive:
                    negative = self.random.randint(0, self.n_items - 1)
                user_inputs.append([positive, negative])
                targets_for_user.append([1.0, 0.0])
            self.inputs.append(user_inputs)
            self.targets.append(targets_for_user)
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
    
    def get_targets(self, start, end):
        return [self.inputs[start:end]], self.targets[start:end]




