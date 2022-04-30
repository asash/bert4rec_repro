import numpy as np
from scipy.sparse.csr import csr_matrix
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder


class FullMatrixTargetsBuilder(TargetBuilder):
    def __init__(self, max_target_label=1.0, target_decay=1.0, min_target_val=0.1):
        self.max_target_label = max_target_label
        self.target_decay = target_decay
        self.min_target_val = min_target_val

    def build(self, user_targets):
        rows = []
        cols = []
        vals = []
        for i in range(len(user_targets)):
            cur_val = self.max_target_label 
            for action_num in range(len(user_targets[i])):
                action = user_targets[i][action_num]
                rows.append(i)
                cols.append(action[1])
                vals.append(cur_val)
                cur_val *= self.target_decay
                if cur_val < self.min_target_val:
                    cur_val = self.min_target_val
        self.target_matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_targets), self.n_items),
                                                                                        dtype='float32')
    def get_targets(self, start, end):
        target_inputs = [] 
        target_outputs = np.array(self.target_matrix[start:end].todense())
        return target_inputs, target_outputs