import random
import numpy as np
from scipy.sparse.csr import csr_matrix


class TargetBuilder(object):
    def __init__(self):
        pass

    def set_n_items(self, n):
        self.n_items = n
    
    def set_sequence_len(self, sequence_len):
        self.sequence_len = sequence_len
    
    def build(self, user_targets):
        raise NotImplementedError()

    def get_targets(self, start, end):
        raise NotImplementedError()



