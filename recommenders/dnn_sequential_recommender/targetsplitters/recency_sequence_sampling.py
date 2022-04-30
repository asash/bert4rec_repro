from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter
import numpy as np
from random import Random

def exponential_importance(p):
    return lambda n, k: p**(n - k)

def linear_importance(a=1, b=1):
    return lambda n, k: a*k+b

class RecencySequenceSampling(TargetSplitter):
    #recency importance is a function that defines the chances of k-th element 
    #to be sampled as a positive in the sequence of the length n

    def __init__(self, max_pct, recency_importance=exponential_importance(0.8), seed=31337) -> None:
        super().__init__()
        self.max_pct = max_pct
        self.recency_iportnace = recency_importance
        self.random = np.random.default_rng(seed=seed)

    
    def split(self, sequence):
        if len(sequence) == 0:
            return [], []
        target = set() 
        cnt = max(1, int(len(sequence)*self.max_pct))
        f = lambda j: self.recency_iportnace(len(sequence), j)
        f_vals = np.array([f(i) for i in range(len(sequence))])
        f_sum = sum(f_vals)
        sampled_idx = set(self.random.choice(range(len(sequence)), cnt, p=f_vals/f_sum, replace=True))
        input = list() 
        for i in range(len(sequence)):
            if i not in sampled_idx:
                input.append(sequence[i])
            else:
                target.add(sequence[i])
        return input, list(target)


