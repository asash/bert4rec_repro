from typing import Callable

import numpy as np

from .targetsplitter import TargetSplitter


def exponential_importance(p: float) -> Callable[[float, float], float]:
    def exponential_importance_func(n: float, k: float) -> float:
        res: float = p ** (n - k)
        return res
    return exponential_importance_func


def linear_importance(a: float = 1, b: float = 1) -> Callable[[float, float], float]:
    return lambda n, k: a * k + b


class RecencySequenceSampling(TargetSplitter):
    # recency importance is a function that defines the chances of k-th element
    # to be sampled as a positive in the sequence of the length n

    def __init__(
        self, max_pct, recency_importance=exponential_importance(0.8), seed=31337
    ) -> None:
        super().__init__()
        self.max_pct = max_pct
        self.recency_iportnace = recency_importance
        self.random = np.random.default_rng(seed=seed)

    def split(self, sequence):
        if len(sequence) == 0:
            return [], []
        target = set()
        cnt = max(1, int(len(sequence) * self.max_pct))

        def recency_importance_func(j: int) -> float:
            res: float = self.recency_iportnace(len(sequence), j)
            return res

        f_vals = np.array([recency_importance_func(i)
                           for i in range(len(sequence))])
        f_sum = sum(f_vals)
        sampled_idx = set(
            self.random.choice(
                range(len(sequence)), cnt, p=f_vals / f_sum, replace=True
            )
        )
        input = list()
        for i in range(len(sequence)):
            if i not in sampled_idx:
                input.append(sequence[i])
            else:
                target.add(sequence[i])
        return input, list(target)
