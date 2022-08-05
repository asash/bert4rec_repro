import copy

import numpy as np

from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter

class ItemsMasking(TargetSplitter):
    def __init__(self,  masking_prob = 0.2,
                 max_predictions_per_seq = 20,
                 random_seed = 31337, force_last=False, recency_importance = lambda n, k: 1) -> None:
        super().__init__()
        self.masking_prob = masking_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(random_seed)))
        self.force_last = force_last 
        self.recency_importance = recency_importance

    def split(self, sequence):
        seq = sequence[-self.seqence_len: ]
        seq_len = len(seq)

        if len(seq) < self.seqence_len:
            seq = [(-1, self.num_items)] * (self.seqence_len - len(seq)) + seq

        if not self.force_last:
            n_masks = min(self.max_predictions_per_seq,
                            max(1, int(round(len(sequence) * self.masking_prob))))
            sample_range = list(range(len(seq) - seq_len, len(seq)))
            rss_vals = np.array([self.recency_importance(self.seqence_len, pos) for pos in sample_range])
            rss_vals_sum = np.sum(rss_vals)
            probs = rss_vals / rss_vals_sum
            mask_positions = self.random.choice(sample_range, n_masks, p=probs, replace=False)
        else:
            n_masks = 1
            mask_positions = [len(seq) - 1]
        train = copy.deepcopy(seq)
        labels = []
        mask_token = self.num_items + 1 #self.num_items is used for padding
        for position in mask_positions:
            labels.append((position, seq[position]))
            train[position] = (train[position][0], mask_token)
        return train, (seq_len, labels)
    