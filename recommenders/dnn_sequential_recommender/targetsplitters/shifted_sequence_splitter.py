from random import Random
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter


class ShiftedSequenceSplitter(TargetSplitter):
    def __init__(self, max_len=100) -> None:
        super().__init__()
        self.max_len = max_len
    
    def split(self, sequence):
        train = sequence[-self.max_len - 1: -1]
        label = sequence[-len(train):]
        return train, label
    
