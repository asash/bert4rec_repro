from random import Random
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter


class RandomFractionSplitter(TargetSplitter):
    def __init__(self, min_targets=1, random_seed=31337) -> None:
        super().__init__()
        self.min_targets = min_targets
        self.random = Random(random_seed)
    
    def split(self, sequence):
        if(len(sequence) == 0):
            return [], []
        target_actions = self.random.randint(1, max(len(sequence) -1, 1)) 
        train_actions = len(sequence) - target_actions
        return sequence[:train_actions], sequence[-target_actions:]

