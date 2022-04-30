import random
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter


class RandomSplitter(TargetSplitter):
    def __init__(self, seed=31337, target_chance = 0.25) -> None:
        self.random = random.Random()
        self.random.seed(seed)
        self.target_chance = target_chance
        super().__init__()
    
    def split(self, sequence):
        input = []
        target = []
        for item in sequence:
            if self.random.random() < self.target_chance:
                target.append(item)
            else:
                input.append(item)
        return input, target
