from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter


class SequenceContinuation(TargetSplitter):
    def __init__(self) -> None:
        super().__init__()
    
    def split(self, sequence, max_targets=1):
        if len(sequence) == 0:
            return [], []
        train = sequence[:-max_targets]
        target = sequence[-max_targets:]
        return train, target