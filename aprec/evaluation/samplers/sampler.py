class TargetItemSampler(object):
    def __init__(self, target_size) -> None:
        super().__init__()
        self.target_size = target_size
    
    def set_actions(self, all_actions, test_actions):
        self.actions = all_actions
        self.test = test_actions


    def get_sampled_ranking_requests(self):
        raise NotImplementedError()