import numpy as np
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.history_vectorizer import HistoryVectorizer

class AddMaskHistoryVectorizer(HistoryVectorizer):
    def __call__(self, user_actions):
        mask = self.padding_value + 1
        if len(user_actions) >= self.sequence_len - 1:
            return np.array([action[1] for action in user_actions[-self.sequence_len + 1:]] + [mask])
        else:
            n_special = self.sequence_len - 1  - len(user_actions)
            result_list = [self.padding_value] * n_special + [action[1] for action in user_actions] + [mask]
            return np.array(result_list)
