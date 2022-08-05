import numpy as np
from aprec.recommenders.recommender import Recommender


class RandomRecommender(Recommender):
    def __init__(self):
        self.items_set = set()

    def add_action(self, action):
        self.items_set.add(action.item_id)

    def rebuild_model(self):
        self.items = list(self.items_set)

    def recommend(self, user_id, limit, features=None):
        recommended_items = np.random.choice(self.items, limit, replace=False)
        result = []
        current_score = 1.0
        for item in recommended_items:
            result.append((item, current_score))
            current_score *= 0.9
        return result
