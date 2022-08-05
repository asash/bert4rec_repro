from .recommender import Recommender

class ConstantRecommender(Recommender):
    def __init__(self, recommendations):
        self.recommendations = recommendations

    def name(self):
        return "ConstantRecommender"

    def add_action(self, action):
        pass

    def rebuild_model(self):
        pass

    def recommend(self, user_id, limit, features=None):
        return self.recommendations[:limit]

    def get_similar_items(self, item_id, limit):
        return self.recommendations[:limit]

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)
