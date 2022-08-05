from collections import Counter
from aprec.api.user import User
from aprec.recommenders.recommender import Recommender


class ConditionalTopRecommender(Recommender):
    """
    This recommender calculates top items based on some condition. For example, we want to recommend
    the most popular hotel in the city, not globally (for global top we can use @TopRecommender).
    """
    def __init__(self, conditional_field: str):
        self.conditional_field: str = conditional_field
        self.items_counts: dict = dict()
        self.precalculated_top_items: dict = dict()
        self.user_field_values: dict = dict()
    
    def add_user(self, user: User):
        if self.conditional_field in user.cat_features:
            self.user_field_values[user.user_id] = user.cat_features[self.conditional_field]
        

    def add_action(self, action):

        if self.conditional_field in action.data:
            field_value = action.data[self.conditional_field]
        elif action.user_id in self.user_field_values:
            field_value = self.user_field_values[action.user_id]
        else:
            field_value = "N/A"
        if field_value not in self.items_counts:
            self.items_counts[field_value] = Counter()
        self.user_field_values[action.user_id] = field_value

        if action.item_id is not None:
            self.items_counts[field_value][action.item_id] += 1

    def rebuild_model(self):
        self.precalculated_top_items = {
            field_value: counter.most_common() for field_value, counter in self.items_counts.items()
        }

    def recommend(self, user_id, limit, features=None):
        if user_id not in self.user_field_values:
            field_value = "N/A"
        else:
            field_value = self.user_field_values[user_id]
        return self.precalculated_top_items.get(field_value, [])[:limit]

    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def name(self):
        return "ConditionalTopItemsRecommender"
