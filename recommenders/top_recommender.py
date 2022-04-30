from aprec.recommenders.recommender import Recommender
from collections import Counter

class TopRecommender(Recommender):
    def __init__(self, recency=1.0): #recency parameter controls how many actions are considered out of all actions
        super().__init__()
        self.items_counter=Counter()
        self.item_scores = {}
        self.actions = []
        self.recency = recency

    def add_action(self, action):
        self.actions.append(action)

    def rebuild_model(self):
        self.actions.sort(key=lambda x: x.timestamp)
        n_actions = int(len(self.actions) * self.recency)
        for action in self.actions[-n_actions:]:
            self.items_counter[action.item_id] += 1
        self.actions = []
        self.most_common = self.items_counter.most_common()
        for item, score in self.most_common:
            self.item_scores[item] = score

    def recommend(self, user_id, limit, features=None):
        return self.most_common[:limit]

    def get_metadata(self):
        return {"top 20 items":  self.most_common[:20]}


    def get_similar_items(self, item_id, limit):
        return self.most_common[:limit]

    def name(self):
        return "TopItemsRecommender"

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            request_result = []
            for item_id in request.item_ids:
                score = self.item_scores.get(item_id, 0)
                request_result.append((item_id, score))
            request_result.sort(key=lambda x: -x[1])
            result[request.user_id] = request_result
        return result



