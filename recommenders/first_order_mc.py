from argparse import Action
from collections import Counter, defaultdict
from aprec.recommenders.recommender import Recommender


class FirstOrderMarkovChainRecommender(Recommender):
    def __init__(self, cache_items=1000):
        super().__init__()
        self.user_actions = defaultdict(list)
        self.cache_items = cache_items

    def add_action(self, action: Action):
       self.user_actions[action.user_id].append(action.item_id)

    def rebuild_model(self):
        self.item_pairs_counter = defaultdict(Counter)
        for user in self.user_actions:
            for i in range(1, len(self.user_actions[user])):
                src = self.user_actions[user][i-1]
                dst = self.user_actions[user][i]
                self.item_pairs_counter[src][dst] += 1

        self.cache = defaultdict(list)
        for item in self.item_pairs_counter:
            self.cache[item] = self.item_pairs_counter[item].most_common(self.cache_items)

    def recommend(self, user_id, limit: int, features=None):
        if user_id not in self.user_actions:
            return []
        return self.cache[self.user_actions[user_id][-1]][:limit]

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            user_result = []
            user_id = request.user_id
            last_item = self.user_actions[user_id][-1]
            scores = self.item_pairs_counter[last_item]

            for item_id in request.item_ids:
                    score = scores.get(item_id, 0) 
                    user_result.append((item_id, score))
            user_result.sort(key=lambda x: -x[1])
            result[request.user_id] = user_result
        return result
    