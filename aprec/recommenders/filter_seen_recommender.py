from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.recommenders.recommender import Recommender
from collections import defaultdict

INF = float('inf')
class FilterSeenRecommender(Recommender):
    def __init__(self, recommender):
        super().__init__()
        self.recommender = recommender
        self.user_seen = defaultdict(set)

    def name(self):
        return self.recommender.name() + "FilterSeen"

    def add_user(self, user):
        self.recommender.add_user(user)

    def add_item(self, item):
        self.recommender.add_item(item)


    def add_action(self, action):
        self.user_seen[action.user_id].add(action.item_id) 
        self.recommender.add_action(action)

    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.recommender.add_test_items_ranking_request(request)


    def rebuild_model(self):
        self.recommender.rebuild_model()

    def recommend(self, user_id, limit, features=None):
        user_seen_cnt = len(self.user_seen[user_id])
        raw = self.recommender.recommend(user_id, limit + user_seen_cnt)
        filtered = filter(lambda item_score: item_score[0] not in self.user_seen[user_id], raw)
        return list(filtered)[:limit]

    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def recommend_by_items(self, items_list, limit):
        raw = self.recommender.recommend_by_items(items_list, limit + len(items_list))
        filtered = filter(lambda item_score: item_score[0] not in items_list, raw)
        return list(filtered)[:limit]

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)

    def get_metadata(self):
        metadata = self.recommender.get_metadata()
        metadata['proxy_model'] = 'filter_seen_recommender'
        return metadata


    def  set_out_dir(self, out_dir):
        self.recommender.set_out_dir(out_dir)

    def get_item_rankings(self):
        base_rankings = self.recommender.get_item_rankings()
        result = {}
        for user_id in base_rankings:
            request_result = []
            for item, score in base_rankings[user_id]:
                if item in self.user_seen[user_id]:
                    request_result.append((item, -INF))
                else:
                    request_result.append((item, score))
            request_result.sort(key = lambda x: -x[1])
            result[user_id] = request_result
        return result


    def set_val_users(self, val_users):
        self.recommender.set_val_users(val_users)
