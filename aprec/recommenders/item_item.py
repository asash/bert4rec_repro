from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from collections import defaultdict, Counter


class ItemItemRecommender(Recommender):
    def __init__ (self, keep_max_items=200):
        self.users = ItemId()
        self.items = ItemId()
        self.item_cnt = Counter()
        self.pair_cnt = defaultdict(lambda: Counter())
        self.user_items = defaultdict(lambda: [])
        self.action_cnt = 0 
        self.keep_max_items = keep_max_items
        self.most_common = []
        
    def name(self):
        return "ItemItemRecommender"

    def add_action(self, action):
        user_id = self.users.get_id(action.user_id)
        item_id = self.items.get_id(action.item_id)
        self.action_cnt += 1
        self.item_cnt[item_id] += 1
        for other_item_id in self.user_items[user_id]:
            self.pair_cnt[item_id][other_item_id] += 1
            self.pair_cnt[other_item_id][item_id] += 1
        self.user_items[user_id].append(item_id)
        if len(self.user_items[user_id]) > self.keep_max_items:
            self.user_items[user_id] = self.user_items[user_id][-self.keep_max_items:]
            

    def rebuild_model(self):
        self.item_sims = defaultdict(lambda: [])
        for item_id in self.pair_cnt:
            for other_item_id in self.pair_cnt[item_id]:
                sim = self.get_similarity(item_id, other_item_id)
                self.item_sims[item_id].append((other_item_id, sim))
            self.item_sims[item_id].sort(key=lambda x: -x[1])
        self.most_common = self.item_cnt.most_common() 
        
    def recommend(self, user_id, limit, features=None):
        internal_user_id = self.users.get_id(user_id)
        user_history = self.user_items[internal_user_id]
        return self.predict_by_history(user_history, limit)

    def recommend_by_items(self, items, limit):
        items_internal = [self.items.get_id(item) for item in items]
        return self.predict_by_history(items_internal, limit)

    def predict_by_history(self, history, limit):
        recs = []
        for item in history:
            recs += self.item_sims[item]
        recs.sort(key = lambda x: -x[1])
        already_recommended = set()
        cnt  = 0
        result = []

        for rec in recs:
           if rec[0] in already_recommended:
                continue
           result.append((self.items.reverse_id(rec[0]), rec[1]))
           already_recommended.add(rec[0])
           cnt += 1
           if cnt >= limit:
                break
 

        for rec in self.most_common:
           if rec[0] in already_recommended:
               continue
           result.append((self.items.reverse_id(rec[0]), rec[1]))
           already_recommended.add(rec[0])
           cnt += 1
           if cnt >= limit:
               break
 

        return result


    def get_similarity(self, item_id1, item_id2):
        return self.pair_cnt[item_id1][item_id2] ** 2 / (self.item_cnt[item_id1] * self.item_cnt[item_id2])

    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)
