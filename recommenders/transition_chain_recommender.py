import math
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from aprec.recommenders.recommender import Recommender
from aprec.recommenders.top_recommender import TopRecommender


class TransitionsChainRecommender(Recommender):
    """
    This recommender was written for situation when we have sequence of events for each user and we want
    to predict the last element of this sequence.
    During the training phase we calculate counts between events in a sequence and last event (target).
    During the inference phase we take all events of user and sum up all counts to predict next event.
    """
    def __init__(self):
        self.top_recommender = TopRecommender()
        self.item_id_to_index: dict = dict()
        self.index_to_item_id: dict = dict()
        self.items_count: int = 0
        self.transition_matrix = defaultdict(Counter)

        self.user_to_items: dict = dict()

    def add_action(self, action):
        if action.item_id not in self.item_id_to_index:
            self.item_id_to_index[action.item_id] = self.items_count
            self.index_to_item_id[self.items_count] = action.item_id
            self.items_count += 1

        if action.user_id not in self.user_to_items:
            self.user_to_items[action.user_id] = [action.item_id]
        else:
            self.user_to_items[action.user_id].append(action.item_id)
        self.top_recommender.add_action(action)

    def rebuild_model(self):
        self.top_recommender.rebuild_model()
        self.transition_matrix = defaultdict(Counter)
        df = Counter()
        for _, items in self.user_to_items.items():
            for item in set(items):
                df[item] += 1

        df = dict(df)
        idf = dict()
        for item in df:
            idf[item] = len(self.user_to_items) / math.log(df[item] + 1)


        print("building transitions matrix...")
        for _, items in tqdm(self.user_to_items.items()):
            for t in range(1, len(items)):
                target_item = items[t]
                for item_id in items[:t]:
                    self.transition_matrix[self.item_id_to_index[item_id]][self.item_id_to_index[target_item]] += 1
        self.graph = defaultdict(list)

        print("caching predictions...")
        for start in tqdm(self.transition_matrix):
            for stop in self.transition_matrix[start].most_common(500):
                self.graph[start].append(stop)
        pass

    def recommend(self, user_id, limit, features=None):
        if user_id not in self.user_to_items:
            return [] #"New user without history"
        return self.recommend_by_items(self.user_to_items[user_id], limit)

    def recommend_by_items(self, items_list, limit):
        last_items_indices = [self.item_id_to_index[idx] for idx in items_list]
        sums = defaultdict(lambda: 0)
        for item_idx in last_items_indices:
            for next, score in self.graph[item_idx]:
                sums[next] += score
        predictions = sorted(sums.keys(), key=lambda x: -sums[x])
        result = []
        recommended = set()
        for prediction in predictions:
            item = self.index_to_item_id[prediction]
            score = sums[prediction]
            recommended.add(item)
            result.append((item, score))

        if len(recommended) < limit:
            for item, score in self.top_recommender.recommend(0, 2 * limit):
                if item in recommended:
                    continue
                result.append((item, 0))
                if len(result) >= limit:
                    break
        return result[:limit]


    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def name(self):
        return "TransitionsChainRecommender"
