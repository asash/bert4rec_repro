from collections import Counter, defaultdict
import math
import sys
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.recommenders.featurizer import Featurizer
from tqdm import tqdm
from aprec.utils.item_id import ItemId

def dot(x, y):
    result = 0
    for i in x:
        result += x[i] * y.get(i, 0)
    return result

def vec_len(x):
    return math.sqrt(dot(x, x))

def norm(x):
    x_len = vec_len(x)
    res = {}
    for i in x:
        res[i] = x[i]/ x_len
    return res


class KionChallengeFeaturizer(Featurizer):
    def __init__(self):
        self.items = {}
        self.users = {}
        self.user_actions = defaultdict(list) 
        self.max_sim_actions_per_user=100
        self.max_feature_items = 5 
        self.item_pairs = Counter()
        self.item_counts = Counter()
        self.feature_names_initialized = False
        self.user_cat_features_initialized = False
        self.history_feature_names_initialized = False
        self.genre_counts = Counter()
        self.item_genres = defaultdict(set)
        self.item_genre_vectors = {}
        self.user_cat_feature_ids = defaultdict(ItemId) 
        self.user_cat_features =  defaultdict(dict)
        self.latest_timestamp = 0
    
    def add_item(self, item: Item):
        self.items[item.item_id] = item
        for type, val in item.cat_features:
            if type == 'genre':
                genre = val.strip().lower()
                self.item_genres[str(item.item_id)].add(genre)
                self.genre_counts[genre] += 1


    def add_user(self, user):
        user_id = str(user.user_id)
        self.users[user_id] = user
        for feature_type in user.cat_features:
            feature_val =  user.cat_features[feature_type]
            feature_id = self.user_cat_feature_ids[feature_type].get_id(feature_val)
            self.user_cat_features[user_id][feature_type] = feature_id
            pass
            

    def add_action(self, action: Action):
        self.user_actions[action.user_id].append(action)
        self.latest_timestamp = max(self.latest_timestamp, action.timestamp)
    
    def build(self):
        for item_id in self.item_genres:
            self.item_genre_vectors[item_id] ={}
            for genre in self.item_genres[item_id]:
                genre_cnt = self.genre_counts[genre]
                self.item_genre_vectors[item_id][genre] = math.log(len(self.item_genres) / (genre_cnt + 1)) 
            self.item_genre_vectors[item_id] = norm(self.item_genre_vectors[item_id])
            pass

        print("building item pair counts...")
        for user in tqdm(self.user_actions, ascii=True):
            actions = self.user_actions[user][-self.max_sim_actions_per_user:]
            for action in actions:
                    self.item_counts[action.item_id] += 1
            if len(actions) < 2:
                continue
            for action1 in actions:
                for action2 in actions:
                    if action1.item_id != action2.item_id:
                        self.item_pairs[(action1.item_id, action2.item_id)] += 1

    def get_features(self, user_id, items):
        user_history = self.user_actions.get(user_id, [])[-self.max_feature_items:][::-1]
        result = []
        for item in items:

            if user_id in self.user_actions:
                recency = self.latest_timestamp - self.user_actions[user_id][-1].timestamp
                session_len = len(self.user_actions[user_id])
            else:
                recency = sys.maxsize
                session_len = 0
            item_features = [recency, session_len]

            item_genre_vec = self.item_genre_vectors.get(item, {})
            user_cat_feature_names = []
            for feature_type in self.user_cat_feature_ids:
                features = [0] *self.user_cat_feature_ids[feature_type].size() 
                if not self.user_cat_features_initialized:
                    user_cat_feature_names += [f"{feature_type}_{i}" for i in range(self.user_cat_feature_ids[feature_type].size())]
                if user_id in self.user_cat_features:
                    features[self.user_cat_features[user_id][feature_type]] = 1
                item_features += features

            if not self.user_cat_features_initialized:
                self.user_cat_feature_names = user_cat_feature_names
                self.user_cat_features_initialized = True

            history_feature_names = []
            for i in range(self.max_feature_items):
                if i < len(user_history):
                    action = user_history[i]
                else:
                    action = None

                #popularity of historical movies
                if not self.history_feature_names_initialized:
                    history_feature_names.append(f"history_item_{i+1}_pop")
                if action is not None:
                    item_features.append(self.item_counts.get(action.item_id, 0))
                else:
                     item_features.append(0)

                #genre_sim
                if not self.history_feature_names_initialized:
                    history_feature_names.append(f"history_item_{i+1}_genre_sim")
                if action is not None:
                    genre_dict = self.item_genre_vectors.get(action.item_id, {})
                    item_features.append(dot(genre_dict, item_genre_vec))
                    pass
                else:
                     item_features.append(0)


                #similarity of historical movies
                if not self.history_feature_names_initialized:
                    history_feature_names.append(f"history_item_{i+1}_cos_sim")
                if action is not None:
                    cnt1 = self.item_counts.get(action.item_id, 0)
                    cnt2 = self.item_counts.get(item, 0)
                    pair_cnt = self.item_pairs.get((item, action.item_id), 0)
                    if (cnt1 == 0) or (cnt2 == 0) or pair_cnt == 0:
                        item_features.append(0)
                    else:
                        item_features.append(pair_cnt**2 / (cnt1 * cnt2))
                else:
                     item_features.append(0)
            if not self.history_feature_names_initialized:
                self.history_feature_names = history_feature_names
                self.history_feature_names_initialized = True
            result.append(item_features)
        if not self.feature_names_initialized:
            self.feature_names =['recency', 'session_len'] + self.user_cat_feature_names + self.history_feature_names
            self.feature_names_initialized = True
        return result

