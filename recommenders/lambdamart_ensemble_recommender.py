import gzip
import os
from re import I
import sys
from typing import List

import lightgbm
from tqdm import tqdm
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.api.user import User
from aprec.recommenders.featurizer import Featurizer
from aprec.recommenders.recommender import Recommender
import numpy as np
from lightgbm import Dataset

class LambdaMARTEnsembleRecommender(Recommender):
    def __init__(self, candidates_selection_recommender: Recommender, 
                                other_recommenders: List[Recommender], 
                                n_ensemble_users = 1000,
                                n_ensemble_val_users = 256,
                                candidates_limit = 1000, 
                                ndcg_at = 40,
                                num_leaves = 15,
                                max_trees = 20000,
                                early_stopping=1000,
                                lambda_l2=0.0,
                                booster='gbdt',
                                featurizer: Featurizer = None, 
                                recently_interacted_hours = None, #only users who interacted within specificed time window will be selected for ensembling. 
                                log_dir = None, #write dataset with user features for offline experimentation
                                ):
        super().__init__()
        self.candidates_selection_recommender = candidates_selection_recommender
        self.other_recommenders = other_recommenders
        self.n_ensemble_users = n_ensemble_users
        self.user_actions = {}
        self.candidates_limit = candidates_limit
        self.max_trees = max_trees 
        self.early_stopping=early_stopping
        self.booster = booster
        self.ndcg_at = ndcg_at
        self.n_ensemble_val_users = n_ensemble_val_users
        self.num_leaves = num_leaves
        self.lambda_l2 = lambda_l2
        self.featurizer = featurizer
        self.recently_interacted_hours = recently_interacted_hours
        self.latest_action_time = 0
        self.set_out_dir(log_dir)


    def set_out_dir(self, log_dir):
        self.log_dir = log_dir 
        if log_dir is not None:
            self.train_users_file  = gzip.open(os.path.join(log_dir, "ensemble_train.csv.gz"), "w") 
            self.val_users_file  = gzip.open(os.path.join(log_dir, "ensemble_val.csv.gz"), "w") 
            self.predictions_file  = gzip.open(os.path.join(log_dir, "ensemble_prediction_features.csv.gz"), "w")
        else:
            self.train_users_file  = None 
            self.val_users_file  =  None
            self.predictions_file = None 

    
    def add_item(self, item: Item):
        self.candidates_selection_recommender.add_item(item)
        for other_recommender in self.other_recommenders:
            self.other_recommenders[other_recommender].add_item(item)
        if self.featurizer is not None:
            self.featurizer.add_item(item)

    def add_user(self, user: User):
        self.candidates_selection_recommender.add_user(user)
        for other_recommender in self.other_recommenders:
            self.other_recommenders[other_recommender].add_user(user)
        if self.featurizer is not None:
            self.featurizer.add_user(user)

    def add_action(self, action: Action):
        self.latest_action_time = max(self.latest_action_time, action.timestamp)
        if action.user_id not in self.user_actions:
            self.user_actions[action.user_id] = [action]
        else:
            self.user_actions[action.user_id].append(action)

    def rebuild_model(self):
        all_users = list(self.user_actions.keys())
        eligible_users = set()
        if self.recently_interacted_hours is not None:
            for user in self.user_actions:
                if (self.latest_action_time - self.user_actions[user][-1].timestamp) < self.recently_interacted_hours * 3600:
                    eligible_users.add(user)
        else:
            eligible_users = self.user_actions.keys()
        ensemble_users_selection = list(eligible_users - set(self.val_users))
        ensemble_users = set(np.random.choice(ensemble_users_selection, self.n_ensemble_users))
        ensemble_val_users_selection = list(eligible_users - set(self.val_users) - ensemble_users)
        ensemble_val_users = set(np.random.choice(ensemble_val_users_selection, self.n_ensemble_val_users))
        
        selected_users = ensemble_users.union(ensemble_val_users)

        for user in all_users:
            if user not in selected_users:
                all_actions = self.user_actions[user]
            else:
                all_actions = self.user_actions[user][:-1]

            for action in all_actions:
                self.candidates_selection_recommender.add_action(action)
                for recommender in self.other_recommenders:
                    self.other_recommenders[recommender].add_action(action)
                if self.featurizer is not None:
                    self.featurizer.add_action(action)
        if self.featurizer is not None:
            print("rebuilding featurizer...")
            self.featurizer.build()
        print("rebuilding candidates selection recommender...")
        self.candidates_selection_recommender.rebuild_model()

        for other_recommender in self.other_recommenders:
            print(f"rebuilding recommender {other_recommender}")
            self.other_recommenders[other_recommender].rebuild_model()


        print ("building ensemble train dataset...")
        train_dataset = self.get_data(ensemble_users, self.train_users_file)
        print ("building ensemble val dataset...")
        val_dataset = self.get_data(ensemble_val_users, self.val_users_file)

        if self.log_dir is not None:
            self.train_users_file.close()
            self.val_users_file.close()

        self.ranker = lightgbm.train(
            params={
             'objective': 'lambdarank',
             'eval_at': self.ndcg_at,
             'boosting': self.booster, 
             'num_leaves': self.num_leaves, 
             'lambda_l2': self.lambda_l2
            },
            train_set=train_dataset, 
            valid_sets=[val_dataset], 
            num_boost_round=self.max_trees,
            early_stopping_rounds=self.early_stopping
        )
        feature_names = train_dataset.feature_name
        if self.predictions_file is not None:
            self.predictions_file.write(f"user_id;item_id;target;{';'.join(feature_names)}\n".encode('utf-8'))

    
    def get_metadata(self):
        feature_importance =  list(zip(self.ranker.feature_name(), self.ranker.feature_importance()))
        result = {}
        result['feature_importance'] = []
        for feature, score in sorted(feature_importance, key=lambda x: -x[1]):
            result['feature_importance'].append((feature, int(score)))
        return result


    def get_data(self, users, log_file):
        features_list = ['candidate_recommender_idx', 'candidate_recommender_score']
        for recommender in self.other_recommenders:
            features_list += (f'is_present_in_{recommender}', f'{recommender}_idx', f'{recommender}_score')

        samples = []
        target = []
        group = []
        user_ids = []
        item_ids  = []
        
        for user_id in tqdm(users, ascii=True):
            candidates = self.build_candidates(user_id)
            target_id = self.user_actions[user_id][-1].item_id
            for candidate in candidates:
                samples.append(candidates[candidate])
                target.append(int(candidate == target_id))
                user_ids.append(user_id)
                item_ids.append(candidate)
            group.append(len(candidates))
        if self.featurizer is not None:
            features_list += self.featurizer.feature_names
        if log_file is not None:
            log_file.write(f"user_id;item_id;target;{';'.join(features_list)}\n".encode('utf-8'))
            self.log_candidates(log_file, user_ids, item_ids, samples, target)
        return Dataset(np.array(samples),label=target, group=group, feature_name=features_list, free_raw_data=False).construct()

    def recommend(self, user_id, limit: int, features=None):
        candidates = self.build_candidates(user_id)
        items = []
        features = []
        for candidate in candidates:
            items.append(candidate)
            features.append(candidates[candidate])
        user_ids = [user_id] * len(items)
        scores = self.ranker.predict(features) 
        recs = list(zip(items, scores))
        if self.predictions_file is not None:
            self.log_candidates(self.predictions_file, user_ids, items, features, scores)
        return sorted(recs, key=lambda x: -x[1])[:limit]

    def log_candidates(self, logfile, user_ids, items, features, targets=None):
        for idx in range(len(items)):
            if targets is None:
                target = 0
            else:
                target = targets[idx]
            features_str=";".join((str(feature) for feature in features[idx]))
            log_str = f"{user_ids[idx]};{items[idx]};{target};{features_str}\n".encode('utf-8')
            logfile.write(log_str)
        logfile.flush()
            
        

    
    def build_candidates(self, user_id):
        candidates = self.candidates_selection_recommender.recommend(user_id, limit=self.candidates_limit)
        candidate_features = {}
        for idx, candidate in enumerate(candidates):
            candidate_features[candidate[0]] = [idx, candidate[1]]

        cnt = 1
        for recommender in self.other_recommenders:
            cnt += 1
            recs = self.other_recommenders[recommender].recommend(user_id, limit=self.candidates_limit)
            recommender_processed_candidates = set()
            for idx, candidate in enumerate(recs):
                if candidate[0] in candidate_features:
                    candidate_features[candidate[0]] += [1, idx, candidate[1]]
                    recommender_processed_candidates.add(candidate[0])
            for candidate in candidate_features:
                if candidate not in recommender_processed_candidates:
                    candidate_features[candidate] += [0, self.candidates_limit, -1000]
                    recommender_processed_candidates.add(candidate)
        if self.featurizer is not None:
            candidate_ids = list(candidate_features.keys())
            features = self.featurizer.get_features(user_id, candidate_ids)
            for candidate, features in zip(candidate_ids, features):
                candidate_features[candidate] += features
        return candidate_features
    
    def set_val_users(self, val_users):
        self.val_users = val_users
        self.candidates_selection_recommender.set_val_users(val_users=val_users)
        for recommender in self.other_recommenders:
            self.other_recommenders[recommender].set_val_users(val_users)