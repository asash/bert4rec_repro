import copy
import gzip
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from aprec import recommenders

from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.evaluation.samplers.sampler import TargetItemSampler
from aprec.utils.os_utils import mkdir_p
from aprec.evaluation.filter_cold_start import filter_cold_start
from aprec.evaluation.metrics.sampled_proxy_metric import SampledProxy
from aprec.evaluation.evaluation_utils import group_by_user

def evaluate_recommender(recommender, test_actions,
                         metrics, out_dir, recommender_name,
                         features_from_test=None,
                         recommendations_limit=900,
                         evaluate_on_samples = False,
                         ):

    test_actions_by_user = group_by_user(test_actions)
    metric_sum = defaultdict(lambda: 0.0)
    sampled_metric_sum = defaultdict(lambda: 0.0)
    all_user_ids = list(test_actions_by_user.keys())
    requests = []
    for user_id in all_user_ids:
        if features_from_test is not None:
            requests.append((user_id, features_from_test(test_actions)))
        else:
            requests.append((user_id, None))


    print("generating predictions...")
    all_predictions = recommender.recommend_batch(requests, recommendations_limit)

    if evaluate_on_samples:
        sampled_rankings = recommender.get_item_rankings()

    print('calculating metrics...')
    user_docs = []
    for i in tqdm(range(len(all_user_ids)), ascii=True):
        user_id = all_user_ids[i]
        predictions = all_predictions[i]
        user_test_actions = test_actions_by_user[user_id]
        user_doc = {"user_id": user_id,
                    "metrics": {},
                    "test_actions": [action.to_json() for action in user_test_actions],
                    "predictions": [(prediction[0], float(prediction[1])) for prediction in predictions],
                    }
        if evaluate_on_samples:
            user_doc["sampled_metrics"] = {}
        for metric in metrics:
            metric_value = metric(predictions, test_actions_by_user[user_id])
            metric_sum[metric.name] += metric_value
            user_doc["metrics"][metric.name] = metric_value
            if evaluate_on_samples:
                sampled_metric_value = metric(sampled_rankings[user_id], test_actions_by_user[user_id])
                sampled_metric_sum[metric.name] += sampled_metric_value
                user_doc["sampled_metrics"][metric.name] = sampled_metric_value

        user_docs.append(user_doc)

    mkdir_p(f"{out_dir}/predictions/")
    predictions_filename = f"{out_dir}/predictions/{recommender_name}.json.gz"
    with gzip.open(predictions_filename, "w") as output:
        for user_doc in user_docs:
            try:
                output.write(json.dumps(user_doc).encode("utf-8") + b"\n")
            except:
                pass

    result = {}
    sampled_result = {}
    for metric in metric_sum:
        result[metric] = metric_sum[metric]/len(test_actions_by_user)
        if evaluate_on_samples:
            sampled_result[metric] = sampled_metric_sum[metric]/len(test_actions_by_user)
    if evaluate_on_samples:
        result["sampled_metrics"] = sampled_result
    return result

class RecommendersEvaluator(object):
    def __init__(self, actions, recommenders, metrics, out_dir, data_splitter,
                 n_val_users, recommendations_limit, callbacks=(),
                 users=None,
                 items=None,
                 experiment_config=None,
                 target_items_sampler: TargetItemSampler = None,
                 remove_cold_start=True
                 ):
        self.actions = actions
        self.metrics = metrics
        self.recommenders = recommenders
        self.data_splitter = data_splitter
        self.callbacks = callbacks
        self.out_dir = out_dir
        self.features_from_test = None
        self.n_val_users = n_val_users
        self.train, self.test = self.data_splitter(actions)
        self.save_split(self.train, self.test)
        if remove_cold_start:
            self.test = filter_cold_start(self.train, self.test)
        self.users = users
        self.items = items
        all_train_user_ids = list(set([action.user_id for action in self.train]))
        self.recommendations_limit = recommendations_limit
        random.shuffle(all_train_user_ids)
        self.val_user_ids = all_train_user_ids[:self.n_val_users]
        self.sampled_requests = None
        if target_items_sampler is not None:
            target_items_sampler.set_actions(self.actions, self.test)
            self.sampled_requests = target_items_sampler.get_sampled_ranking_requests() 
        self.experiment_config = experiment_config

    def set_features_from_test(self, features_from_test):
        self.features_from_test = features_from_test

    def __call__(self):
        result = {"recommenders": {}}
        print(f"recommenders to evaluate:")
        for i, recommender_name in enumerate(self.recommenders):
            print(f"{i+1}. {recommender_name}")
        for recommender_name in self.recommenders:
            try:
                sys.stdout.write("!!!!!!!!!   ")
                print("evaluating {}".format(recommender_name))
                recommender = self.recommenders[recommender_name]()
                recommender.set_out_dir(self.out_dir)
                print("adding train actions...")
                for action in tqdm(self.train, ascii=True):
                    recommender.add_action(action)
                recommender.set_val_users(self.val_user_ids)
                print("rebuilding model...")

                if self.users is not None:
                    print("adding_users")
                    for user in self.users:
                        recommender.add_user(user)

                if self.items is not None:
                    print("adding items")
                    for item in self.items:
                        recommender.add_item(item)


                build_time_start = time.time()
                if self.sampled_requests is not None:
                    for request in self.sampled_requests:
                        recommender.add_test_items_ranking_request(request)

                recommender.rebuild_model()
                build_time_end = time.time()
                print("done")

                print("calculating metrics...")
                evaluate_time_start = time.time()
                evaluation_result = evaluate_recommender(recommender, self.test,
                                                         self.metrics, self.out_dir,
                                                         recommender_name, self.features_from_test,
                                                         recommendations_limit=self.recommendations_limit,
                                                         evaluate_on_samples=self.sampled_requests is not None)
                evaluate_time_end = time.time()
                print("calculating metrics...")
                evaluation_result['model_build_time'] =  build_time_end - build_time_start
                evaluation_result['model_inference_time'] =  evaluate_time_end - evaluate_time_start
                evaluation_result['model_metadata'] = copy.deepcopy(recommender.get_metadata())
                print("done")
                print(json.dumps(evaluation_result))
                result['recommenders'][recommender_name] = evaluation_result
                for callback in self.callbacks:
                    callback(recommender, recommender_name, evaluation_result, self.experiment_config)
                del(recommender)
            except Exception as ex:
                print(f"ERROR: exception during evaluating {recommender_name}")
                print(ex)

                print(traceback.format_exc())


                try:
                    del(recommender)
                except:
                    continue
        return result

    def save_split(self, train, test):
        self.save_actions(train, "train.json.gz")
        self.save_actions(test, "test.json.gz")

    def save_actions(self, actions, filename):
        with gzip.open(os.path.join(self.out_dir, filename), 'w') as output:
            for action in actions:
                output.write(action.to_json().encode('utf-8') + b"\n")