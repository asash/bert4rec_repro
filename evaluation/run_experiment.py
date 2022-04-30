import os
import sys
import importlib.util
import json
import mmh3

from aprec.utils.os_utils import shell
from aprec.evaluation.evaluate_recommender import RecommendersEvaluator
from aprec.datasets.datasets_register import DatasetsRegister

import tensorflow as tf



def config():
    """ from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path"""

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if len(sys.argv) > 2:
        config.out_file = open(sys.argv[2], 'w')
        config.out_dir = os.path.dirname(sys.argv[2])
    else:
        config.out_file = sys.stdout
        config.out_dir = os.getcwd()

    return config


def real_hash(obj):
    str_val = str(obj)
    result = (mmh3.hash(str_val) + (1 << 31)) * 1.0 / ((1 << 32) - 1)
    return result

def run_experiment(config):
    result = []
    print(f"Dataset: {config.DATASET}")
    print("reading  data...")
    all_actions = [action for action in DatasetsRegister()[config.DATASET]()]
    print("done")
    callbacks = ()
    if hasattr(config, 'CALLBACKS'):
        callbacks = config.CALLBACKS

    for users_fraction in config.USERS_FRACTIONS:
        every_user = 1/users_fraction
        print("use one out of every {} users ({}% fraction)".format(every_user, users_fraction*100))
        actions = list(filter( lambda action: real_hash(action.user_id)  < users_fraction,
                         all_actions))
        print("actions in dataset: {}".format(len(actions)))
        item_id_set = set([action.item_id for action in actions])
        user_id_set = set([action.user_id for action in actions])

        if hasattr(config, 'N_VAL_USERS'):
            n_val_users = config.N_VAL_USERS
        else:
            n_val_users = len(user_id_set) // 10

        if hasattr(config, 'USERS'):
            users = config.USERS()
        else:
            users = None

        if hasattr(config, 'ITEMS'):
            items = config.ITEMS()
        else:
            items=None



        if hasattr(config, 'RECOMMENDATIONS_LIMIT'):
            recommendations_limit = config.RECOMMENDATIONS_LIMIT
        else:
            recommendations_limit = 900

        print("number of items in the dataset: {}".format(len(item_id_set)))
        print("number of users in the dataset: {}".format(len(user_id_set)))
        print("number of val_users: {}".format(n_val_users))
        print("evaluating...")

        data_splitter = config.SPLIT_STRATEGY
        target_items_sampler = None
        if hasattr(config, "TARGET_ITEMS_SAMPLER"):
            target_items_sampler = config.TARGET_ITEMS_SAMPLER
        
        filter_cold_start = True

        if hasattr(config, "FILTER_COLD_START"):
            filter_cold_start = config.FILTER_COLD_START
 
        
        recommender_evaluator = RecommendersEvaluator(actions,
                                                      config.RECOMMENDERS,
                                                      config.METRICS,
                                                      config.out_dir,
                                                      data_splitter,
                                                      n_val_users,
                                                      recommendations_limit,
                                                      callbacks,
                                                      users=users,
                                                      items=items,
                                                      experiment_config=config,
                                                      target_items_sampler=target_items_sampler, 
                                                      remove_cold_start=filter_cold_start
                                                      )

        if  hasattr(config, 'FEATURES_FROM_TEST'):
            recommender_evaluator.set_features_from_test(config.FEATURES_FROM_TEST)
        result_for_fraction = recommender_evaluator()
        result_for_fraction['users_fraction'] = users_fraction
        result_for_fraction['num_items'] = len(item_id_set)
        result_for_fraction['num_users'] = len(user_id_set)
        result.append(result_for_fraction)
        write_result(config, result)
        shell(f"python3 statistical_signifficance_test.py --predictions-dir={config.out_dir}/predictions/ "
              f"--output-file={config.out_dir}/statistical_signifficance.json")

def get_max_test_users(config):
    if hasattr(config, 'MAX_TEST_USERS'):
        max_test_users = config.MAX_TEST_USERS
    else:
        max_test_users = 943 #number of users in movielens 100k dataset
    return max_test_users


def write_result(config, result):
    if config.out_file != sys.stdout:
        config.out_file.seek(0)
    config.out_file.write(json.dumps(result, indent=4))
    if config.out_file != sys.stdout:
        config.out_file.truncate()
        config.out_file.flush()


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config = config()
    run_experiment(config)
    
