from collections import defaultdict
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

import numpy as np

from aprec.datasets.datasets_register import DatasetsRegister

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 256)


def num_users(users, items, session_lens):
    return len(users)


def num_items(users, items, session_lens):
    return len(items)

def num_interactions(users, items, session_lens):
    return sum(session_lens)

def average_session_len(users, items, session_lens):
    return float(np.mean(session_lens))

def median_session_len(users, items, session_lens):
    return int(np.median(session_lens))

def min_session_len(users, items, session_lens):
    return int(np.min(session_lens))

def max_session_len(users, items, session_lens):
    return int(np.max(session_lens))


def p80_session_len(user, items, session_lens):
    return float(np.percentile(session_lens, 80))

def sparsity(users, items, session_lens):
    sum_interacted = 0
    for user in users:
        interacted_items = len(set(users[user]))
        sum_interacted += interacted_items
    return 1 - sum_interacted/(len(users)*len(items))

all_metrics = {
    "num_users": num_users, 
    "num_items": num_items, 
    "num_interactions": num_interactions, 
    "average_session_len": average_session_len, 
    "median_session_len": median_session_len, 
    "min_session_len": min_session_len, 
    "max_session_len": max_session_len, 
    "p80_session_len": p80_session_len,
    "sparsity": sparsity 
}


def dataset_stats(dataset, metrics, dataset_name=None):
    users = defaultdict(list)
    item_ids = set()
    for action in dataset:
        users[action.user_id].append(action)
        item_ids.add(action.item_id)
    session_lens = [len(users[user_id]) for user_id in users]
    result = {}
    for metric in metrics:
        if metric not in all_metrics:
            raise Exception(f"unknown dataset metric: {metric}")
        else:
            result[metric] = all_metrics[metric](users, item_ids, session_lens)

    if dataset_name is not None:
        result['name'] = dataset_name
    return result

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--datasets", required=True, help=f"Available Datasets: {','.join(DatasetsRegister().all_datasets())}")
    parser.add_argument("--metrics", required=False, help=f"Available Columns: {','.join(all_metrics.keys())}", default=','.join(all_metrics.keys()))
    parser.add_argument("--latex_table", required=False, default=False)
    args = parser.parse_args()

    metrics = args.metrics.split(",")
    datasets = args.datasets.split(",")
    for dataset in datasets:
        if dataset not in DatasetsRegister().all_datasets():
            print(f"unknown dataset {dataset}")
            exit(1)
    docs = []
    for dataset_name in tqdm(datasets):
        dataset = DatasetsRegister()[dataset_name]()
        stats = dataset_stats(dataset, metrics, dataset_name=dataset_name)
        docs.append(stats)
        del(dataset)
    df = pd.DataFrame(docs).set_index("name")
    if not args.latex_table:
        print(df)
    else:
        print(df.to_latex())
