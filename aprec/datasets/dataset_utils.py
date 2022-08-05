from collections import Counter
import gzip
import logging
import os

from aprec.utils.os_utils import get_dir, mkdir_p, mkdir_p_local, shell


def filter_popular_items(actions_generator, max_actions):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    popular_items = set([item_id for (item_id, cnt) in items_counter.most_common(max_actions)])
    return filter(lambda action: action.item_id in popular_items, actions)

def filter_cold_users(actions_generator, min_actions_per_user = 0):
    actions = []
    user_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        user_counter[action.user_id] += 1
    return filter(lambda action: user_counter[action.user_id] >= min_actions_per_user, actions)

def unzip(zipped_file, unzip_dir):
    full_dir_name = os.path.join(get_dir(), unzip_dir)
    if os.path.isdir(full_dir_name):
        logging.info(f"{unzip_dir} already exists, skipping")
    else:
        mkdir_p(full_dir_name)
        shell(f"unzip -o {zipped_file} -d {full_dir_name}")
    return full_dir_name

def gunzip(gzip_file):
    full_file_name = os.path.abspath(gzip_file)
    if not(gzip_file.endswith(".gz")):
        raise Exception(f"{gzip_file} is not a gzip file")
    unzipped_file_name = full_file_name[:-3]
    if os.path.isfile(unzipped_file_name):
        logging.info(f"{unzipped_file_name} already exists, skipping")
        return unzipped_file_name

    with gzip.open(full_file_name) as input:
        data = input.read()
        with open(unzipped_file_name, 'wb') as output:
            output.write(data)
    return unzipped_file_name 
    