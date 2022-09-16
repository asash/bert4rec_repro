import calendar
import json
import datetime
import os
import numpy as np

import tqdm
from aprec.api.action import Action
from aprec.utils.item_id import ItemId
from aprec.utils.os_utils import get_dir, shell


YELP_DIR = "data/yelp"
YELP_RAW_FILE = "yelp_dataset.tar"
YELP_URL = "https://www.yelp.com/dataset"
YELP_REVIEWS_FILE = "yelp_academic_dataset_review.json"
YELP_PROCESSED_FILE = "yelp.dat"
TOTAL_REVIEWS = 6990280

YELP_DATA_DIR = os.path.join(get_dir(), YELP_DIR)

def get_yelp_tar_file():
    full_filename = os.path.join(YELP_DATA_DIR, YELP_RAW_FILE)
    if not (os.path.isfile(full_filename)):
        raise Exception(f"We do not support automatic download for Yelp dataset.\n" +
                            f"Please download it manually from {YELP_URL} and put it into {YELP_DATA_DIR}")
    return full_filename


def preprocess(reviews_file):
    output_file = os.path.join(YELP_DATA_DIR, YELP_PROCESSED_FILE)
    if os.path.isfile(output_file):
        return output_file
    users = ItemId()
    items = ItemId()
    actions = []
    print("preprocessing yelp dataset")
    with open(reviews_file) as input:
        for line in tqdm.tqdm(input, ascii=True, total=TOTAL_REVIEWS):
            doc = json.loads(line)
            user_id = users.get_id(doc['user_id']) 
            item_id = items.get_id(doc['business_id'])
            timestamp = calendar.timegm(datetime.datetime.strptime(doc['date'],  "%Y-%m-%d %H:%M:%S").timetuple())
            actions.append((user_id, item_id, timestamp))

    actions.sort(key = lambda x: x[2])

    actions_np = np.array(actions, dtype='int32')
    fp = np.memmap(output_file, shape=actions_np.shape, dtype='int32', mode='w+')
    fp[:] = actions_np[:]
    fp.flush()
    return output_file

def extract_yelp_dataset(yelp_tar_file):
    reviews_file = os.path.join(YELP_DATA_DIR, YELP_REVIEWS_FILE)
    if os.path.isfile(reviews_file):
        return reviews_file
    shell(f"tar xvf {yelp_tar_file} -C {YELP_DATA_DIR}")
    return reviews_file


def get_yelp_actions(binary_file, max_actions):
    data = np.memmap(binary_file, shape=(TOTAL_REVIEWS, 3), dtype='int32', mode='r')
    dataset = []
    for i in range(max_actions):
        dataset.append(Action(user_id=int(data[i, 0]), item_id=int(data[i, 1]), timestamp=int(data[i, 2])))
    return dataset



def get_yelp_dataset(max_actions = TOTAL_REVIEWS):
    tar_file = get_yelp_tar_file()
    reviews_file = extract_yelp_dataset(tar_file)
    binary_file = preprocess(reviews_file)
    actions = get_yelp_actions(binary_file, max_actions)
    return actions