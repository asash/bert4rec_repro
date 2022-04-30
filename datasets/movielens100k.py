import os
import logging

from aprec.utils.os_utils import get_dir, console_logging, shell
from aprec.api.action import Action
from aprec.datasets.download_file import download_file

DATASET_NAME = 'ml-100k'
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/{}.zip".format(DATASET_NAME)
MOVIELENS_DIR = "data/movielens100k"
MOVIELENS_FILE = "movielens.zip"
MOVIELENS_FILE_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR, MOVIELENS_FILE)
MOVIELENS_DIR_ABSPATH = os.path.join(get_dir(), MOVIELENS_DIR)
RATINGS_FILE = os.path.join(MOVIELENS_DIR_ABSPATH, 'u.data')


def extract_movielens_dataset():
    if os.path.isfile(RATINGS_FILE):
        logging.info("movielens dataset is already extracted")
        return
    shell("unzip -o {} -d {}".format(MOVIELENS_FILE_ABSPATH, MOVIELENS_DIR_ABSPATH))
    dataset_dir = os.path.join(MOVIELENS_DIR_ABSPATH, DATASET_NAME)
    for filename in os.listdir(dataset_dir):
        shell("mv {} {}".format(os.path.join(dataset_dir, filename), MOVIELENS_DIR_ABSPATH))
    shell("rm -rf {}".format(dataset_dir))


def prepare_data():
    download_file(MOVIELENS_URL,  MOVIELENS_FILE, MOVIELENS_DIR)
    extract_movielens_dataset()


def get_movielens100k_actions(min_rating=4.0):
    prepare_data()
    with open(RATINGS_FILE, 'r') as data_file:
        i = 0
        for line in data_file:
            i += 1
            user_id, movie_id, rating_str, timestamp_str = line.strip().split('\t')
            rating = float(rating_str)
            timestamp = int(timestamp_str)
            if rating >= min_rating:
                yield Action(user_id, movie_id, timestamp, {"rating": rating})


if __name__ == "__main__":
    console_logging()
    prepare_data()
