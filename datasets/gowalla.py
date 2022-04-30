import datetime
import time
import dateutil.parser

from aprec.datasets.dataset_utils import gunzip
from aprec.datasets.download_file import download_file
from aprec.api.action import Action


GOWALLA_DATASET_URL='https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
DIR="data/gowalla"
GOWALLA_GZIPPED="gowalla.txt.gz"

def prepare_data():
    gowalla_file_zipped = download_file(GOWALLA_DATASET_URL,GOWALLA_GZIPPED, DIR)
    unzipped_gowalla_file = gunzip(gowalla_file_zipped)
    return unzipped_gowalla_file

def parse_line(line):
    user_id, timestamp, lat, lon, item_id = line.split("\t")
    timestamp = time.mktime(dateutil.parser.isoparse(timestamp).timetuple())
    return Action(user_id, item_id, timestamp)

def get_gowalla_dataset(max_actions=None):
    dataset_file = prepare_data()
    actions = []
    for line in open(dataset_file):
        actions.append(parse_line(line.strip()))
        if max_actions is not None and len(actions) >= max_actions:
            break
    return actions