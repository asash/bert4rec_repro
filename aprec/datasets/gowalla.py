import time
from typing import Iterator, Optional

import dateutil.parser

from aprec.api.action import Action
from aprec.datasets.dataset_utils import gunzip
from aprec.datasets.download_file import download_file

GOWALLA_DATASET_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
DIR = "data/gowalla"
GOWALLA_GZIPPED = "gowalla.txt.gz"


def prepare_data() -> str:
    gowalla_file_zipped = download_file(GOWALLA_DATASET_URL, GOWALLA_GZIPPED, DIR)
    unzipped_gowalla_file = gunzip(gowalla_file_zipped)
    return unzipped_gowalla_file


def parse_line(line) -> Action:
    user_id, timestamp, lat, lon, item_id = line.split("\t")
    timestamp = time.mktime(dateutil.parser.isoparse(timestamp).timetuple())
    return Action(user_id, item_id, timestamp)


def get_gowalla_dataset(max_actions: Optional[int] = None) -> Iterator[Action]:
    dataset_file = prepare_data()
    n_action = 0
    for line in open(dataset_file):
        yield parse_line(line.strip())
        n_action += 1
        if max_actions is not None and n_action >= max_actions:
            break
