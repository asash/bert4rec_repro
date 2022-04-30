from aprec.api.action import Action
from aprec.datasets.download_file import download_file

BERT4REC_DATASET_URL="https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/{}.txt"
BERT4REC_DIR = "data/bert4rec"
VALID_DATASETS={"beauty", "ml-1m", "steam"}

def get_bert4rec_dataset(dataset):
    if dataset not in VALID_DATASETS:
        raise ValueError(f"unknown bert4rec dataset {dataset}")
    dataset_filename = download_file(BERT4REC_DATASET_URL.format(dataset), dataset + ".txt", BERT4REC_DIR)
    actions = []
    prev_user = None
    current_timestamp = 0
    with open(dataset_filename) as input:
        for line in input:
            user, item = [str(id) for id in line.strip().split()]
            if user != prev_user:
                current_timestamp = 0
            prev_user = user
            current_timestamp += 1
            actions.append(Action(user, item, current_timestamp))
    return actions