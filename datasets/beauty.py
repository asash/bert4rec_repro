from aprec.api.action import Action
from aprec.datasets.download_file import download_file

dataset_url="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv"
dataset = "ratings.csv"
dir = "data/beauty"

def get_beauty_dataset():
    dataset_filename = download_file(dataset_url, dataset, dir)
    actions = []
    with open(dataset_filename) as input:
        for line in input:
            user, item, rating, timestamp = line.strip().split(",")
            timestamp = int(timestamp)
            actions.append(Action(user, item, timestamp))
    return actions