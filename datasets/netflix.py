import calendar
import datetime
import os

from tqdm import tqdm
from aprec.datasets.download_file import download_file
from aprec.utils.os_utils import get_dir, shell
from aprec.api.action import Action


#DATASET_KEY = "MAGNET"
#DATASET_LINK = b'ICEiIyQld389OXwkPy5/LzU4JXp8L3BifHh2KSJlKXZ1e3ZnL3dyfiJoKHImKXc1KHUgeHRlfyMgKHgwazQ3cCklOTA2aHIQaHIDaHMXLCMkKSQ8JCMxIjMjKC4xPm8yIi1gfwcwIy4qOC8yKG41JTF3OTJ4OCUhaHMEaHMXaHIDOTMwLisgP28yIjA1KDMiODIjKDN/OStgfgBndHZ8azUjcDUhPWRiDGV3C2RjCzQ3LCI6KDJrIjE0IzQ3LCI6P24qPyZ0fgF0fnJmaHIDLC8/IjUrLiQ='

#def decode_link(input, key):
#    res = b'' 
#    pos = 0
#    for char in base64.b64decode(input):
#        res_chr = bytes([char ^ ord(DATASET_KEY[pos]) ^ pos])
#        res += res_chr 
#        pos  = (pos + 1) % len(key)
#    return res.decode()

DATASET_LINK = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"
DATASET_FILE = "nf_prize_dataset.tar.gz"
DIR="data/netflix"

def download_netflix_dataset():
    print("downloading netflix dataset...")
    return download_file(DATASET_LINK, DATASET_FILE, DIR)

def extract_netflix_data():
    local_dir_name = os.path.join(get_dir(), DIR) 
    local_file_name = os.path.join(local_dir_name, DATASET_FILE)
    output_dir = os.path.join(local_dir_name, "download")
    training_set_archive = os.path.join(output_dir, "training_set.tar") 
    if os.path.isdir(output_dir):
        print("dataset is already extracted")
    else:
        shell(f"tar -xzvf {local_file_name} -C {local_dir_name}")
        shell(f"tar -xvf {training_set_archive} -C {output_dir}")
    netflix_files_dir = os.path.join(output_dir, "training_set")
    return netflix_files_dir 

def prepare_data():
    download_netflix_dataset()
    return extract_netflix_data()

def parse_netflix_actions(full_filename):
    result = []
    date_format = "%Y-%m-%d"
    with open(full_filename, "r") as input:
        movie_id = input.readline().strip().strip(":")
        for line in input:
            user_id, rating, date_str = line.strip().split(',')
            parsed_time = datetime.datetime.strptime(date_str, date_format).replace(tzinfo=datetime.timezone.utc)
            timestamp = calendar.timegm(parsed_time.timetuple())
            result.append(Action(user_id, movie_id, date_str, {"rating": rating}))
    return result



def get_netflix_dataset():
    dataset_dir = prepare_data()
    actions = []
    for file in tqdm(os.listdir(dataset_dir), ascii=True):
        if file.startswith("mv_"):
            full_filename = os.path.join(dataset_dir, file)
            actions += parse_netflix_actions(full_filename)
    return actions