import datetime
from datetime import timezone
import calendar
import pandas as pd


from aprec.api.user import User
from aprec.api.item import Item
from aprec.api.action import Action
from aprec.datasets.download_file import download_file

MTS_KION_DIR = "data/mts_kion"

MTS_KION_INTERACTIONS_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/04adaecc/interactions.csv"
MTS_KION_INTERACTIONS_FILE = "interactions.csv"

MTS_KION_ITEMS_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/f90231b6/items.csv"
MTS_KION_ITEMS_FILE = "items.csv"

MTS_KION_USERS_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/6503d6ab/users.csv"
MTS_KION_USERS_FILE = "users.csv"


MTS_KION_SAMPLE_SUBMISSION_URL = "https://storage.yandexcloud.net/datasouls-ods/materials/faa61a41/sample_submission.csv"
MTS_KION_SAMPLE_SUBMISSION_FILE = "sample_submission.txt"


def get_actions(actions_file, max_actions=None):
    actions = []
    with open(actions_file) as input:
        header = input.readline()
        cnt = 0
        for line in input:
            user_id, item_id, last_watch_date, total_dur, watched_pct = line.split(",")
            ts = datetime.datetime.strptime(last_watch_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            timestamp = calendar.timegm(ts.timetuple()) 
            cnt += 1
            actions.append(Action(user_id=user_id, item_id=item_id, timestamp=timestamp))
            if max_actions is not None and cnt >= max_actions:
                break
        return actions


def get_submission_user_ids():
    sample_submission_file = download_file(MTS_KION_SAMPLE_SUBMISSION_URL,
                                           MTS_KION_SAMPLE_SUBMISSION_FILE, MTS_KION_DIR)
    result = []
    with open(sample_submission_file) as input:
        input.readline()
        line = input.readline()
        while(len(line) > 0):
            user_id = line.split(",")[0]
            result.append(user_id)
            line = input.readline()
    return result


def get_users():
    users_file = download_file(MTS_KION_USERS_URL, MTS_KION_USERS_FILE, MTS_KION_DIR)
    result = []
    with open(users_file) as input:
        input.readline()
        line = input.readline()
        while (len(line) > 0):
            user_id,age,income,sex,kids_flg = line.strip().split(",")
            cat_features = {"income": income, "sex": sex, "age": age, "kids_flg": kids_flg}
            result.append(User(user_id, cat_features=cat_features))
            line = input.readline()
    return result

def get_items():
    items_file = download_file(MTS_KION_ITEMS_URL, MTS_KION_ITEMS_FILE, MTS_KION_DIR)
    data = pd.read_csv(items_file) 
    items = []
    for _, item in data.iterrows():
        item_cat_features = []
        item_cat_features.append(('content_type', item['content_type']))
        item_cat_features.append(('age_rating', str(item['age_rating'])))
        tags = []
        for genre in item['genres'].split(","):
            item_cat_features.append(('genre', genre.lower()))
            tags.append(genre)
        for country in str(item['countries']).split(","):
            item_cat_features.append(('country', country.lower()))
        for studio in str(item['studios']).split(","):
            item_cat_features.append(('studio', studio.lower()))
        for director in str(item['directors']).split(","):
            item_cat_features.append(('director', director.lower()))
        for actor in str(item['actors']).split(","):
            item_cat_features.append(('actor', actor.lower()))
        for keyword in str(item['keywords']).split(","):
            item_cat_features.append(('keyword', keyword.lower()))
            tags.append(keyword)
        item_cat_features.append(('decade', item['release_year'] // 10))
        item_cat_features.append(('fiveyear', (item['release_year'] // 5) * 5))
        items.append(Item(item_id=item['item_id'], cat_features=item_cat_features)
                      .with_title(item['title'])
                      .with_tags(tags))
    return items
def get_mts_kion_dataset(max_actions=None):
    interactions_file = download_file(MTS_KION_INTERACTIONS_URL, MTS_KION_INTERACTIONS_FILE, MTS_KION_DIR)
    actions = get_actions(interactions_file, max_actions)
    return actions

