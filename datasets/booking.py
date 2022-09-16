import pytz

from aprec.api.action import Action
import requests
import calendar
from aprec.utils.os_utils import mkdir_p_local, get_dir
from datetime import datetime, timezone
import logging
import os


BOOKING_DIR = "data/booking"
BOOKING_TRAIN_URL = "https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/train_set.csv"
BOOKING_TEST_URL = "https://raw.githubusercontent.com/bookingcom/ml-dataset-mdt/main/test_set.csv"
BOOKING_TRAIN_FILE = "booking_train_set.csv"
BOOKING_TEST_FILE = "booking_test_set.csv"

def split_csv_string(string):
    parts = []
    current_pos = 0
    state = 'unquoted'
    current_str = ""
    for i in range(len(string.strip())):
        if state == 'unquoted':
            if string[i] == '"':
                state = 'quoted'
            elif string[i] == ',':
                parts.append(current_str)
                current_str = ""
            else:
                current_str += string[i]
        else:
            if string[i] == '"':
                state = "unquoted"
            else:
                current_str += string[i]
    parts.append(current_str)
    return parts



def get_booking_dataset_one_file(filename, is_testset=False, max_actions=None, unix_timestamps=False, mark_control=True):
    actions = []
    submit_actions = []
    with open(filename) as input_file:
        input_file.readline()
        next_line = input_file.readline()
        while(len(next_line) != 0):
            parts = split_csv_string(next_line)
            try:
                if not is_testset:
                    user_id, checkin_str, checkout_str, city_id,\
                            device_class, affiliate_id,\
                            booker_country, hotel_country, utrip_id = parts
                else:
                    user_id, checkin_str, checkout_str, device_class, affiliate_id,booker_country,utrip_id,\
                                city_id,hotel_country = parts

            except Exception as ex:
                raise Exception(f"incorrect line: {next_line}")
            date_format = "%Y-%m-%d"
            checkin = datetime.strptime(checkin_str, date_format).replace(tzinfo=timezone.utc)
            checkout = datetime.strptime(checkout_str, date_format).replace(tzinfo=timezone.utc)
            if unix_timestamps:
                checkin = calendar.timegm(checkin.timetuple())
                checkout = calendar.timegm(checkout.timetuple())
            data = {'user_id': user_id,
                    'device_class': device_class,
                    'affiliate_id': affiliate_id,
                    'hotel_country': hotel_country,
                    'booker_country': booker_country,
                    'checkin_date': checkin,
                    'checkout_date': checkout,
                    }
            if mark_control:
                data['is_control'] = is_testset

            action = Action(user_id = utrip_id, item_id = city_id, timestamp=checkin, data = data)
            if action.item_id == "0":
                submit_actions.append(action)
            else:
                actions.append(action)
            if max_actions is not None and len(actions) >= max_actions:
                break
            next_line = input_file.readline()
    return actions, submit_actions

def get_booking_dataset_internal(train_filename, test_filename,
                                    max_actions_per_file=None, unix_timestamps=False, mark_control=True):

    train_actions, _ = get_booking_dataset_one_file(train_filename, is_testset=False,
                                                 max_actions=max_actions_per_file,
                                                 unix_timestamps=unix_timestamps, mark_control=mark_control)
    test_actions, submit_actions = get_booking_dataset_one_file(test_filename, is_testset=True,
                                                 max_actions=max_actions_per_file,
                                                 unix_timestamps=unix_timestamps, mark_control=mark_control)
    return train_actions + test_actions, submit_actions

def download_booking_file(url, filename):
    mkdir_p_local(BOOKING_DIR)
    full_filename = os.path.join(get_dir(), BOOKING_DIR, filename) 
    if not os.path.isfile(full_filename):
        logging.info(f"downloading booking {filename} file")
        response = requests.get(url)
        with open(full_filename, 'wb') as out_file:
            out_file.write(response.content)
        logging.info(f"{filename} dataset downloaded")
        full_name = get_dir()
    else:
        logging.info(f"booking {filename} file already exists, skipping")
    return full_filename 



def download_booking_train():
    return download_booking_file(BOOKING_TRAIN_URL, BOOKING_TRAIN_FILE)

def download_booking_test():
    return download_booking_file(BOOKING_TEST_URL, BOOKING_TEST_FILE)



def get_booking_dataset(max_actions_per_file=None, unix_timestamps=False, mark_control=True):
    train_filename = download_booking_train()
    test_filename = download_booking_test()
    return get_booking_dataset_internal(train_filename, test_filename, max_actions_per_file, unix_timestamps, mark_control)

