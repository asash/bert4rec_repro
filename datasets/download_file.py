import logging
import os

import requests

from aprec.utils.os_utils import mkdir_p_local, get_dir
def download_file(url, filename, data_dir):
    mkdir_p_local(data_dir)
    full_filename = os.path.join(get_dir(), data_dir, filename)
    if not os.path.isfile(full_filename):
        logging.info(f"downloading  {filename} file")
        response = requests.get(url)
        with open(full_filename, 'wb') as out_file:
            out_file.write(response.content)
        logging.info(f"{filename} dataset downloaded")
        full_name = get_dir()
    else:
        logging.info(f"booking {filename} file already exists, skipping")
    return full_filename