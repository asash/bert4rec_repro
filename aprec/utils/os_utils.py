import hashlib
import logging
import os
import shlex
import subprocess
from typing import List


def get_dir() -> str:
    utils_dirname = os.path.dirname(os.path.abspath(__file__))
    lib_dirname = os.path.abspath(os.path.join(utils_dirname, ".."))
    return lib_dirname


def recursive_listdir(dir_name: str) -> List[str]:
    result = []
    for name in os.listdir(dir_name):
        full_name = os.path.join(dir_name, name)
        if os.path.isdir(full_name):
            result += recursive_listdir(full_name)
        else:
            result.append(full_name)
    return result


def shell(cmd: str) -> None:
    logging.info("running shell command: \n {}".format(cmd))
    subprocess.check_call(shlex.split(cmd))


def mkdir_p(dir_path: str) -> None:
    shell("mkdir -p {}".format(dir_path))


def mkdir_p_local(relative_dir_path: str) -> None:
    """create folder inside of library if does not exists"""
    local_dir = get_dir()
    abspath = os.path.join(local_dir, relative_dir_path)
    mkdir_p(abspath)


def file_md5(fname: str) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def console_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )
