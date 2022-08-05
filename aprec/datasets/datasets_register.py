"""This file contains register of all available datasets in our system.
Unless necessary only use datasets from this file."""
from typing import Callable, Dict, List, Iterable

from aprec.api.action import Action
from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.datasets.dataset_utils import filter_cold_users
from aprec.datasets.gowalla import get_gowalla_dataset
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.datasets.movielens100k import get_movielens100k_actions


class DatasetsRegister(object):
    _all_datasets: Dict[str, Callable[[], Iterable[Action]]] = {
        "BERT4rec.ml-1m": lambda: get_bert4rec_dataset("ml-1m"),
        "BERT4rec.steam": lambda: get_bert4rec_dataset("steam"),
        "BERT4rec.beauty": lambda: get_bert4rec_dataset("beauty"),
        "ml-20m": lambda: get_movielens20m_actions(min_rating=0.0),
        "ml-100k": lambda: get_movielens100k_actions(min_rating=0.0),
        "gowalla": get_gowalla_dataset,
        "ml-20m_warm5": lambda: filter_cold_users(
            get_movielens20m_actions(min_rating=0.0), 5
        ),
        "gowalla_warm5": lambda: filter_cold_users(get_gowalla_dataset(), 5),
        "ml-20m_warm10": lambda: filter_cold_users(
            get_movielens20m_actions(min_rating=0.0), 10
        ),
        "gowalla_warm10": lambda: filter_cold_users(get_gowalla_dataset(), 10),
    }

    def __getitem__(self, item) -> Callable[[], Iterable[Action]]:
        if item not in DatasetsRegister._all_datasets:
            raise KeyError(f"The dataset {item} is not registered")
        return self._all_datasets[item]

    def all_datasets(self) -> List[str]:
        return list(DatasetsRegister._all_datasets.keys())
