
from unittest import TestCase
import unittest


class TestBeautyDataset(TestCase):
    def test_beauty_dataset(self):
        from aprec.datasets.dataset_stats import dataset_stats
        from aprec.datasets.beauty import get_beauty_dataset
        from aprec.datasets.dataset_utils import filter_cold_users

        dataset = filter_cold_users(get_beauty_dataset(), 5)
        result = dataset_stats(dataset, metrics=['num_users', 'num_items', 'num_interactions'])
        print(result)

if __name__ == "__main__":
    unittest.main()