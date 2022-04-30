import unittest

class TestBert4recDatasets(unittest.TestCase):
    def test_bert4rec_dataset(self):
        import json
        from aprec.datasets.dataset_stats import dataset_stats
        from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset

        for dataset_name in ["beauty", "steam", "ml-1m"]:
            print(f"analyzing dataset {dataset_name}")
            dataset = get_bert4rec_dataset(dataset_name)
            stats = dataset_stats(dataset, metrics=['num_users', 'num_items', 'num_interactions'])
            print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    unittest.main()