import unittest

class TestMtsKionDataset(unittest.TestCase):
    def test_get_mts_kion(self):
        import os
        import json
        from aprec.datasets.mts_kion import get_mts_kion_dataset

        local_path = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(local_path, "mts_kion_reference_actions.json")) as reference_file:
            reference_data = json.load(reference_file)
        data = [json.loads(action.to_json()) for action in get_mts_kion_dataset(max_actions=10)]
        self.assertEqual(reference_data, data)


    def test_get_submission_user_ids(self):
        from aprec.datasets.mts_kion import get_submission_user_ids
        submission_users = get_submission_user_ids()
        self.assertEqual(submission_users[:10],
                         ['3', '11', '29', '30', '33', '39', '46', '47', '51', '61'])

    def test_get_users(self):
        from aprec.datasets.mts_kion import get_users

        users = get_users()
        pass

    def test_get_items(self):
        from aprec.datasets.mts_kion import get_items

        items = get_items()
        self.assertEquals(items[0].cat_features[:2], [('content_type', 'film'), ('age_rating', '16.0')])
        self.assertEquals(len(items), 15963)




if __name__ == '__main__':
    unittest.main()
