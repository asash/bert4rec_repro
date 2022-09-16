import json
import unittest

reference_actions =  [{'user_id': 430450, 'item_id': 91854, 'data': {}, 'timestamp': 1108524202}, 
                      {'user_id': 430450, 'item_id': 137692, 'data': {}, 'timestamp': 1108524579}, 
                      {'user_id': 430450, 'item_id': 105383, 'data': {}, 'timestamp': 1108526786},
                      {'user_id': 6662, 'item_id': 30082, 'data': {}, 'timestamp': 1109696237}, 
                      {'user_id': 6662, 'item_id': 105487, 'data': {}, 'timestamp': 1109696377},
                      {'user_id': 6662, 'item_id': 75311, 'data': {}, 'timestamp': 1109697913}, 
                      {'user_id': 6662, 'item_id': 124, 'data': {}, 'timestamp': 1109699235}, 
                      {'user_id': 6662, 'item_id': 15422, 'data': {}, 'timestamp': 1109699966},
                      {'user_id': 6662, 'item_id': 76445, 'data': {}, 'timestamp': 1109705615},
                       {'user_id': 6662, 'item_id': 76263, 'data': {}, 'timestamp': 1109705744}]

class TestYelpDataset(unittest.TestCase):
    def test_yelp_dataset(self):
        from aprec.datasets.yelp import get_yelp_dataset
        dataset = [json.loads(action.to_json()) for action in get_yelp_dataset(max_actions=10)]
        self.assertEqual(reference_actions, dataset)

if __name__ == "__main__":
    unittest.main()