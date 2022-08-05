import unittest

reference_actions = [{'user_id': '0', 'item_id': '22847', 'data': {}, 'timestamp': 1287532527.0},
                     {'user_id': '0', 'item_id': '420315', 'data': {}, 'timestamp': 1287440263.0}, 
                     {'user_id': '0', 'item_id': '316637', 'data': {}, 'timestamp': 1287358923.0}, 
                     {'user_id': '0', 'item_id': '16516', 'data': {}, 'timestamp': 1287343565.0}, 
                     {'user_id': '0', 'item_id': '5535878', 'data': {}, 'timestamp': 1287255042.0}]

class TestGowallaDataset(unittest.TestCase):
    def test_gowalla_dataset(self):
        import json
        from aprec.datasets.gowalla import get_gowalla_dataset 
        actions = [json.loads(action.to_json()) for action in get_gowalla_dataset(5)]
        self.assertEquals(actions, reference_actions)