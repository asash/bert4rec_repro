import unittest 

class TestFilterColdUsers(unittest.TestCase):
    def test_filter_cold_users(self):
        from aprec.api.action import  Action
        from aprec.datasets.dataset_utils import filter_cold_users
        actions = [Action(item_id=1, user_id=1, timestamp=1), 
                   Action(item_id=2, user_id=1, timestamp=2), 
                   Action(item_id=1, user_id=2, timestamp=1)]
        result = list(filter_cold_users(actions, 2))
        self.assertEquals(str(result), "[Action(uid=1, item=1, ts=1), Action(uid=1, item=2, ts=2)]") 

if __name__ == "__main__":
    unittest.main()