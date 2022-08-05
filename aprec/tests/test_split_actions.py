import unittest
from aprec.evaluation.split_actions import TemporalGlobal, RandomSplit
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.tests.generate_actions import generate_actions
from aprec.utils.generator_limit import generator_limit
from aprec.evaluation.evaluate_recommender import group_by_user


class TestSplitActions(unittest.TestCase):

    def test_split_actions(self):
        actions =  generate_actions(100)
        split_actions = TemporalGlobal((7, 1, 2))
        splitted = split_actions(actions)
        self.assertEqual(len(splitted), 3)
        self.assertEqual(len(splitted[0]), 70)
        self.assertEqual(len(splitted[1]), 10)
        self.assertEqual(len(splitted[2]), 20)
        assert(times_func(splitted[0], max) <= times_func(splitted[1], min))
        assert(times_func(splitted[1], max) <= times_func(splitted[2], min))
        self.assertEqual(set(actions), set(splitted[0] + splitted[1] + splitted[2]))

    def test_random_split(self):
        user_ids = set()
        actions = []
        for action in generator_limit(get_movielens20m_actions(), 10000):
            actions.append(action)
            user_ids.add(action.user_id)
        random_split = RandomSplit(0.5, 10)
        train, test = random_split(actions)
        train_users =  group_by_user(train)
        test_users = group_by_user(test)
        self.assertEqual(len(test_users), 10)
        for user in test_users:
            self.assertTrue(abs(len(test_users[user]) - len(train_users[user])) <= 1)
            test_items = set([action.item_id for action in test_users[user]])
            train_items = set([action.item_id for action in train_users[user]])
            self.assertEqual(len(train_items.intersection(test_items)), 0)



def times_func(actions, func):
    return func([action.timestamp for action in actions])
        
if __name__ == "__main__":
    unittest.main()
