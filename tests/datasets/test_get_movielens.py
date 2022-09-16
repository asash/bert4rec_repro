import unittest

REFERENCE_LINES_MOVIELENS_20M =\
"""Action(uid=1, item=151, ts=1094785734, data={'rating': 4.0})
Action(uid=1, item=223, ts=1112485573, data={'rating': 4.0})
Action(uid=1, item=253, ts=1112484940, data={'rating': 4.0})
Action(uid=1, item=260, ts=1112484826, data={'rating': 4.0})
Action(uid=1, item=293, ts=1112484703, data={'rating': 4.0})
Action(uid=1, item=296, ts=1112484767, data={'rating': 4.0})
Action(uid=1, item=318, ts=1112484798, data={'rating': 4.0})
Action(uid=1, item=541, ts=1112484603, data={'rating': 4.0})
Action(uid=1, item=1036, ts=1112485480, data={'rating': 4.0})
Action(uid=1, item=1079, ts=1094785665, data={'rating': 4.0})
"""

REFERENCE_LINES_MOVIELENS_25M =\
"""Action(uid=1, item=296, ts=1147880044, data={'rating': 5.0})
Action(uid=1, item=307, ts=1147868828, data={'rating': 5.0})
Action(uid=1, item=665, ts=1147878820, data={'rating': 5.0})
Action(uid=1, item=1088, ts=1147868495, data={'rating': 4.0})
Action(uid=1, item=1237, ts=1147868839, data={'rating': 5.0})
Action(uid=1, item=1250, ts=1147868414, data={'rating': 4.0})
Action(uid=1, item=1653, ts=1147868097, data={'rating': 4.0})
Action(uid=1, item=2351, ts=1147877957, data={'rating': 4.5})
Action(uid=1, item=2573, ts=1147878923, data={'rating': 4.0})
Action(uid=1, item=2632, ts=1147878248, data={'rating': 5.0})
"""



REFERENCE_LINES_MOVIELENS_100K="""Action(uid=196, item=242, ts=881250949, data={'rating': 3.0})
Action(uid=186, item=302, ts=891717742, data={'rating': 3.0})
Action(uid=22, item=377, ts=878887116, data={'rating': 1.0})
Action(uid=244, item=51, ts=880606923, data={'rating': 2.0})
Action(uid=166, item=346, ts=886397596, data={'rating': 1.0})
Action(uid=298, item=474, ts=884182806, data={'rating': 4.0})
Action(uid=115, item=265, ts=881171488, data={'rating': 2.0})
Action(uid=253, item=465, ts=891628467, data={'rating': 5.0})
Action(uid=305, item=451, ts=886324817, data={'rating': 3.0})
Action(uid=6, item=86, ts=883603013, data={'rating': 3.0})
"""

class TestMovielensActions(unittest.TestCase):
    def test_get_actions20m(self):
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        lines = ""
        for action in generator_limit(get_movielens20m_actions(), 10):
            lines += action.to_str() + "\n" 
        self.assertEqual(lines, REFERENCE_LINES_MOVIELENS_20M)


    def test_get_actions_25m(self):
        from aprec.datasets.movielens25m import get_movielens25m_actions
        from aprec.utils.generator_limit import generator_limit
        lines = ""
        for action in generator_limit(get_movielens25m_actions(), 10):
            lines += action.to_str() + "\n" 
        print(lines)
        self.assertEqual(lines, REFERENCE_LINES_MOVIELENS_25M)

    def test_get_actions_100k(self):
        from aprec.datasets.movielens100k import get_movielens100k_actions
        from aprec.utils.generator_limit import generator_limit
        from collections import Counter


        lines = ""
        for action in generator_limit(get_movielens100k_actions(min_rating=1), 10):
            lines += action.to_str() + "\n"
        self.assertEqual(lines, REFERENCE_LINES_MOVIELENS_100K)
        all_actions = []
        user_cnt = Counter()
        item_cnt = Counter()
        for action in get_movielens100k_actions(min_rating=1):
            all_actions.append(action)
            user_cnt[action.user_id] += 1
            item_cnt[action.item_id] += 1
        self.assertEqual(len(all_actions), 100000)
        self.assertEqual(len(user_cnt), 943)
        self.assertEqual(len(item_cnt), 1682)



    def test_get_catalog(self):
        from aprec.datasets.movielens20m import get_movies_catalog
        catalog = get_movies_catalog()
        movie = catalog.get_item("2571")
        self.assertEqual(movie.title, "Matrix, The (1999)")


if __name__ == "__main__":
    unittest.main()
