
import unittest

USER_ID = '120' 

REFERENCE_COLD_START =  [('296', 62), ('318', 62), ('356', 60),
                         ('593', 48), ('260', 44), ('50', 43), ('527', 43), ('608', 42), ('47', 41), ('480', 40)]

REFERENCE_USER_RECOMMENDATIONS = [('276', 0.5), ('450', 0.5), ('296', 0.48612153038259565), ('292', 0.47265625),
                                  ('361', 0.4444444444444444), ('225', 0.4375),
                                  ('593', 0.436046511627907), ('474', 0.4166666666666667),
                                  ('1089', 0.38813151563753007), ('588', 0.3820662768031189)]

class TestItemItemRecommender(unittest.TestCase):
    def compare_recommendations(self, rec1, rec2):
        print(rec1, rec2)
        self.assertEqual(len(rec1), len(rec2))
        for i in range(len(rec1)):
            self.assertEqual(rec1[i][0], rec2[i][0])
            self.assertAlmostEqual(rec1[i][1], rec2[i][1])
         
    def test_item_item_recommender(self):
        from aprec.recommenders.item_item import ItemItemRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.utils.generator_limit import generator_limit
        from aprec.api.action import Action
        item_item_recommender = ItemItemRecommender()
        recommender = FilterSeenRecommender(item_item_recommender)
        catalog = get_movies_catalog()

        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs_cold_start = recommender.recommend(12341324, 10)
        self.compare_recommendations(recs_cold_start, REFERENCE_COLD_START)
        recs = recommender.recommend(USER_ID, 10)
        self.compare_recommendations(recs, REFERENCE_USER_RECOMMENDATIONS)

        actions =  [Action('1', 1, 1), 
                    Action('1', 2, 2),
                    Action('2', 2, 1),
                    Action('2', 3, 1)]
        recommender = ItemItemRecommender()
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()






if __name__ == "__main__":
    unittest.main()

