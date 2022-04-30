import unittest

USER_ID = '120' 

REFERENCE_COLD_START = [('318', 0.6019900660660039), ('296', 0.5928136146373703), ('356', 0.5671645460239426),
                        ('593', 0.494680602882191), ('50', 0.46695169879496523), ('47', 0.46184204110408533),
                        ('527', 0.4398795906398074), ('260', 0.43692734916941883),
                        ('1', 0.4210339121252358), ('589', 0.4195799728444275)]

REFERENCE_USER_RECOMMENDATIONS = [('296', 0.5097028006608604),
                                  ('457', 0.46596785899698745),
                                  ('110', 0.46393997126655373),
                                  ('380', 0.4291430391625074),
                                  ('593', 0.4159414958428441),
                                  ('1', 0.398391005348504),
                                  ('1210', 0.35877141070731267),
                                  ('260', 0.35489876705579815),
                                  ('292', 0.34561595303551884),
                                  ('733', 0.34348521664244525)]
class TestSvdRecommender(unittest.TestCase):
    def compare_recommendations(self, rec1, rec2):
        self.assertEqual(len(rec1), len(rec2))
        for i in range(len(rec1)):
            self.assertEqual(rec1[i][0], rec2[i][0])
            self.assertAlmostEqual(rec1[i][1], rec2[i][1])
         
    def test_svd_recommender(self):
        from aprec.recommenders.svd import SvdRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        from aprec.api.action import Action

        svd_recommender = SvdRecommender(10, random_seed=31337)
        recommender = FilterSeenRecommender(svd_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        self.compare_recommendations(recommender.recommend(12341324, 10), REFERENCE_COLD_START)
        recs = recommender.recommend(USER_ID, 10)
        self.compare_recommendations(recs, REFERENCE_USER_RECOMMENDATIONS)

        actions =  [Action('1', 1, 1), 
                    Action('1', 2, 2),
                    Action('2', 2, 1),
                    Action('2', 3, 1)]
        recommender = SvdRecommender(2, random_seed=31337)
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()






if __name__ == "__main__":
    unittest.main()

