import unittest
from aprec.recommenders.kion_challenge_featurizer import KionChallengeFeaturizer
from aprec.datasets.mts_kion import get_users
from aprec.datasets.mts_kion import get_items
from aprec.datasets.mts_kion import get_mts_kion_dataset

class TestKionChallengeFeaturizer(unittest.TestCase):
    def test_kion_challenge_featurizer(self):
        featurizer = KionChallengeFeaturizer()
        for user in get_users():
            featurizer.add_user(user)
        for item in get_items():
            featurizer.add_item(item)
        for action in get_mts_kion_dataset(20000):
            featurizer.add_action(action)
        featurizer.build()
        candidates =  ['7638', '6686', '9506']
        features = featurizer.get_features('176549', candidates)
        self.assertEquals(len(features),len(candidates))
        for i in range(len(candidates)):
            self.assertEquals(len(features[i]), len(featurizer.feature_names))
        pass

if __name__ == "__main__":
    unittest.main()
