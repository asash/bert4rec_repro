import random
import unittest

import numpy as np

from aprec.api.action import Action
from aprec.evaluation.metrics.sampled_proxy_metric import SampledProxy
from aprec.evaluation.metrics.precision import Precision


class TestSampledProxyMetric(unittest.TestCase):
    def test_proxy_precision(self):
        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [Action(user_id = 1, item_id = 1, timestamp=1),
                  Action(user_id = 1, item_id = 3, timestamp=2)]
        all_item_ids = [1, 2, 3, 4, 5, 6]
        random.seed(31337)
        np.random.seed(31337)
        metric = SampledProxy(all_item_ids, [1./6] * 6, 2, Precision(3))
        self.assertAlmostEqual(metric(recommended, actual), 2./3)

if __name__ == "__main__":
    unittest.main()

