import unittest


class TestRecommenderEvaluator(unittest.TestCase):
    def test_recommender_evaluator(self):
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
        from aprec.utils.generator_limit import generator_limit
        from aprec.evaluation.split_actions import LeaveOneOut
        from aprec.evaluation.metrics.precision import Precision
        from aprec.recommenders.top_recommender import TopRecommender
        from aprec.evaluation.evaluate_recommender import RecommendersEvaluator
        import tempfile


        actions = [action for action in generator_limit(get_movielens20m_actions(), 100000)]
        recommenders= {"top_recommender": TopRecommender}

        data_splitter = LeaveOneOut(max_test_users=128)
        metrics = [Precision(5)]
        out_dir = tempfile.mkdtemp()
        n_val_users=10
        recommendations_limit = 10
        target_items_sampler = PopTargetItemsSampler(20)
        evaluator = RecommendersEvaluator(actions, recommenders, metrics,
                                          out_dir, data_splitter, n_val_users,
                                          recommendations_limit, 
                                          target_items_sampler=target_items_sampler)
        result = evaluator()['recommenders']['top_recommender']
        del(result["model_build_time"])
        del(result["model_inference_time"])
        self.assertEqual(result, 
                {'precision@5': 0.0078125, 'sampled_metrics': {'precision@5': 0.039062500000000014},
                'model_metadata': {"top 20 items": [("318", 556), ("296", 523), ("356", 501), ("593", 493),
                ("260", 425), ("50", 410), ("527", 407), ("2571", 403),
                ("110", 372), ("1196", 356), ("457", 355), ("1198", 355), 
                ("2858", 349), ("589", 341), ("608", 339), ("1210", 338),
                ("1", 334), ("858", 334), ("47", 324), ("2959", 321)]}})
        
if __name__ == "__main__":
    unittest.main()
