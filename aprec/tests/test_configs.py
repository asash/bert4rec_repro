import unittest

class TestConfigs(unittest.TestCase):
    def test_configs(self):
        import os
        from aprec.utils.os_utils import get_dir, recursive_listdir
        configs_dir = os.path.join(os.path.join(get_dir()), "evaluation/configs")
        for filename in recursive_listdir(configs_dir):
            if self.should_ignore(filename):
                continue
            self.validate_config(filename)

    def should_ignore(self, filename):
        if not (filename.endswith(".py")):
            return True
        if "/common/" in filename:
            return True

        if "common_benchmark_config" in filename:
            return True 

        if "__pycache__" in filename:
            return True
        if "__init__" in filename:
            return True
        return False

    def validate_config(self, filename):
        from aprec.datasets.datasets_register import DatasetsRegister
        from aprec.evaluation.samplers.sampler import TargetItemSampler
        from aprec.evaluation.split_actions import ActionsSplitter
        from aprec.evaluation.metrics.metric import Metric
        from aprec.recommenders.recommender import Recommender
        import importlib.util
        import os
        import sys



        sys.stderr.write(f"validating {filename}... ")
        config_name = os.path.basename(filename[:-3])
        spec = importlib.util.spec_from_file_location(config_name, filename)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        required_fields = ["DATASET", "METRICS", "SPLIT_STRATEGY", "RECOMMENDERS", "USERS_FRACTIONS"]
        for field in required_fields:
            self.assertTrue(hasattr(config, field), f"missing required field {field}")

        self.assertTrue(config.DATASET in DatasetsRegister().all_datasets(), f"Unknown dataset {config.DATASET}")
        self.assertTrue(isinstance(config.SPLIT_STRATEGY, ActionsSplitter), f"Split strategy has wrong type: f{type(config.SPLIT_STRATEGY)}")
        self.assertTrue(len(config.METRICS) > 0)
        for metric in config.METRICS:
            self.assertTrue(isinstance(metric, Metric))

        for fraction in config.USERS_FRACTIONS:
            self.assertTrue(isinstance(fraction, (float, int)))
            self.assertGreater(fraction, 0)
            self.assertLessEqual(fraction, 1)


        if hasattr(config, "USERS"):
            self.assertTrue(callable(config.USERS), "USERS should be callable")

        if hasattr(config, "SAMPLED_METRICS_ON"):
            raise Exception("SAMPLED_METRICS_ON field is obsolete. Please use  TARGET_ITEMS_SAMPLER")
        if hasattr(config, "TARGET_ITEMS_SAMPLER"):
            self.assertTrue(isinstance(config.TARGET_ITEMS_SAMPLER, TargetItemSampler))

        model_cnt = 0 
        for recommender_name in config.RECOMMENDERS:
            recommender = config.RECOMMENDERS[recommender_name]()
            self.assertTrue(isinstance(recommender, Recommender), f"bad recommender type of {recommender_name}")
            del(recommender)
            model_cnt += 1
        sys.stderr.write(f"{model_cnt} models found\n")



if __name__ == "__main__":
    unittest.main()

