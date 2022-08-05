import numpy as np
from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.evaluation.evaluation_utils import group_by_user
from aprec.evaluation.metrics.sampled_proxy_metric import SampledProxy
from aprec.evaluation.samplers.sampler import TargetItemSampler

class PopTargetItemsSampler(TargetItemSampler):
    def get_sampled_ranking_requests(self):
        items, probs = SampledProxy.all_item_ids_probs(self.actions)
        by_user_test = group_by_user(self.test)
        result = []
        for user_id in by_user_test:
            target_items = set(action.item_id for action in by_user_test[user_id])
            while(len(target_items) < self.target_size):
                item_ids = np.random.choice(items,
                  self.target_size - len(target_items), p=probs, replace=False)
                for item_id in item_ids:
                    if item_id not in target_items:
                        target_items.add(item_id)
            result.append(ItemsRankingRequest(user_id=user_id, item_ids=list(target_items)))
        return result