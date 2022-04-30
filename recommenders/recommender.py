from tqdm import tqdm
from aprec.api.item import Item

from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.api.user import User
from aprec.api.action import Action


class Recommender():
    def __init__(self):
        self.items_ranking_requests = []
        self.val_users = set() 

    def name(self):
        raise NotImplementedError

    def add_action(self, action: Action):
        raise (NotImplementedError)

    def rebuild_model(self):
        raise (NotImplementedError)

    def recommend(self, user_id, limit: int, features=None):
        raise (NotImplementedError)

    # recommendation request = tuple(user_id, features)
    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm(recommendation_requests, ascii=True):
            results.append(self.recommend(user_id, limit, features))
        return results

    # many recommenders don't require users, so leave it doing nothing by default
    def add_user(self, user: User):
        pass

    # many recommenders don't require items, so leave it doing nothing by default
    def add_item(self, item: Item):
        pass



    def recommend_by_items(self, items_list, limit: int):
        raise (NotImplementedError)

    def get_similar_items(self, item_id, limit: int):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)


    #the directory where the recommender can save stuff, like logs
    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def get_metadata(self):
        return {}

    def set_val_users(self, val_users):
        self.val_users = val_users

    # class to run sample-based evaluation.
    # according to https://dl.acm.org/doi/abs/10.1145/3383313.3412259 it is not always correct strategy,
    # However we want to keep it in order to be able to produce results comparable with other
    # Papers in order to be able to understand that we implemented our methods correctly.
    # for example comparison table  in BERT4rec is based on sampled metrics (https://arxiv.org/pdf/1904.06690.pdf, page7)
    # we decompose requests and results because some of third-party libraries
    # it is hard to perform items ranking outside of their evaluation process
    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.items_ranking_requests.append(request)

    # should use item ranking requests produced added by add_test_itmes_ranking_requests
    def get_item_rankings(self):
        raise NotImplementedError