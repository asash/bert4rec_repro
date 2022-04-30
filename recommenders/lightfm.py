from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from scipy.sparse import csr_matrix
from lightfm import LightFM
import numpy as np


class LightFMRecommender(Recommender):
    def __init__(self, num_latent_components, random_seed=None, loss='bpr', n_epochs=20, num_threads=4):
        super().__init__()
        self.latent_components = num_latent_components
        self.users = ItemId()
        self.items = ItemId()
        self.rows = []
        self.cols = []
        self.vals = []
        self.model = None
        self.num_threads = num_threads
        self.loss=loss
        self.n_epochs = n_epochs

    def name(self):
        return "Lightfm_{}@{}".format(self.loss, self.latent_components)

    def add_action(self, action):
        row = self.users.get_id(action.user_id)
        col = self.items.get_id(action.item_id)
        self.rows.append(row)
        self.cols.append(col)
        self.vals.append(1.0)

    def rebuild_model(self):
        matrix_original = csr_matrix((self.vals, (self.rows, self.cols)))
        self.model = LightFM(no_components=self.latent_components, loss=self.loss)
        self.model.fit(matrix_original, epochs=self.n_epochs, verbose=True, num_threads=self.num_threads)

    def recommend(self, user_id_external, limit, features=None):
        if not(self.users.has_item(user_id_external)):
            return [] #can't process unknown users
        user_id = self.users.get_id(user_id_external)
        items_ids = [i for i in range(self.items.size())]
        scores = self.model.predict(user_id, items_ids)
        best_ids = np.argpartition(scores, -limit)[-limit:]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        result.sort(key=lambda x: -x[1])
        return result


    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            user_result = []
            user_id = self.users.get_id(request.user_id)
            for item_id in request.item_ids:
                if self.items.has_item(item_id):
                    score = self.model.predict(user_id, [self.items.get_id(item_id)])
                    user_result.append((item_id, score))
                else:
                    user_result.append((item_id, float("-inf")))
            user_result.sort(key=lambda x: -x[1])
            result[request.user_id] = user_result
        return result

    def get_similar_items(self, item_id, limit):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)
