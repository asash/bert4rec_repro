from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

class SvdRecommender(Recommender):
    def __init__(self, num_latent_components, random_seed=None):
        super().__init__()
        self.latent_components = num_latent_components
        self.users = ItemId()
        self.items = ItemId()
        self.rows = []
        self.cols = []
        self.vals = []
        self.model = None 
        self.user_vectors = None
        self.mean_user = None
        self.random_seed = random_seed
        self.biases = None
        
    def name(self):
        return "Svd@{}".format(self.latent_components)

    def add_action(self, action):
        row = self.users.get_id(action.user_id)
        col = self.items.get_id(action.item_id)
        self.rows.append(row)
        self.cols.append(col)
        self.vals.append(1.0)

    def rebuild_model(self):
        matrix_original = csr_matrix((self.vals, (self.rows, self.cols)))
        self.biases = np.asarray(np.mean(matrix_original, axis=0))[0]
        vals_unbiased = []
        for i in range(len(self.vals)):
            vals_unbiased.append(1.0  - self.biases[self.cols[i]])
        matrix = csr_matrix((vals_unbiased, (self.rows, self.cols)))
        self.model = TruncatedSVD(n_components=self.latent_components, random_state=self.random_seed)
        self.user_vectors = self.model.fit_transform(matrix)
        self.mean_user = np.mean(self.user_vectors, axis=0)

    def recommend(self, user_id, limit, features=None):
        scores = self.get_all_item_scores(user_id)
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            user_result = []
            scores = self.get_all_item_scores(request.user_id)
            for item_id in request.item_ids:
                if self.items.has_item(item_id):
                    user_result.append((item_id, scores[self.items.get_id(item_id)]))
                else:
                    user_result.append((item_id, float("-inf")))
            user_result.sort(key=lambda x: -x[1])
            result[request.user_id] = user_result
        return result


    def get_all_item_scores(self, user_id):
        user_vec = self.mean_user
        if self.users.has_item(user_id):
            user_vec = self.user_vectors[self.users.get_id(user_id)]
        scores = self.model.inverse_transform([user_vec])[0] + self.biases
        return scores


    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)
