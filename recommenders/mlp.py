from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import Sequence
import numpy as np
import math


class GreedyMLP(Recommender):
    def __init__(self,  bottleneck_size=32, train_epochs=300):
        self.users = ItemId()
        self.items = ItemId()
        self.rows = []
        self.cols = []
        self.vals = []
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.bottleneck_size = bottleneck_size
        self.train_epochs = train_epochs

    def name(self):
        return "GreedyMLP"

    def add_action(self, action):
        row = self.users.get_id(action.user_id)
        col = self.items.get_id(action.item_id)
        self.rows.append(row)
        self.cols.append(col)
        self.vals.append(1.0)

    def rebuild_model(self):
        self.matrix = csr_matrix((self.vals, (self.rows, self.cols)))
        self.model = self.get_model(self.matrix.shape[1])
        train_data, val_data = train_test_split(self.matrix)
        generator = BatchGenerator(train_data)
        val_generator = BatchGenerator(val_data)
        self.model.fit(generator, epochs=self.train_epochs, 
                            validation_data=val_generator)

    def get_model(self, n_movies):
        model = Sequential(name='MLP')
        model.add(layers.Input(shape=(n_movies), name="input"))
        model.add(layers.Dropout(0.5, name="input_drouput"))
        model.add(layers.Dense(256, name="dense1", activation="relu"))
        model.add(layers.Dense(128, name="dense2", activation="relu"))
        model.add(layers.Dense(self.bottleneck_size,
                name="bottleneck", activation="relu"))
        model.add(layers.Dense(128, name="dense3", activation="relu"))
        model.add(layers.Dense(256, name="dense4", activation="relu"))
        model.add(layers.Dropout(0.5, name="dropout"))
        model.add(layers.Dense(n_movies, name="output", activation="sigmoid"))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def recommend(self, user_id, limit, features=None):
        if self.users.has_item(user_id):
            user_vec = self.matrix[self.users.get_id(user_id)].todense()
        scores = self.model.predict(user_vec)[0]
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_similar_items(self, item_id, limit):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)


class BatchGenerator(Sequence):
    def __init__(self, matrix, batch_size = 1000):
        self.matrix = matrix
        self.batch_size = batch_size
        self.current_position = 0
        self.max = self.__len__()

    def __len__(self):
        return math.ceil(self.matrix.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch = self.matrix[idx * self.batch_size:(idx + 1) * self.batch_size].todense()
        return batch, batch

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result
