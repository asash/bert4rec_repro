import random
from collections import defaultdict

import numpy as np
from keras.layers import Flatten
from keras.utils.data_utils import Sequence
from scipy.sparse import csr_matrix

from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
from aprec.losses.get_loss import get_loss


class MatrixFactorizationRecommender(Recommender):
    def __init__(self, embedding_size, num_epochs, loss, batch_size, regularization=0.0, learning_rate=0.001):
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(list)
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.sigma = 1.0
        self.max_positives=40
        self.regularization=regularization
        self.learning_rate=learning_rate

    def add_action(self, action):
        self.user_actions[self.users.get_id(action.user_id)].append(self.items.get_id(action.item_id))


    def rebuild_model(self):
        loss = get_loss(self.loss, self.items.size(), self.batch_size, self.max_positives)

        self.model = Sequential()
        self.model.add(Embedding(self.users.size(), self.embedding_size+1, input_length=1, embeddings_regularizer=l2(self.regularization)))
        self.model.add(Flatten())
        self.model.add(Dense(self.items.size(), kernel_regularizer=l2(self.regularization), bias_regularizer=l2(self.regularization)))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=loss)
        data_generator = DataGenerator(self.user_actions, self.users.size(), self.items.size(), self.batch_size)
        for epoch in range(self.num_epochs):
            print(f"epoch: {epoch}")
            data_generator.shuffle()
            self.model.fit(data_generator)

    def recommend(self, user_id, limit, features=None):
       with tf.device('/cpu:0'):
            model_input = np.array([[self.users.get_id(user_id)]])
            predictions = tf.nn.top_k(self.model.predict(model_input), limit)
            result = []
            for item_id, score in zip(predictions.indices[0], predictions.values[0]):
                result.append((self.items.reverse_id(int(item_id)), float(score)))
            return result

    def recommend_batch(self, recommendation_requests, limit):
        model_input = np.array([[self.users.get_id(request[0])] for request in recommendation_requests])
        predictions = tf.nn.top_k(self.model.predict(model_input), limit)
        result = []
        for idx in range(len(recommendation_requests)):
            request_result = []
            for item_id, score in zip(predictions.indices[idx][0], predictions.values[idx][0]):
                request_result.append((self.items.reverse_id(int(item_id)), float(score)))
            result.append(request_result)
        return result


class DataGenerator(Sequence):
    def __init__(self, user_actions, n_users, n_items, batch_size):
        rows = []
        cols = []
        vals = []

        for user in user_actions:
            for item in user_actions[user]:
                rows.append(user)
                cols.append(item)
                vals.append(1.0)
        self.full_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        self.users = list(range(n_users))
        self.batch_size = batch_size



    def shuffle(self):
        random.shuffle(self.users)

    def __len__(self):
        return len(self.users) // self.batch_size

    def __getitem__(self, item):
        start = self.batch_size * item
        end = self.batch_size * (item+1)
        users = []
        targets = []
        for i in range(start, end):
            users.append([self.users[i]])
            targets.append(self.full_matrix[self.users[i]].todense())
        users = np.array(users)
        targets = np.reshape(np.array(targets), (self.batch_size, self.full_matrix.shape[1]))
        return users, targets

