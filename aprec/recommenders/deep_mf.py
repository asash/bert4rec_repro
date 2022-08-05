import keras
import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.keras.optimizers import Adam
from tqdm import trange

from aprec.recommenders.recommender import Recommender
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from aprec.utils.item_id import ItemId


#https://dl.acm.org/doi/abs/10.5555/3172077.3172336
class DeepMFRecommender(Recommender):
    def name(self):
       return "DeepMF"

    def __init__(self, num_users_per_sample, num_items_per_sample,
                 loss=BinaryCrossentropy(from_logits=True),
                 optimizer=Adam(),
                 user_hidden_layers = (256, 256, 256),
                 item_hidden_layers=(256, 256, 256), latent_dim_size=256,
                 activation = 'relu',
                 items_per_pred_iter = 1000,
                 steps = 100000
                 ):
        self.users = ItemId()
        self.items = ItemId()
        self.optimizer=optimizer
        self.num_users_per_sample = num_users_per_sample
        self.num_items_per_sample = num_items_per_sample
        self.user_hidden_layers = user_hidden_layers
        self.steps = steps
        self.item_hidden_layers = item_hidden_layers
        self.latent_dim_size = latent_dim_size
        self.activation = activation
        self.items_per_pred_iter = items_per_pred_iter
        self.loss = loss
        self.rows = []
        self.cols = []
        self.vals = []

    def add_action(self, action):
        row = self.users.get_id(action.user_id)
        col = self.items.get_id(action.item_id)
        self.rows.append(row)
        self.cols.append(col)
        self.vals.append(1.0)

    def rebuild_model(self):
        self.user_item_matrix = csr_matrix((self.vals, (self.rows, self.cols)))
        self.item_user_matrix = csr_matrix((self.vals, (self.cols, self.rows)))
        item_cnt = np.asarray(np.sum(self.item_user_matrix, axis = 1)).reshape(self.items.size())
        full_sum = np.sum(item_cnt)
        item_prob = item_cnt/full_sum
        
        self.model, self.pred_model = self.get_model()

        all_users = np.array(range(self.users.size()))
        all_items = np.array(range(self.items.size()))
        pbar = trange(self.steps, ascii=True)
        for step in pbar:
            users = np.random.choice(all_users, self.num_users_per_sample)
            items = np.random.choice(all_items, self.num_items_per_sample, p=item_prob)
            users_input = np.asarray(self.user_item_matrix[users]\
                                     .todense()).reshape(1, self.num_users_per_sample, self.items.size())
            items_input = np.asarray(self.item_user_matrix[items]\
                                     .todense()).reshape(1, self.num_items_per_sample, self.users.size())
            target = tf.gather(users_input[0], items, axis=1)

            target = tf.reshape(target, (1, self.num_users_per_sample, self.num_items_per_sample))
            history = self.model.fit([users_input, items_input], target, epochs=1, verbose=0)
            loss_val = history.history['loss'][0]
            pbar.set_description(f"loss: {loss_val:.06f}")

    def get_model(self):

        user_input, pred_user_input, user_repr, pred_user_repr  =\
            self.get_tower(self.items.size(), self.num_users_per_sample, self.user_hidden_layers, 1)

        item_input, pred_item_input, items_repr, pred_item_repr = \
            self.get_tower(self.users.size(), self.num_items_per_sample,
                           self.item_hidden_layers, self.items_per_pred_iter)

        items_repr = tf.transpose(items_repr, perm=[0, 2, 1])
        pred_item_repr = tf.transpose(pred_item_repr, perm=[0, 2, 1])

        output = tf.matmul(user_repr, items_repr)
        pred_output = tf.matmul(pred_user_repr, pred_item_repr)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(self.optimizer, self.loss)

        pred_model = Model(inputs=[pred_user_input, pred_item_input], outputs=pred_output)

        return model, pred_model

    def get_tower(self, input_size, num_samples, hidden_layers, pred_size):
        input = Input(shape=(num_samples, input_size), batch_size=1)
        pred_input = Input(shape=(pred_size, input_size), batch_size=1)

        input_proj = Dense(hidden_layers[0], activation=self.activation)
        repr = input_proj(input)
        pred_repr = input_proj(pred_input)

        for hidden_layer_size in hidden_layers[1:]:
            hidden_layer = Dense(hidden_layer_size, activation=self.activation)
            repr = hidden_layer(repr)
            pred_repr = hidden_layer(pred_repr)

        output_layer = Dense(self.latent_dim_size, activation='linear')
        repr = output_layer(repr)
        pred_repr = output_layer(pred_repr)
        return input, pred_input, repr, pred_repr

    def recommend(self, user_id, limit, features=None):
        users_input = np.asarray(self.user_item_matrix[self.users.get_id(user_id)] \
                                 .todense()).reshape(1, 1, self.items.size())
        num_items_left = self.items.size()
        predictions = []
        while(num_items_left > 0):
            start_item = self.items.size() - num_items_left
            end_item = start_item + self.items_per_pred_iter
            item_inputs = np.asarray(self.item_user_matrix[start_item:end_item].todense()
                       ).reshape(1, min(self.items_per_pred_iter, num_items_left), self.users.size())
            pad_samples = 0 if num_items_left > self.items_per_pred_iter else self.items_per_pred_iter - num_items_left
            pads = [(0, 0), (0, pad_samples), (0, 0)]
            item_inputs = np.pad(item_inputs,pads)
            pred = self.pred_model.predict([users_input, item_inputs])[0][0][:num_items_left]
            predictions.append(pred)
            num_items_left -= self.items_per_pred_iter
        all_predictions = np.concatenate(predictions)
        rec_idx = np.argpartition(all_predictions, -limit)[-limit:]
        scores = all_predictions[rec_idx]
        result = []
        for (idx, score) in zip(rec_idx, scores):
            result.append((self.items.reverse_id(idx), score))
        result.sort(key=lambda x: -x[1])
        return result
