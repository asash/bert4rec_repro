import gc
import random
import time

import tensorflow.keras.backend as K
from collections import defaultdict
import tensorflow as tf

from tqdm import tqdm
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.history_vectorizer import HistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.random_fraction_splitter import RandomFractionSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter

from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.dnn_sequential_recommender.data_generator.data_generator import DataGenerator
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from aprec.losses.loss import Loss
from aprec.losses.bce import BCELoss

from tensorflow.keras.optimizers import Adam

import numpy as np


class DNNSequentialRecommender(Recommender):
    def __init__(self, model_arch: SequentialRecsysModel, loss: Loss = BCELoss(),
                 users_featurizer=None,
                 items_featurizer=None,
                 train_epochs=300, optimizer=Adam(),
                 sequence_splitter:TargetSplitter = RandomFractionSplitter, 
                 val_sequence_splitter:TargetSplitter = SequenceContinuation,
                 targets_builder = FullMatrixTargetsBuilder,
                 batch_size=1000, early_stop_epochs=100, target_decay=1.0,
                 training_time_limit=None, debug=False,
                 metric = KerasNDCG(40), 
                 train_history_vectorizer:HistoryVectorizer = DefaultHistoryVectrizer(), 
                 pred_history_vectorizer:HistoryVectorizer = DefaultHistoryVectrizer(),
                 ):
        super().__init__()
        self.model_arch = model_arch
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.train_epochs = train_epochs
        self.loss = loss
        self.early_stop_epochs = early_stop_epochs
        self.optimizer = optimizer
        self.metadata = {}
        self.batch_size = batch_size
        self.target_decay = target_decay
        self.val_users = None
        self.training_time_limit = training_time_limit
        self.users_featurizer = users_featurizer
        self.items_featurizer = items_featurizer
        self.user_features = {}
        self.item_features = {}
        self.users_with_actions = set()
        self.max_user_features = 0
        self.sequence_splitter = sequence_splitter
        self.max_user_feature_val = 0
        self.targets_builder = targets_builder
        self.debug = debug
        self.metric = metric 
        self.val_sequence_splitter = val_sequence_splitter
        self.train_history_vectorizer = train_history_vectorizer
        self.pred_history_vectorizer = pred_history_vectorizer

    def add_user(self, user):
        if self.users_featurizer is None:
            pass
        else:
            self.user_features[self.users.get_id(user.user_id)] = self.users_featurizer(user)

    def add_item(self, item):
        if self.items_featurizer is None:
            pass
        else:
            self.item_features[self.items.get_id(item.item_id)] = self.items_featurizer(item)


    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return self.model

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.users_with_actions.add(user_id_internal)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    # exclude last action for val_users
    def user_actions_by_id_list(self, id_list, val_user_ids=None):
        val_users = set()
        if val_user_ids is not None:
            val_users = set(val_user_ids)
        result = []
        for user_id in id_list:
            if user_id not in val_users:
                result.append(self.user_actions[user_id])
            else:
                result.append(self.user_actions[user_id][:-1])
        return result

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()

    def rebuild_model(self):
        self.sort_actions()
        self.pass_parameters()
        self.max_user_features, self.max_user_feature_val = self.get_max_user_features()
        train_users, train_user_ids, train_features, val_users, val_user_ids, val_features = self.train_val_split()

        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = DataGenerator(val_users, val_user_ids, val_features, self.model_arch.max_history_length,
                                      self.items.size(),
                                      self.train_history_vectorizer,
                                      batch_size=self.batch_size,
                                      sequence_splitter=self.val_sequence_splitter, 
                                      user_id_required=self.model_arch.requires_user_id,
                                      max_user_features=self.max_user_features,
                                      user_features_required=not (self.users_featurizer is None), 
                                      targets_builder=self.targets_builder,
                                      shuffle_data=False
                                      )

        self.model = self.get_model(val_generator)
        if self.metric.less_is_better:
            best_metric_val = float('inf')
        else:
            best_metric_val = float('-inf')

        steps_since_improved = 0
        best_epoch = -1
        best_weights = self.model.get_weights()
        val_metric_history = []
        start_time = time.time()

        if not self.debug:
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])

        for epoch in range(self.train_epochs):
            val_generator.reset()
            generator = DataGenerator(train_users, train_user_ids, train_features, self.model_arch.max_history_length,
                                      self.items.size(),
                                      self.train_history_vectorizer,
                                      batch_size=self.batch_size,
                                      user_id_required=self.model_arch.requires_user_id,
                                      sequence_splitter=self.sequence_splitter,
                                      max_user_features=self.max_user_features,
                                      user_features_required=not (self.users_featurizer is None), 
                                      targets_builder=self.targets_builder, 
                                      shuffle_data=True
                                      )
            print(f"epoch: {epoch}")
            val_metric = self.train_epoch(generator, val_generator)

            total_trainig_time = time.time() - start_time
            val_metric_history.append((total_trainig_time, val_metric))

            steps_since_improved += 1
            if (self.metric.less_is_better and val_metric < best_metric_val) or\
                        (not self.metric.less_is_better and val_metric > best_metric_val):
                steps_since_improved = 0
                best_metric_val = val_metric
                best_epoch = epoch
                best_weights = self.model.get_weights()
            print(f"val_{self.metric.__name__}: {val_metric:.5f}, best_{self.metric.__name__}: {best_metric_val:.5f}, steps_since_improved: {steps_since_improved},"
                  f" total_training_time: {total_trainig_time}")
            if steps_since_improved >= self.early_stop_epochs:
                print(f"early stopped at epoch {epoch}")
                break

            if self.training_time_limit is not None and total_trainig_time > self.training_time_limit:
                print(f"time limit stop triggered at epoch {epoch}")
                break

            K.clear_session()
            gc.collect()
        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, f"best_val_{self.metric.__name__}": best_metric_val,
                         f"val_{self.metric.__name__}_history": val_metric_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_{self.metric.__name__}: {best_metric_val}")

    def pass_parameters(self):
        self.loss.set_num_items(self.items.size())
        self.loss.set_batch_size(self.batch_size)
        self.train_history_vectorizer.set_sequence_len(self.model_arch.max_history_length)
        self.train_history_vectorizer.set_padding_value(self.items.size())
        self.pred_history_vectorizer.set_sequence_len(self.model_arch.max_history_length)
        self.pred_history_vectorizer.set_padding_value(self.items.size())
        

    def train_epoch(self, generator, val_generator):
        if not self.debug:
            return self.train_epoch_prod(generator, val_generator)
        else:
            return self.train_epoch_debug(generator, val_generator)

    def train_epoch_debug(self, generator, val_generator):
        pbar = tqdm(generator, ascii=True)
        variables = self.model.variables
        loss_sum = 0
        metric_sum = 0
        num_batches = 0
        for X, y_true in pbar:
            num_batches += 1
            with tf.GradientTape() as tape:
                y_pred = self.model(X, training=True)
                loss_val = tf.reduce_mean(self.loss(y_true, y_pred))
            grad = tape.gradient(loss_val, variables)
            self.optimizer.apply_gradients(zip(grad, variables))
            metric = self.metric(y_true, y_pred)
            loss_sum += loss_val
            metric_sum += metric
            pbar.set_description(f"loss: {loss_sum/num_batches:.5f}, "
                                 f"{self.metric.__name__}: {metric_sum/num_batches:.5f}")
        val_loss_sum = 0
        val_metric_sum = 0
        num_val_samples = 0
        num_batches = 0
        for X, y_true in val_generator:
            num_batches += 1
            y_pred = self.model(X)
            loss_val = self.loss(y_true, y_pred)
            metric = self.metric(y_true, y_pred)
            val_metric_sum += metric
            val_loss_sum += loss_val
            num_val_samples += y_true.shape[0]
        val_metric = val_metric_sum / num_batches
        return float(val_metric)

    def train_epoch_prod(self, generator, val_generator):
        train_history = self.model.fit(generator, validation_data=val_generator)
        return train_history.history[f"val_{self.metric.__name__}"][-1]

    def train_val_split(self):
        all_user_ids = self.users_with_actions
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids)
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        train_features = [self.user_features.get(id, list()) for id in train_user_ids]
        val_features = [self.user_features.get(id, list()) for id in val_user_ids]
        return train_users, train_user_ids, train_features, val_users, val_user_ids, val_features

    def get_model(self, val_generator):
        self.max_user_features, self.max_user_feature_val = self.get_max_user_features()
        self.model_arch.set_common_params(num_items=self.items.size(),
                                          num_users=self.users.size(),
                                          max_user_features=self.max_user_features,
                                          user_feature_max_val=self.max_user_feature_val,
                                          batch_size=self.batch_size,
                                          item_features=self.item_features,
                                          )
        model = self.model_arch.get_model()

        #call the model first time in order to build it
        X, y = val_generator[0]
        model(X)
        return model

    def recommend(self, user_id, limit, features=None):
        scores = self.get_all_item_scores(user_id)
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            user_id = request.user_id
            scores = self.get_all_item_scores(user_id)
            user_result = []
            for item_id in request.item_ids:
                if (self.items.has_item(item_id)) and (self.items.get_id(item_id) < len(scores)):
                    user_result.append((item_id, scores[self.items.get_id(item_id)]))
                else:
                    user_result.append((item_id, float("-inf")))
            user_result.sort(key = lambda x: -x[1])
            result[user_id] = user_result
        return result

    def get_all_item_scores(self, user_id):
        actions = self.user_actions[self.users.get_id(user_id)]
        items_list = [action[1] for action in actions]
        model_actions = [(0, action) for action in items_list]
        session = self.pred_history_vectorizer(model_actions)
        session = session.reshape(1, self.model_arch.max_history_length)
        model_inputs = [session]
        if (self.model_arch.requires_user_id):
            model_inputs.append(np.array([[self.users.get_id(user_id)]]))

        if self.users_featurizer is not None:
            user_features = self.user_features.get(self.users.get_id(user_id), list())
            features_vector = DataGenerator.get_features_matrix([user_features], self.max_user_features)
            model_inputs.append(features_vector)

        if hasattr(self.model, 'score_all_items'):
            scores = self.model.score_all_items(model_inputs)[0].numpy()
        else: 
            scores = self.model(model_inputs)[0].numpy()
        return scores

    def get_max_user_features(self):
        result = 0
        max_val = 0
        for user_id in self.user_features:
            features = self.user_features[user_id]
            result = max(result, len(features))
            for feature in features:
                max_val = max(feature, max_val)
        return result, max_val
