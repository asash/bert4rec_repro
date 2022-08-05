import random

import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder

from aprec.recommenders.dnn_sequential_recommender.targetsplitters.random_fraction_splitter import RandomFractionSplitter

class DataGenerator(Sequence):
    def __init__(self, user_actions, user_ids,  user_features, history_size,
                 n_items, 
                 history_vectorizer,
                 batch_size=1000,
                 return_drect_positions=False, return_reverse_positions=False,
                 user_id_required=False,
                 max_user_features=0,
                 user_features_required=False, 
                 sequence_splitter = RandomFractionSplitter, 
                 targets_builder = FullMatrixTargetsBuilder,
                 shuffle_data = True
                 ):
        self.user_ids = [[id] for id in user_ids]
        self.user_actions = user_actions
        self.history_size = history_size
        self.n_items = n_items
        self.batch_size = batch_size
        self.sequences_matrix = None
        self.return_direct_positions = return_drect_positions
        self.return_reverse_positions = return_reverse_positions
        self.user_id_required = user_id_required
        self.user_features = user_features
        self.max_user_features = max_user_features
        self.user_features_required = user_features_required
        self.sequence_splitter = sequence_splitter()
        self.sequence_splitter.set_num_items(n_items)
        self.sequence_splitter.set_sequence_len(history_size)
        self.targets_builder = targets_builder()
        self.targets_builder.set_sequence_len(history_size)
        self.do_shuffle_data = shuffle_data
        self.history_vectorizer = history_vectorizer
        self.reset()


    def reset(self):
        if self.do_shuffle_data: 
            self.shuffle_data()
        history, target = self.split_actions(self.user_actions)
        self.sequences_matrix = self.matrix_for_embedding(history)
        if self.return_direct_positions or self.return_reverse_positions:
            self.direct_position, self.reverse_position = self.positions(history, self.history_size)

        if self.user_features_required:
            self.user_features_matrix = self.get_features_matrix(self.user_features, self.max_user_features)

        self.targets_builder.set_n_items(self.n_items)
        self.targets_builder.build(target)
        self.current_position = 0
        self.max = self.__len__()

    def shuffle_data(self):
        actions_with_ids_and_features = list(zip(self.user_actions, self.user_ids, self.user_features))
        random.shuffle(actions_with_ids_and_features)
        user_actions, user_ids, user_features = zip(*actions_with_ids_and_features)

        self.user_actions = user_actions
        self.user_ids = user_ids
        self.user_features = user_features

    @staticmethod
    def get_features_matrix(user_features, max_user_features):
        result = []
        for features in user_features:
            result.append([0] * (max_user_features - len(features)) + features)
        return np.array(result)


    def matrix_for_embedding(self, user_actions):
        result = []
        for actions in user_actions:
            result.append(self.history_vectorizer(actions))
        return np.array(result)

    def build_target_matrix(self, user_targets):
        if self.sampled_target is None:
            self.build_full_target_matrix(user_targets)
        else:
            self.build_sampled_targets(user_targets)

    def split_actions(self, user_actions):
        history = []
        target = []
        for user in user_actions:
            user_history, user_target = self.sequence_splitter.split(user)
            history.append(user_history)
            target.append(user_target)
        return history, target

    def positions(self, sessions, history_size):
        result_direct, result_reverse = [], []
        for session in sessions:
            result_direct.append(direct_positions(len(session), history_size))
            result_reverse.append(reverse_positions(len(session), history_size))
        return np.array(result_direct), np.array(result_reverse)

    def __len__(self):
        return self.sequences_matrix.shape[0] // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        history = self.sequences_matrix[start:end]
        model_inputs = [history]
        if self.return_direct_positions:
            direct_pos = self.direct_position[start:end]
            model_inputs.append(direct_pos)
        if self.return_reverse_positions:
            reverse_pos = self.reverse_position[start:end]
            model_inputs.append(reverse_pos)

        if self.user_id_required:
            user_ids = np.array(self.user_ids[start:end])
            model_inputs.append(user_ids)

        if self.user_features_required:
            features = self.user_features_matrix[start:end]
            model_inputs.append(features)

        target_inputs, target = self.targets_builder.get_targets(start, end)
        model_inputs += target_inputs

        return model_inputs, target 

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result

def direct_positions(session_len, history_size):
    if session_len >= history_size:
        return list(range(1, history_size + 1))
    else:
        return [0] * (history_size - session_len) + list(range(1, session_len + 1))


def reverse_positions(session_len, history_size):
    if session_len >= history_size:
        return list(range(history_size, 0, -1))
    else:
        return [0] * (history_size - session_len) + list(range(session_len, 0, -1))
