#base class for many sequential recsys models

class SequentialRecsysModel(object):
    def __init__(self, output_layer_activation, embedding_size, max_history_len):
        self.max_history_length = max_history_len
        self.embedding_size = embedding_size
        self.output_layer_activation = output_layer_activation
        self.requires_user_id = False
        self.num_items = None
        self.num_users = None
        self.max_user_features = None
        self.user_feature_max_val = None
        self.batch_size = None

    def set_common_params(self, num_items, num_users,
             max_user_features, user_feature_max_val, 
             batch_size, item_features=None, 
             ):
        self.num_items = num_items
        self.num_users = num_users
        self.max_user_features = max_user_features
        self.user_feature_max_val = user_feature_max_val
        self.batch_size = batch_size
        self.item_features = item_features

    def get_model(self):
        raise NotImplementedError
