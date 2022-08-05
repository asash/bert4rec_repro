class User(object):
    def __init__(self, user_id, cat_features=None, real_features=None):
        if real_features is None:
            real_features = dict()

        if cat_features is None:
            cat_features = dict()

        self.user_id = user_id
        self.cat_features = cat_features
        self.real_features = real_features
