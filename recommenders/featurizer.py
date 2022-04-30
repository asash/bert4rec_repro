from aprec.api.action import Action
from aprec.api.user import User
from aprec.api.item import Item

class Featurizer(object):
    def __init__(self):
        pass

    def add_action(self, action: Action):
        pass

    def add_user(self, user: User):
        pass

    def add_item(self, item: Item):
        pass

    def get_features(self, user_id, item_id):
        pass

    def build(self):
        pass
