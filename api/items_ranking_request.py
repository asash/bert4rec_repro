from typing import List


class ItemsRankingRequest(object):
    def __init__(self, user_id, item_ids: List):
        self.user_id = user_id
        self.item_ids = item_ids

    def __str__(self):
        return f"user_id={self.user_id} item_ids=[{','.join(self.item_ids)}]"

    def __repr__(self):
        return self.__str__()
