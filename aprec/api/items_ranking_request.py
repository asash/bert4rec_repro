from typing import List, Union


class ItemsRankingRequest:
    def __init__(self,
                 user_id: Union[str, int],
                 item_ids: List[Union[str, int]]) -> None:
        self.user_id = user_id
        self.item_ids = item_ids

    def __str__(self) -> str:
        items = [str(v) for v in self.item_ids]
        return f"user_id={self.user_id} item_ids=[{','.join(items)}]"

    def __repr__(self) -> str:
        return self.__str__()
