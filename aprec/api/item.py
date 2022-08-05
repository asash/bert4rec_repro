from typing import List, Optional, Union


class Item(object):
    item_id: Union[str, int]
    cat_features: List[str]
    real_features: List[str]
    tags: Optional[List[str]] = None
    title: Optional[str] = None

    def __init__(
        self,
        item_id: Union[str, int],
        cat_features: Optional[List[str]] = None,
        real_features: Optional[List[str]] = None,
    ) -> None:
        if real_features is None:
            self.real_features = []
        else:
            self.real_features = real_features

        if cat_features is None:
            self.cat_features = []
        else:
            self.cat_features = cat_features

        self.item_id = item_id

    def with_tags(self, tags: List[str]) -> "Item":
        self.tags = tags
        return self

    def with_title(self, title: str) -> "Item":
        self.title = title
        return self

    def __str__(self) -> str:
        return "item id={} title={} tags={}".format(self.item_id, self.title, self.tags)

    def __repr__(self) -> str:
        return self.__str__()
