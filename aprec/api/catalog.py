from typing import Dict, List

from .item import Item


def relevancy(keyword: str, string: str) -> int:
    if keyword.lower() == string.lower():
        return -1
    return keyword.lower().find(string.lower())


class Catalog:
    items: Dict[str, Item]

    def __init__(self) -> None:
        self.items = {}

    def add_item(self, item: Item) -> None:
        self.items[item.item_id] = item

    def get_item(self, item_id: str) -> Item:
        return self.items[item_id]

    def search(self, keyword: str) -> List[Item]:
        result: List[Item] = []
        for item in self.items.values():
            if item.title and keyword.lower() in item.title.lower():
                result.append(item)

        def get_relevancy(value: Item) -> int:
            assert value.title is not None
            return relevancy(keyword, value.title)

        result.sort(key=get_relevancy)
        return result
