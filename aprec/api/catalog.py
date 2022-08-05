def relevancy (keyword, string):
    if keyword.lower() == string.lower():
        return -1
    return keyword.lower().find(string.lower())

class Catalog(object):
    def __init__(self):
        self.items = {}

    def add_item(self, item):
        self.items[item.item_id] = item

    def get_item(self, item_id):
        return self.items[item_id]

    def search(self, keyword):
        result = []
        for item in self.items.values():
            if keyword.lower() in item.title.lower():
                result.append(item)
        result.sort(key=lambda value: relevancy(keyword, value.title))
        return result

