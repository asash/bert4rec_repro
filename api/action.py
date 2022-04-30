import json
class Action(object):
    def __init__(self, user_id, item_id, timestamp, data=None):
        if data is None:
            data = dict()
        self.user_id = user_id
        self.item_id = item_id
        self.data = data
        self.timestamp = timestamp

    def to_str(self):
       result = "Action(uid={}, item={}, ts={}".format(
                    self.user_id, 
                    self.item_id, 
                    self.timestamp)
       if self.data != {}:
           result += ", data={}".format(str(self.data))
       result += ")"
       return result

    def to_json(self):
        try:
            #check if data is json serializable
            json.dumps(self.data)
            data = self.data

        except:
            #fallback to just string representation
            #TODO: restore may work incorrectly with some datasets
            data = str(self.data)

        return json.dumps({
            "user_id": self.user_id,
            "item_id": self.item_id,
            "data": data,
            "timestamp": self.timestamp
        })

    @staticmethod
    def from_json(action_str):
        doc = json.loads(action_str)
        return Action(doc["user_id"], doc["item_id"], doc["data"], doc["timestamp"])

    def __str__(self):
        return self.to_str()
        
    def __repr__(self):
        return self.to_str()
        
