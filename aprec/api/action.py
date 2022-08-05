import json
from typing import Dict, Optional, Union


class Action:
    def __init__(
        self,
        user_id: Union[str, int],
        item_id: Union[str, int],
        timestamp: int,
        data: Optional[Dict[str, Union[float, int]]] = None,
    ):
        self.data: Dict[str, Union[float, int]] = data if data is not None else {}
        self.user_id = user_id
        self.item_id = item_id
        self.timestamp = timestamp

    def to_str(self) -> str:
        result = "Action(uid={}, item={}, ts={}".format(
            self.user_id, self.item_id, self.timestamp
        )
        if self.data != {}:
            result += ", data={}".format(str(self.data))
        result += ")"
        return result

    def to_json(self) -> str:
        try:
            # check if data is json serializable
            json.dumps(self.data)
            data = self.data

        except Exception:
            # fallback to just string representation
            # TODO: restore may work incorrectly with some datasets
            data = {}

        return json.dumps(
            {
                "user_id": self.user_id,
                "item_id": self.item_id,
                "data": data,
                "timestamp": self.timestamp,
            }
        )

    @staticmethod
    def from_json(action_str: str) -> "Action":
        doc = json.loads(action_str)
        return Action(doc["user_id"], doc["item_id"], doc["data"], doc["timestamp"])

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()
