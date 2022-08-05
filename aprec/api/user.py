from typing import List, Optional, Union


class User:
    def __init__(self,
                 user_id: Union[str, int],
                 cat_features: Optional[List[str]] = None,
                 real_features: Optional[List[str]] = None):
        if real_features is None:
            self.real_features = []
        else:
            self.real_features = real_features

        if cat_features is None:
            self.cat_features = []
        else:
            self.cat_features = cat_features

        self.user_id = user_id
