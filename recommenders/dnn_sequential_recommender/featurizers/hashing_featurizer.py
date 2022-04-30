import mmh3

class HashingFeaturizer(object):
    def __init__(self, num_cat_hashes=3, cat_hashes_space=1000):
        self.num_cat_hashes = num_cat_hashes
        self.cat_hashes_space = cat_hashes_space

    def __call__(self, obj):
        result = []
        if type(obj.cat_features) == dict:
            features = list(obj.cat_features.items())
        else:
            features = obj.cat_features

        for feature in features:
            for hash_num in range(self.num_cat_hashes):
                val = f"{feature[0]}_" + str(feature[1]) + f"_hash{hash_num}"
                hash_val = mmh3.hash(val) % self.cat_hashes_space + 1
                result.append(hash_val)
        return result
