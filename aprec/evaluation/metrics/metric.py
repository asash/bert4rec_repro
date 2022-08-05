class Metric(object):
    def __call__(self, recommendations, actual):
        raise NotImplementedError
