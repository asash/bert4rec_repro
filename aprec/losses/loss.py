class Loss():
    def __init__(self, num_items=None, batch_size=None):
        self.num_items = num_items
        self.batch_size = batch_size

    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def set_num_items(self, num_items):
        self.num_items = num_items

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size