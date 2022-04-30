class HistoryVectorizer(object):
    def __init__(self) -> None:
        self.sequence_len = None
        self.padding_value = None

    def set_sequence_len(self, sequence_len):
        self.sequence_len =  sequence_len

    def set_padding_value(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, user_actions):
        raise NotImplementedError
