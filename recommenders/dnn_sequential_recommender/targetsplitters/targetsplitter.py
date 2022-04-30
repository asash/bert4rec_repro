class TargetSplitter(object):
    def __init__(self) -> None:
        self.num_items = None 
        self.seqence_len = None

    def split(self, sequence):
        return NotImplementedError() 

    def set_num_items(self, num_items):
        self.num_items = num_items

    def set_sequence_len(self, sequence_len):
        self.seqence_len = sequence_len
