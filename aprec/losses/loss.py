from typing import Optional

import tensorflow as tf


class Loss:
    num_items: Optional[int]
    batch_sizes: Optional[int]

    def __init__(self,
                 num_items: Optional[int] = None,
                 batch_size: Optional[int] = None) -> None:
        self.num_items = num_items
        self.batch_size = batch_size

    def __call__(self,
                 y_true_raw: tf.Tensor,
                 y_pred: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def set_num_items(self, num_items) -> None:
        self.num_items = num_items

    def set_batch_size(self, batch_size) -> None:
        self.batch_size = batch_size
