from typing import Optional

import tensorflow as tf

from aprec.losses.loss import Loss


class MeanPredLoss(Loss):
    """this is a dummy loss function, that does not use y_true
    It can be useful when the model itself already computes loss
    Example = BERT masking model"""

    def __init__(
        self,
        num_items: Optional[int] = None,
        batch_size: Optional[int] = None,
        name: str = "mean_ypred",
    ) -> None:
        super().__init__(num_items, batch_size)
        self.__name__ = name
        self.less_is_better = True

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        result = tf.reduce_mean(y_pred)
        return result
