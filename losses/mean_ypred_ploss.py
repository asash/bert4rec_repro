import tensorflow as tf

from aprec.losses.loss import Loss

#this is a dummy loss function, that does not use y_true
#It can be useful when the model itself already computes loss 
#Example = BERT masking model

class MeanPredLoss(Loss):
    def __init__(self, num_items=None, batch_size=None, name="mean_ypred"):
        super().__init__(num_items, batch_size)
        self.__name__ = name 
        self.less_is_better=True

    def __call__(self, y_true, y_pred):
        result = tf.reduce_mean(y_pred)
        return result
