import tensorflow as tf

from aprec.losses.bce import BCELoss
from aprec.losses.bpr import BPRLoss
from aprec.losses.climf import CLIMFLoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.top1 import TOP1Loss
from aprec.losses.xendcg import XENDCGLoss

losses = {
    "xendcg": XENDCGLoss,
    "bpr": BPRLoss,
    "climf": CLIMFLoss,
    "bce": BCELoss,
    "top1": TOP1Loss,
}


def get_loss(
    loss_name: str,
    items_num: int,
    batch_size: int,
    max_positives: int = 40,
    internal_dtype: type = tf.float32,
    lambda_normalization: bool = True,
    lambdarank_pred_truncate=None,
    lambdarank_bce_weight=0.0,
) -> tf.Tensor:
    if loss_name == "lambdarank":
        return LambdaGammaRankLoss(
            num_items=items_num,
            batch_size=batch_size,
            ndcg_at=max_positives,
            dtype=internal_dtype,
            lambda_normalization=lambda_normalization,
            pred_truncate_at=lambdarank_pred_truncate,
            bce_grad_weight=lambdarank_bce_weight,
        )
    else:
        return losses[loss_name](num_items=items_num, batch_size=batch_size)
