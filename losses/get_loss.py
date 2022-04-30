from aprec.losses.bpr import BPRLoss
from aprec.losses.bce import BCELoss
from aprec.losses.climf import CLIMFLoss
from aprec.losses.top1 import TOP1Loss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.xendcg import XENDCGLoss
import tensorflow as tf

losses = {
    'xendcg': XENDCGLoss, 
    'bpr': BPRLoss, 
    'climf': CLIMFLoss, 
    'bce': BCELoss, 
    'top1': TOP1Loss, 
}

def get_loss(loss_name, items_num, batch_size, max_positives=40,
             internal_dtype=tf.float32, lambda_normalization=True,
             lambdarank_pred_truncate=None,
             lambdarank_bce_weight=0.0,
             ):
    if loss_name == 'lambdarank':
        return LambdaGammaRankLoss(num_items=items_num, batch_size=batch_size, ndcg_at=max_positives,
                                   dtype=internal_dtype,
                                   lambda_normalization=lambda_normalization,
                                   pred_truncate_at=lambdarank_pred_truncate,
                                   bce_grad_weight=lambdarank_bce_weight)
    return losses[loss_name](num_items=items_num, batch_size=batch_size)
