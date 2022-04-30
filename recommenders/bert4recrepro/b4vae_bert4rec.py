import os
import torch
import tqdm
from aprec.recommenders.recommender import Recommender
from aprec.api.action import Action
from aprec.utils.item_id import ItemId
from b4rvae.dataloaders.bert import BertDataloader
from b4rvae.models.bert import BERTModel
from b4rvae.trainers.bert import BERTTrainer

from tempfile import TemporaryDirectory

class B4rVaeDataset(object):
    def __init__(self, user_actions, val_actions, user_id, item_id):
        self.user_actions = user_actions
        self.val_actions = val_actions
        self.user_id = user_id
        self.item_id = item_id
        self.tempdir = None
        
    def __enter__(self):
        self.tempdir = TemporaryDirectory(prefix="B4RVAE_") 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tempdir.cleanup()

    def _get_preprocessed_folder_path(self):
        return self.tempdir.name

    def load_dataset(self):
        return {"train": self.user_actions, 
                "val": self.val_actions,
                "test": {},
                "umap": self.user_id, 
                "smap": self.item_id
        }


class B4rVaeArgs(object):
    def __init__(self):
        self.dataloader_random_seed = 31337

        batch=128
        self.train_batch_size = batch
        self.val_batch_size = batch
        self.test_batch_size = batch

        self.train_negative_sampler_code = 'random'
        self.train_negative_sample_size = 0
        self.train_negative_sampling_seed = 0
        self.test_negative_sampler_code = 'random'
        self.test_negative_sample_size = 100
        self.test_negative_sampling_seed = 98765

        self.trainer_code = 'bert'
        self.device = 'cuda'
        self.num_gpu = 1
        self.device_idx = '0'
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.enable_lr_schedule = True
        self.decay_step = 25
        self.gamma = 1.0
        self.num_epochs = 200 
        self.metric_ks = [1, 5, 10, 20, 50, 100]
        self.log_period_as_iter=128000
        self.best_metric = 'NDCG@10'
        self.weight_decay = 0 

        self.model_code = 'bert'
        self.model_init_seed = 0

        self.bert_dropout = 0.1
        self.bert_hidden_units = 256
        self.bert_mask_prob = 0.15
        self.bert_max_len = 100
        self.bert_num_blocks = 2
        self.bert_num_heads = 4



class B4rVaeBert4Rec(Recommender):
    def __init__(self, epochs=None):
        super().__init__()
        self.item_id = ItemId()
        self.user_id = ItemId()
        assert(self.item_id.get_id('[PAD]') == 0)
        self.user_actions = {} 
        self.args = B4rVaeArgs()
        if epochs is not None:
            self.args.num_epochs = epochs

    def add_action(self, action: Action):
        item_id = self.item_id.get_id(action.item_id)
        user_id = self.user_id.get_id(action.user_id)
        if user_id not in self.user_actions:
            self.user_actions[user_id] = [] 
        self.user_actions[user_id].append(item_id)

    def rebuild_model(self):
        self.mask_val = self.item_id.size() + 1
        self.val_actions = self.separate_val_actions() 
        with B4rVaeDataset(self.user_actions, self.val_actions,
                            self.user_id.straight, self.item_id.straight) as dataset:
            dataloader = BertDataloader(self.args, dataset) 
            train, val, test = dataloader.get_pytorch_dataloaders()
            self.model = BERTModel(self.args)
            tempdir = dataset._get_preprocessed_folder_path()
            trainer = BERTTrainer(self.args, self.model, train, val, test, tempdir)
            trainer.train()
            best_model = torch.load(os.path.join(tempdir, 'models', 'best_acc_model.pth')).get('model_state_dict')
            self.model.load_state_dict(best_model)
            self.model.eval()
            pass
    
    def separate_val_actions(self):
        result = {}
        for user_id in self.user_actions.keys():
            val_action = self.user_actions[user_id].pop()
            result[user_id] = [val_action]
        return result


    def get_pred_user_sequence(self, external_user_id):
        internal_id = self.user_id.get_id(external_user_id)
        actions_seq = self.user_actions[internal_id] + \
                      self.val_actions[internal_id] + \
                      [self.mask_val]
        actions_seq = actions_seq[-self.args.bert_max_len:]
        if len(actions_seq) < self.args.bert_max_len:
            actions_seq = [0] * (self.args.bert_max_len - len(actions_seq)) + actions_seq
        return torch.tensor([actions_seq]).to(self.args.device)


    def recommend(self, user_id, limit: int, features=None):
        scores = self.get_user_scores(user_id)
        best_scores =  torch.topk(scores, limit)
        result = []
        for (internal_id, val) in zip(best_scores.indices, best_scores.values):
            if internal_id < self.item_id.size():
                item_id = self.item_id.reverse_id(int(internal_id))
                if item_id == '[PAD]': continue
                result.append((item_id, float(val)))      
        return result[:limit]

    def get_user_scores(self, user_id):
        seq = self.get_pred_user_sequence(user_id)
        scores = self.model(seq)
        scores = scores[:, -1, :][0]
        return scores
    
    def get_item_rankings(self):
        result = {}
        print('generating sampled predictions...')
        for request in tqdm.tqdm(self.items_ranking_requests):
            scores = self.get_user_scores(request.user_id) 
            user_result = []
            for item_id in request.item_ids:
                if self.item_id.has_item(item_id):
                    user_result.append((item_id, float(scores[self.item_id.get_id(item_id)])))
                else:
                    user_result.append((item_id, float("-inf")))
            user_result.sort(key=lambda x: -x[1])
            result[request.user_id] = user_result
        return result
