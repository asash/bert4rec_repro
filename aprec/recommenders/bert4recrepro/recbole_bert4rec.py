import os

import torch
import tqdm

from aprec.recommenders.recommender import Recommender
from aprec.api.action import Action
import tempfile
from aprec.utils.item_id import ItemId
from recbole.config import Config
from recbole.data import create_dataset, get_dataloader
from recbole.utils import get_model, get_trainer
from collections import defaultdict


RECBOLE_CONFIG = """
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
data_path: {data_path}
eval_args:
    group_by: user
    split:
        RS: [1.0]
    order:
        TO

load_col:
    inter: [user_id, item_id, rating, timestamp]

user_inter_num_interval: "[0,inf)"
item_inter_num_interval: "[0,inf)"
val_interval:
    rating: "[0,inf)"
    timestamp: "[0, inf)"
"""


class RecboleBERT4RecRecommender(Recommender):
    def __init__(self, max_sequence_len=50, epochs=None):
        super().__init__()
        self.item_id = ItemId()
        self.user_id = ItemId()
        assert (self.item_id.get_id('[PAD]') == 0)
        assert (self.user_id.get_id('[PAD]') == 0)
        self.actions = []
        self.extra_config = f"max_sequence_len: {max_sequence_len}\n"
        self.max_sequence_len = max_sequence_len
        self.user_actions = defaultdict(list)
        if epochs is not None:
            self.extra_config += f"epochs: {epochs}\n"

    def add_action(self, action: Action):
        user_id = self.user_id.get_id(action.user_id)
        item_id = self.item_id.get_id(action.item_id)
        timestamp = action.timestamp
        self.user_actions[user_id].append(item_id)
        self.actions.append((user_id, item_id, timestamp))

    def build_recbole_model(self, dataset_name, yaml_filename):
        parameter_dict = {
            'neg_sampling': None,
        }
        config = Config(model='BERT4Rec',
                        dataset=dataset_name,
                        config_file_list=[yaml_filename],
                        config_dict=parameter_dict)
        #interactions = self.get_interactions()
        self.train_dataset = create_dataset(config).build()[0]
        self.train_data = get_dataloader(config, 'train')(config, self.train_dataset, None, shuffle=True)
        self.model = get_model(config['model'])(config, self.train_data.dataset).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, self.model)
        trainer.fit(self.train_data, show_progress=config['show_progress'])
        pass

    def get_sequence(self, user_id):
            user_actions = []
            for action in self.user_actions[user_id][-self.max_sequence_len:]:
                item_id = self.train_dataset.token2id('item_id', str(action))
                user_actions.append(item_id)
            seq_len = len(user_actions)
            if len(user_actions) < self.max_sequence_len:
                user_actions += [0] * (self.max_sequence_len - seq_len)
            return torch.tensor([user_actions]), torch.tensor([seq_len])

    def rebuild_model(self):
        with tempfile.TemporaryDirectory(prefix = "recbole_bert4rec_") as tempdir:
            dataset_name = os.path.basename(tempdir)
            data_path = os.path.dirname(tempdir)
            interactions_filename = dataset_name + ".inter"
            full_interactions_filename = os.path.join(tempdir, interactions_filename)
            with open(full_interactions_filename, 'w') as output:
                 output.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
                 for user_id, item_id, timestamp in self.actions:
                     output.write(f"{user_id}\t{item_id}\t1\t{timestamp}\n")
                 pass
            yaml_filename = dataset_name + ".yaml"
            full_yaml_filename = os.path.join(tempdir, yaml_filename)
            with open(full_yaml_filename, 'w') as output:
                output.write(RECBOLE_CONFIG.format(data_path=data_path))
                output.write(self.extra_config)
            self.build_recbole_model(dataset_name, full_yaml_filename)

    def recommend(self, user_id, limit: int, features=None):
        scores = self.get_user_scores(user_id)
        best_scores = torch.topk(scores, k=limit)
        result = []
        for (id, val) in zip(best_scores.indices, best_scores.values):
            token = self.train_dataset.id2token('item_id', id)
            if token == '[PAD]': continue
            internal_id = int(token)
            item_id = self.item_id.reverse_id(internal_id)
            result.append((item_id, float(val)))            
        return result

    def get_user_scores(self, user_id):
        internal_id = self.user_id.get_id(user_id)
        user_sequence, user_seq_len  = self.get_sequence(internal_id)
        interaction = {'item_id_list': user_sequence.to(self.model.device), 'item_length':user_seq_len.to(self.model.device)}
        scores = self.model.full_sort_predict(interaction)
        return scores[0]

    def get_item_rankings(self):
        result = {}
        print('generating sampled predictions...')
        for request in tqdm.tqdm(self.items_ranking_requests, ascii=True):
            scores = self.get_user_scores(request.user_id) 
            user_result = []
            for item_id in request.item_ids:
                if self.item_id.has_item(item_id):
                    token = str(self.item_id.get_id(item_id))
                    id = self.train_dataset.token2id('item_id', token)
                    user_result.append((item_id, float(scores[id])))
                else:
                    user_result.append((item_id, float("-inf")))
            user_result.sort(key=lambda x: -x[1])
            result[request.user_id] = user_result
        return result

