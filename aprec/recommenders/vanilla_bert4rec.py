#This code uses port of the original bert4rec implementation
import copy
import json
import os
import pickle
import random
import shlex
import subprocess
import tempfile
from collections import defaultdict
from aprec.recommenders.BERT4rec.gen_data_fin import create_training_instances, write_instance_to_example_file
from aprec.recommenders.BERT4rec.vocab import FreqVocab
from aprec.recommenders.recommender import Recommender
from aprec.recommenders import BERT4rec
from aprec.utils.item_id import ItemId


class VanillaBERT4Rec(Recommender):
    def __init__(self,
                 max_seq_length = 20,
                 masked_lm_prob = 0.2,
                 max_predictions_per_seq = 20,
                 batch_size = 256,
                 num_train_steps = 400000,
                 prop_sliding_window = 0.5,
                 mask_prob = 1.0,
                 dupe_factor = 10,
                 pool_size = 10,
                 num_warmup_steps = 100,
                 learning_rate = 1e-4,
                 random_seed = 31337,
                 training_time_limit=None,

                 attention_probs_dropout_prob = 0.2,
                 hidden_act = "gelu",
                 hidden_dropout_prob = 0.2,
                 hidden_size = 64,
                 initializer_range = 0.02,
                 intermediate_size = 256,
                 max_position_embeddings = 200,
                 num_attention_heads = 2,
                 num_hidden_layers = 2,
                 type_vocab_size = 2):
        super().__init__()
        self.user_actions = defaultdict(list)
        self.user_ids = ItemId()
        self.item_ids = ItemId()
        self.max_seq_length = max_seq_length
        self.dupe_factor = dupe_factor
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.rng = random.Random(random_seed)
        self.mask_prob = mask_prob
        self.prop_sliding_window = prop_sliding_window
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.training_time_limit = training_time_limit
        self.predictions_cache = {}

        bert_config = {
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": hidden_dropout_prob,
            "hidden_size": hidden_size,
            "initializer_range": initializer_range,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "type_vocab_size": type_vocab_size,
        }
        self.bert_config = bert_config

    def add_action(self, action):
        self.user_actions[action.user_id].append(action)

    def rebuild_model(self):
        pred_item = ["[PRED]"]
        bert4rec_docs = {}
        bert4rec_pred_docs = {}

        for user in self.user_actions:
            self.user_actions[user].sort(key=lambda action: action.timestamp)
            user_str, doc = self.get_bert4rec_doc(user)
            doc_for_prediction = doc + pred_item

            #mask_last code in the vanilla bert4rec implementation requires at least two docs in the collection
            if len(doc) > 1:
                bert4rec_docs[user_str] = doc

            bert4rec_pred_docs[user_str] = doc_for_prediction

        vocab = FreqVocab(bert4rec_pred_docs)
        user_test_data_output = {
            k: [vocab.convert_tokens_to_ids(v)]
            for k, v in bert4rec_pred_docs.items()
        }

        train_instances = create_training_instances(bert4rec_docs, self.max_seq_length, self.dupe_factor, self.masked_lm_prob,
                                              self.max_predictions_per_seq, self.rng, vocab, self.mask_prob,
                                              self.prop_sliding_window, self.pool_size, False)

        pred_instances = create_training_instances(bert4rec_pred_docs, self.max_seq_length, 1, self.masked_lm_prob,
                                              self.max_predictions_per_seq, self.rng, vocab, self.mask_prob,
                                              -1, self.pool_size, True)


        with tempfile.TemporaryDirectory() as tmpdir:
            train_instances_filename = os.path.join(tmpdir, "train_instances.tfrecords")
            train_instances_file = open(train_instances_filename, "wb")

            pred_instances_filename = os.path.join(tmpdir, "pred_instances.tfrecords")
            pred_instances_file = open(pred_instances_filename, "wb")


            sampled_instances_filename = os.path.join(tmpdir, "sampled_instances.csv")
            self.write_sampled_instances(sampled_instances_filename)


            bert_config_filename = os.path.join(tmpdir, "bert_config_file.json")
            bert_config_file = open(bert_config_filename, "wb")

            vocab_filename = os.path.join(tmpdir, "vocab.pickle")
            vocab_file = open(vocab_filename, "wb")

            history_filename =  os.path.join(tmpdir, "history.pickle")
            history_file = open(history_filename, "wb")


            predictions_filename =  os.path.join(tmpdir, "predictions.csv")
            sampled_predictions_filename = os.path.join(tmpdir, "sampled_predictions")

            write_instance_to_example_file(train_instances,
                                       self.max_seq_length,
                                       self.max_predictions_per_seq, vocab, train_instances_file.name,)

            write_instance_to_example_file(pred_instances,
                                           self.max_seq_length,
                                           self.max_predictions_per_seq, vocab, pred_instances_file.name,)
            bert_config = copy.deepcopy(self.bert_config)
            bert_config["vocab_size"] = vocab.get_vocab_size()
            bert_config_file.write(json.dumps(bert_config, indent=4).encode("utf-8"))
            bert_config_file.flush()
            pickle.dump(vocab, vocab_file, protocol=2)
            pickle.dump(user_test_data_output, history_file, protocol=2)
            self.train_and_predict(train_instances_filename,
                                   pred_instances_filename,
                                   vocab_filename,
                                   history_filename,
                                   bert_config_filename,
                                   predictions_filename,
                                   sampled_instances_filename,
                                   sampled_predictions_filename,
                                   tmpdir)

    def get_bert4rec_doc(self, user):
        user_id = self.user_ids.get_id(user)
        user_str = f"user_{user_id}"
        doc = [f"item_{self.item_ids.get_id(action.item_id)}" for action in self.user_actions[user]]
        return user_str, doc

    def train_and_predict(self, train_instances_filename,
                          pred_instances_filename,
                          vocab_filename, user_history_filename,
                          bert_config_filename,
                          predictions_filename,
                          sampled_instances_filename,
                          sampled_predictions_file,
                          tmpdir):

        bert4rec_dir = os.path.dirname(BERT4rec.__file__)
        bert4rec_runner = os.path.join(bert4rec_dir, "run.py")
        signature = tmpdir.split("/")[-1]
        cmd = f"python {bert4rec_runner}\
            --train_input_file={train_instances_filename} \
            --test_input_file={pred_instances_filename} \
            --vocab_filename={vocab_filename} \
            --user_history_filename={user_history_filename} \
            --checkpointDir={tmpdir} \
            --signature={signature}\
            --do_train=True \
            --do_eval=True \
            --bert_config_file={bert_config_filename} \
            --batch_size={self.batch_size} \
            --max_seq_length={self.max_seq_length} \
            --max_predictions_per_seq={self.max_predictions_per_seq} \
            --num_train_steps={self.num_train_steps} \
            --num_warmup_steps={self.num_warmup_steps} \
            --save_predictions_file={predictions_filename} \
            --sampled_instances_file={sampled_instances_filename} \
            --save_sampled_predictions_file={sampled_predictions_file}\
            --learning_rate={self.learning_rate} "

        if self.training_time_limit is not None:
            cmd += f" --training_time_limit={self.training_time_limit}"

        subprocess.check_call(shlex.split(cmd))
        self.predictions_cache = self.read_predictions_cache(predictions_filename)
        self.sampled_items_ranking_predictions_cache = self.read_predictions_cache(sampled_predictions_file)
        pass


    def read_predictions_cache(self, predictions_filename):
        result = {}
        with open(predictions_filename) as predictions:
            for line in predictions:
                splits = line.strip().split(';')
                user_id = splits[0]
                result[user_id] = []
                for item_with_score in splits[1:]:
                    item, score = item_with_score.split(":")
                    score = float(score)
                    result[user_id].append((item, score))
        return result

    def recommend(self, user_id, limit, features=None):
        internal_user_id = "user_" + str(self.user_ids.get_id(user_id))
        if internal_user_id not in self.predictions_cache:
            return []
        recs = self.predictions_cache[internal_user_id]
        result = []
        for internal_item_id, score in recs[:limit]:
            if not internal_item_id.startswith("item_"):
                continue
            item_id = self.item_ids.reverse_id(int(internal_item_id.split("_")[1]))
            result.append((item_id, score))
        return result

    def write_sampled_instances(self, sampled_instances_filename):
        sampled_instances_file = open(sampled_instances_filename, "w")
        for request in self.items_ranking_requests:
            user_id = "user_{}".format(self.user_ids.get_id(request.user_id))
            item_ids = [f"item_{self.item_ids.get_id(item)}" for item in request.item_ids]
            sampled_instances_file.write(";".join([user_id] + item_ids) + "\n")
        sampled_instances_file.close()


    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            internal_id = "user_{}".format(self.user_ids.get_id(request.user_id))
            predictions = self.sampled_items_ranking_predictions_cache[internal_id]
            user_result = []
            for item, score in predictions:
                item_id = int(item.split("_")[1])
                external_item_id = self.item_ids.reverse_id(item_id)
                user_result.append((external_item_id, score))
            result[request.user_id] = user_result
        return result


