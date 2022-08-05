# -*- coding: UTF-8 -*-
import os
import collections
import random

import tensorflow as tf

from aprec.recommenders.BERT4rec.util import *
from aprec.recommenders.BERT4rec.vocab import *
import pickle
import multiprocessing
from argparse import ArgumentParser
import time


random_seed = 12345


def parse_args():
    global FLAGS
    parser = ArgumentParser()
    parser.add_argument("--signature", default="default", help="signature_name", type=str)
    parser.add_argument("--pool_size", default=10, help="multiprocesses pool size.", type=int)
    parser.add_argument("--max_seq_length", default=200, help="max sequence length", type=int)
    parser.add_argument("--max_predictions_per_seq", default=20, help="max_predictions_per_seq.", type=int)
    parser.add_argument("--masked_lm_prob", default=0.15, help="Masked LM probability.", type=float)
    parser.add_argument("--mask_prob", default=1.0, help="mask probability", type=float)
    parser.add_argument("--dupe_factor", default=10,
                        help="Number of times to duplicate the input data (with different masks).", type=int)
    parser.add_argument("--prop_sliding_window", default=0.1, help="sliding window step size.", type=float)
    parser.add_argument("--data_dir", default='./data/', help="data dir.", type=str)
    parser.add_argument("--dataset_name", default='./ml-1m/', help="dataset name.", type=str)
    FLAGS = parser.parse_args()




def printable_text(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_file(instances, max_seq_length,
                                   max_predictions_per_seq, vocab,
                                   output_file):
    """Create TF file from `TrainingInstance`s."""
    writer = tf.io.TFRecordWriter(output_file)
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["info"] = create_int_feature(instance.info)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(
            masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())

        total_written += 1

        if inst_index < 20:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.compat.v1.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    writer.close()
    tf.compat.v1.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(all_users_raw, max_seq_length, dupe_factor, masked_lm_prob, max_predictions_per_seq, rng,
                              vocab, mask_prob, prop_sliding_window, pool_size, force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = get_training_documents(all_users_raw, force_last, max_seq_length, prop_sliding_window)
    instances = create_instances(all_documents, dupe_factor, force_last, mask_prob, masked_lm_prob,
                                 max_predictions_per_seq, max_seq_length, pool_size, vocab)
    rng.shuffle(instances)
    return instances


def create_instances(all_documents, dupe_factor, force_last, mask_prob, masked_lm_prob, max_predictions_per_seq,
                     max_seq_length, pool_size, vocab):
    instances = []
    if force_last:
        instances = create_instances_force_last(all_documents, max_seq_length)
    else:
        instances = create_instances_no_force_last(all_documents, dupe_factor, instances, mask_prob, masked_lm_prob,
                                                   max_predictions_per_seq, max_seq_length, pool_size, vocab)
    return instances


def create_instances_no_force_last(all_documents, dupe_factor, instances, mask_prob, masked_lm_prob, max_predictions_per_seq,
                                   max_seq_length, pool_size, vocab):
    start_time = time.process_time()
    pool = multiprocessing.Pool(processes=pool_size)
    instances = []
    print("document num: {}".format(len(all_documents)))

    def log_result(result):
        print("callback function result type: {}, size: {} ".format(type(result), len(result)))
        instances.extend(result)

    for step in range(dupe_factor):
        pool.apply_async(
            create_instances_threading, args=(
                all_documents, max_seq_length, masked_lm_prob,
                max_predictions_per_seq, vocab, random.Random(random.randint(1, 10000)),
                mask_prob, step), callback=log_result)
    pool.close()
    pool.join()
    for user in all_documents:
        new_instance = mask_last(all_documents, user, max_seq_length)
        instances.extend(new_instance)
    print("num of instance:{}; time:{}".format(len(instances), time.process_time() - start_time))
    return instances


def create_instances_force_last(all_documents, max_seq_length):
    instances = []
    for user in all_documents:
        document = all_documents[user]
        info = [int(user.split("_")[1])]
        instances.extend(
            create_instances_from_document_test(document, info, max_seq_length))
    print("num of instance:{}".format(len(instances)))
    return instances


def get_training_documents(all_users_raw, force_last, max_seq_length, prop_sliding_window):
    if force_last:
        return get_training_documents_force_last(all_users_raw, max_seq_length)
    else:
        return get_training_documents_no_force_last(all_users_raw, max_seq_length, prop_sliding_window)


def get_training_documents_no_force_last(all_users_raw, max_seq_length, prop_sliding_window):
    all_documents = {}
    max_num_tokens = max_seq_length  # we need two sentence
    sliding_step = (int)(
        prop_sliding_window *
        max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
    for user, item_seq in all_users_raw.items():
        if len(item_seq) == 0:
            print("got empty seq:" + user)
            continue

        # todo: add slide
        if len(item_seq) <= max_num_tokens:
            all_documents[user] = [item_seq]
        else:
            beg_idx = list(range(len(item_seq) - max_num_tokens, 0, -sliding_step))
            beg_idx.append(0)
            all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]
    return all_documents


def get_training_documents_force_last(all_users_raw, max_seq_length):
    all_documents = {}
    max_num_tokens = max_seq_length
    for user, item_seq in all_users_raw.items():
        if len(item_seq) == 0:
            print("got empty seq:" + user)
            continue
        all_documents[user] = [item_seq[-max_num_tokens:]]
    return all_documents


def create_instances_threading(all_documents, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    cnt = 0
    start_time = time.process_time()
    instances = []
    for user in all_documents:
        cnt += 1
        if cnt % 1000 == 0:
            print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt, time.process_time()-start_time))
            start_time = time.process_time()
        document = all_documents[user]
        info = [int(user.split("_")[1])]
        instances.extend(create_instances_from_document_train(document, info, max_seq_length, masked_lm_prob,
                                                              max_predictions_per_seq, vocab, rng, mask_prob))
        
    return instances


def mask_last(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length
    
    instances = []
    info = [int(user.split("_")[1])]

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens
        
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def create_instances_from_document_test(document, info, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    max_num_tokens = max_seq_length
    
    assert len(document) == 1 and len(document[0]) <= max_num_tokens
    
    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    return [instance]


def create_instances_from_document_train(document, info, max_seq_length, masked_lm_prob, max_predictions_per_seq,
                                         vocab, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""

    max_num_tokens = max_seq_length

    instances = []
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens
        
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq,
             vocab_items, rng, mask_prob)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                masked_token = rng.choice(vocab_words)  

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def gen_samples(data, output_filename, rng, vocab, max_seq_length, dupe_factor, mask_prob, masked_lm_prob,
                max_predictions_per_seq, prop_sliding_window, pool_size, force_last=False):
    # create train
    instances = create_training_instances(data, max_seq_length, dupe_factor, masked_lm_prob, max_predictions_per_seq,
                                          rng, vocab, mask_prob, prop_sliding_window, pool_size, force_last)

    tf.compat.v1.logging.info("*** Writing to output files ***")
    tf.compat.v1.logging.info("  %s", output_filename)

    write_instance_to_example_file(instances, max_seq_length,
                                   max_predictions_per_seq, vocab,
                                   output_filename)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    
    max_seq_length = FLAGS.max_seq_length
    max_predictions_per_seq = FLAGS.max_predictions_per_seq
    masked_lm_prob = FLAGS.masked_lm_prob
    mask_prob = FLAGS.mask_prob
    dupe_factor = FLAGS.dupe_factor
    prop_sliding_window = FLAGS.prop_sliding_window
    pool_size = FLAGS.pool_size

    output_dir = FLAGS.data_dir
    dataset_name = FLAGS.dataset_name
    version_id = FLAGS.signature
    print(version_id)

    if not os.path.isdir(output_dir):
        print(output_dir + ' is not exist')
        print(os.getcwd())
        exit(1)

    dataset = data_partition(output_dir+dataset_name+'.txt')
    [user_train, user_test, usernum, itemnum] = dataset
    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))
    print(f"len_train:{len(user_train)}, len_test:{len(user_test)}, usernum:{usernum}, itemnum:{itemnum}")

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }
    rng = random.Random(random_seed)

    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: [vocab.convert_tokens_to_ids(v)]
        for k, v in user_test_data.items()
    }

    print('begin to generate train')
    output_filename = output_dir + dataset_name + version_id + '.train.tfrecord'
    gen_samples(user_train_data, output_filename, rng, vocab, max_seq_length, dupe_factor, mask_prob, masked_lm_prob,
                max_predictions_per_seq, prop_sliding_window, pool_size, force_last=False)
    print('train:{}'.format(output_filename))

    print('begin to generate test')
    output_filename = output_dir + dataset_name + version_id + '.test.tfrecord'
    gen_samples(user_test_data, output_filename, rng, vocab, max_seq_length, dupe_factor, mask_prob, masked_lm_prob,
                max_predictions_per_seq, -1.0, pool_size, force_last=True)
    print('test:{}'.format(output_filename))

    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + dataset_name + version_id + '.his'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')


if __name__ == "__main__":
    parse_args()
    main()
