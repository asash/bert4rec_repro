import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import activations

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
class SASRec(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, 
                 dropout_rate=0.2,
                 num_blocks=3,
                 num_heads=1,
                 reuse_item_embeddings=True, #use same item embeddings for
                                             # sequence embedding and for the embedding matrix
                 encode_output_embeddings=False, #encode item embeddings with a dense layer
                                                          #may be useful if we reuse item embeddings
                 vanilla=False, #vanilla sasrec model uses shifted sequence prediction at the training time. 
                 sampled_targets=None,
                 ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.reuse_item_embeddings=reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.sampled_targets = sampled_targets
        self.vanilla = vanilla


    encode_embedding_with_dense_layer = False,

    def get_model(self):
        model = OwnSasrecModel(self.num_items, self.batch_size, self.output_layer_activation,
                               self.embedding_size,
                               self.max_history_length,
                               self.dropout_rate,
                               self.num_blocks,
                               self.num_heads,
                               self.reuse_item_embeddings,
                               self.encode_output_embeddings,
                               self.sampled_targets, 
                               vanilla=self.vanilla
        )
        return model



class OwnSasrecModel(tensorflow.keras.Model):
    def __init__(self, num_items, batch_size, output_layer_activation='linear', embedding_size=64,
                 max_history_length=64, dropout_rate=0.5, num_blocks=2, num_heads=1,
                 reuse_item_embeddings=False,
                 encode_output_embeddings=False,
                 sampled_target=None,
                 vanilla = False, #vanilla implementation; 
                                  #at the training time we calculate one positive and one negative per sequence element
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(not (vanilla and sampled_target), "only vanilla or sampled targetd strategy can be used at once")
        self.output_layer_activation = output_layer_activation
        self.embedding_size = embedding_size
        self.max_history_length = max_history_length
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_items = num_items
        self.batch_size = batch_size
        self.sampled_target = sampled_target
        self.reuse_item_embeddings=reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.vanilla = vanilla

        self.positions = tf.constant(tf.tile(tf.expand_dims(tf.range(self.max_history_length), 0), [self.batch_size, 1]))

        self.item_embeddings_layer = layers.Embedding(self.num_items + 1, output_dim=self.embedding_size,
                                                       dtype='float32')
        self.postion_embedding_layer = layers.Embedding(self.max_history_length,
                                                        self.embedding_size,
                                                         dtype='float32')

        self.embedding_dropout = layers.Dropout(self.dropout_rate)

        self.attention_blocks = []
        for i in range(self.num_blocks):
            block_layers = {
                "first_norm": layers.LayerNormalization(),
                "attention_layers": {
                    "query_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "key_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "val_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "dropout": layers.Dropout(self.dropout_rate),
                },
                "second_norm": layers.LayerNormalization(),
                "dense1": layers.Dense(self.embedding_size, activation='relu'),
                "dense2": layers.Dense(self.embedding_size),
                "dropout": layers.Dropout(self.dropout_rate)
            }
            self.attention_blocks.append(block_layers)
        self.output_activation = activations.get(self.output_layer_activation)
        self.seq_norm = layers.LayerNormalization()
        self.all_items = tf.range(0, self.num_items)
        if not self.reuse_item_embeddings:
            self.output_item_embeddings = layers.Embedding(self.num_items, self.embedding_size)

        if self.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.embedding_size, activation='gelu')


    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x = multihead_attention(queries, keys, self.num_heads, self.attention_blocks[i]["attention_layers"],
                                     causality=True)
        x =x + queries
        x = self.attention_blocks[i]["second_norm"](x)
        residual = x
        x = self.attention_blocks[i]["dense1"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x = self.attention_blocks[i]["dense2"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x += residual
        x *= mask
        return x

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb = self.get_seq_embedding(input_ids, training)

        if self.vanilla or (self.sampled_target is not None):
            target_ids = inputs[1]
        else:
            target_ids = self.all_items
        target_embeddings = self.get_target_embeddings(target_ids)
        if self.vanilla:
            positive_embeddings = target_embeddings[:,:,0,:]
            negative_embeddings = target_embeddings[:,:,1,:]
            positive_results = tf.reduce_sum(seq_emb*positive_embeddings, axis=-1)
            negative_results = tf.reduce_sum(seq_emb*negative_embeddings, axis=-1)
            output = tf.stack([positive_results, negative_results], axis=-1)
        elif self.sampled_target:
            seq_emb = seq_emb[:, -1, :]
            output = tf.einsum("ij,ikj ->ik", seq_emb, target_embeddings)
        else:
            seq_emb = seq_emb[:, -1, :]
            output = seq_emb @ tf.transpose(target_embeddings)
        output = self.output_activation(output)
        return output
    
    def score_all_items(self, inputs):
        input_ids = inputs[0]
        seq_emb = self.get_seq_embedding(input_ids)
        seq_emb = seq_emb[:, -1, :]
        target_ids = self.all_items
        target_embeddings = self.get_target_embeddings(target_ids)
        output = seq_emb @ tf.transpose(target_embeddings)
        output = self.output_activation(output)
        return output

    def get_target_embeddings(self, target_ids):
        if self.reuse_item_embeddings:
            target_embeddings = self.item_embeddings_layer(target_ids)
        else:
            target_embeddings = self.output_item_embeddings(target_ids)
        if self.encode_output_embeddings:
            target_embeddings = self.output_item_embeddings_encode(target_embeddings)
        return target_embeddings

    def get_seq_embedding(self, input_ids, training=None):
        seq = self.item_embeddings_layer(input_ids)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.num_items), dtype=tf.float32), -1)
        pos_embeddings = self.postion_embedding_layer(self.positions)
        seq += pos_embeddings
        seq = self.embedding_dropout(seq)
        seq *= mask
        for i in range(self.num_blocks):
            seq = self.block(seq, mask, i)
        seq_emb = self.seq_norm(seq)
        return seq_emb


