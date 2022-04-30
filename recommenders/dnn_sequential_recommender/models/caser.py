import tensorflow.keras.layers as layers
from  tensorflow.keras.models import Model
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel

#https://dl.acm.org/doi/abs/10.1145/3159652.3159656
class Caser(SequentialRecsysModel):
    def __init__(self,
                 output_layer_activation='linear', embedding_size=64, max_history_len=64,
                 n_vertical_filters=4, n_horizontal_filters=16,
                 dropout_ratio=0.5, activation='relu', requires_user_id = True,
                 user_extra_features=False,
                 user_features_attention_heads=4,
                 ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.n_vertical_filters = n_vertical_filters
        self.n_horizontal_filters = n_horizontal_filters
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.requires_user_id = requires_user_id
        self.model_type = "Caser"
        self.user_extra_features = user_extra_features
        self.user_features_attention_heads = user_features_attention_heads

    def get_model(self):
        input = layers.Input(shape=(self.max_history_length))
        model_inputs = [input]
        x = layers.Embedding(self.num_items + 1, self.embedding_size, dtype='float32')(input)
        x = layers.Reshape(target_shape=(self.max_history_length, self.embedding_size, 1))(x)
        vertical = layers.Convolution2D(self.n_vertical_filters, kernel_size=(self.max_history_length, 1),
                                        activation=self.activation)(x)
        vertical = layers.Flatten()(vertical)
        horizontals = []
        for i in range(self.max_history_length):
            horizontal_conv_size = i + 1
            horizontal_convolution = layers.Convolution2D(self.n_horizontal_filters,
                                                          kernel_size=(horizontal_conv_size,
                                                                       self.embedding_size), strides=(1, 1),
                                                          activation=self.activation)(x)
            pooled_convolution = layers.MaxPool2D(pool_size=(self.max_history_length - horizontal_conv_size + 1, 1)) \
                (horizontal_convolution)
            pooled_convolution = layers.Flatten()(pooled_convolution)
            horizontals.append(pooled_convolution)
        x = layers.Concatenate()([vertical] + horizontals)
        x = layers.Dropout(self.dropout_ratio)(x)
        x = layers.Dense(self.embedding_size, activation=self.activation)(x)

        if self.requires_user_id:
            user_id_input = layers.Input(shape=(1,))
            model_inputs.append(user_id_input)
            user_embedding = layers.Embedding(self.num_users, self.embedding_size, dtype='float32')(user_id_input)
            user_embedding = layers.Flatten()(user_embedding)
            x = layers.Concatenate()([x, user_embedding])

        if self.user_extra_features:
            user_features_input = layers.Input(shape=(self.max_user_features))
            model_inputs.append(user_features_input)
            user_features = layers.Embedding(self.user_feature_max_val + 1, self.embedding_size, dtype='float32')(user_features_input)
            user_features = layers.MultiHeadAttention(self.user_features_attention_heads, key_dim=self.embedding_size)(user_features, user_features)
            user_features = layers.Dense(self.embedding_size, activation=self.activation)(user_features)
            user_features = layers.MaxPool1D(self.max_user_features)(user_features)
            user_features = layers.Flatten()(user_features)
            x = layers.Concatenate()([x, user_features])



        output = layers.Dense(self.num_items, activation=self.output_layer_activation)(x)
        model = Model(model_inputs, outputs=output)
        return model
