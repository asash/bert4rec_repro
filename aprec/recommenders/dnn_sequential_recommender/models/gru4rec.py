import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel

#https://arxiv.org/abs/1511.06939
class GRU4Rec(SequentialRecsysModel):
    def __init__(self,
                 output_layer_activation='linear', embedding_size=64, max_history_len=64,
                 num_gru_layers=3, num_dense_layers=1, activation='relu'):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.num_gru_layers = num_gru_layers
        self.num_dense_layers = num_dense_layers
        self.activation = activation
        self.model_type = "GRU4rec"


    def get_model(self):
        input = layers.Input(shape=(self.max_history_length))
        x = layers.Embedding(self.num_items + 1, self.embedding_size, dtype='float32')(input)
        for i in range(self.num_gru_layers - 1):
            x = layers.GRU(self.embedding_size, activation=self.activation, return_sequences=True)(x)
        x = layers.GRU(self.embedding_size, activation=self.activation)(x)

        for i in range(self.num_dense_layers):
            x = layers.Dense(self.embedding_size, activation=self.activation)(x)
        output = layers.Dense(self.num_items, name="output", activation=self.output_layer_activation)(x)
        model = Model(inputs=[input], outputs=[output], name='GRU')
        return model
