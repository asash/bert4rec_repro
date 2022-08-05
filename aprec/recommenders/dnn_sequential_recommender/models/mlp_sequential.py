from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel

class SequentialMLPModel(SequentialRecsysModel):
    def __init__(self,
                 output_layer_activation='linear', embedding_size=64, max_history_len=64,
                 bottleneck_size=32, layers_before_bottleneck=(256, 128),
                        layers_after_bottleneck=(128, 256),
                        activation='relu',
                        dropout_rate = 0.5,
                    ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.model_type = "SequentialMLP"
        self.layers_before_bottleneck = layers_before_bottleneck
        self.layers_after_bottleneck = layers_after_bottleneck
        self.bottleneck_size = bottleneck_size
        self.activation = activation
        self.dropout_rate = dropout_rate

    def get_model(self):
        model = Sequential(name='MLP')
        model.add(layers.Embedding(self.num_items + 1, 32, input_length=self.max_history_length, dtype='float32'))
        model.add(layers.Flatten())
        for dim in self.layers_before_bottleneck:
            model.add(layers.Dense(dim, activation=self.activation))
        model.add(layers.Dense(self.bottleneck_size, activation=self.activation))
        model.add(layers.Dropout(self.dropout_rate))
        for dim in self.layers_after_bottleneck:
            model.add(layers.Dense(dim, activation=self.activation))
        model.add(layers.Dense(self.num_items, name="output", activation=self.output_layer_activation))
        return model