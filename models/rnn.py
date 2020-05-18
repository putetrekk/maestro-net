from tensorflow.keras.layers import Embedding, Dense, Input, SimpleRNN
from tensorflow.keras.models import Model


class RNNModel(Model):
    Name = 'Attention Model v1'

    def __init__(self, sequence_length, total_words, *args, **kwargs):
        input_layer = Input(shape=[sequence_length], dtype='int32')

        # get the embedding layer
        embedded = Embedding(
            input_dim=total_words,
            output_dim=5,
            input_length=sequence_length,
            trainable=True
        )(input_layer)

        rnn1 = SimpleRNN(
            150,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            dropout=0.2,
            recurrent_dropout=0.0,
            return_sequences=True,
            go_backwards=False,
            stateful=False,
            unroll=True,
        )(embedded)

        rnn2 = SimpleRNN(
            100,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            dropout=0.2,
            unroll=True,
        )(rnn1)

        probabilities = Dense(total_words, activation='softmax')(rnn2)

        super().__init__(inputs=input_layer, outputs=probabilities, *args, **kwargs)
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

        print(self.summary())
