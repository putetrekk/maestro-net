from tensorflow.keras.layers import Embedding, Dense, Input, GRU, Bidirectional
from tensorflow.keras.models import Model


class GRUModel(Model):
    Name = 'GRU Model'

    def __init__(self, sequence_length, total_words, *args, **kwargs):
        input_layer = Input(shape=[sequence_length], dtype='int32')

        embedding = Embedding(input_dim=total_words, output_dim=10, input_length=sequence_length)(input_layer)
        rnn1 = GRU(150, dropout=0.2, return_sequences=True)(embedding)
        rnn2 = GRU(100, dropout=0.2)(rnn1)
        probabilities = Dense(total_words, activation='softmax')(rnn2)

        super().__init__(inputs=input_layer, outputs=probabilities, *args, **kwargs)
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

        print(self.summary())