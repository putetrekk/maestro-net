from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Permute, Lambda, Multiply, Attention
from tensorflow.keras.layers import TimeDistributed, Flatten, Activation, RepeatVector
from tensorflow.keras.backend import sum
from tensorflow.keras.models import Model


class AttentionModel(Model):
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

        activations = LSTM(total_words, return_sequences=True)(embedded)

        # compute importance for each step
        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(total_words)(attention)
        attention = Permute([2, 1])(attention)

        # apply the attention
        sent_representation = Multiply()([activations, attention])
        sent_representation = Lambda(lambda xin: sum(xin, axis=1))(sent_representation)

        probabilities = Dense(total_words, activation='softmax')(sent_representation)

        super().__init__(inputs=input_layer, outputs=probabilities, *args, **kwargs)
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

        print(self.summary())
