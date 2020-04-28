from tensorflow_core.python.keras.layers import Embedding, CuDNNLSTM, Dense, Dropout
from tensorflow_core.python.keras.models import Sequential


class CuDNNLSTMModel(Sequential):
    Name = 'CuDNNLSTM'

    def __init__(self, sequence_length, total_words):
        super().__init__()

        self.add(Embedding(input_dim=total_words, output_dim=10, input_length=sequence_length))
        self.add(CuDNNLSTM(150, return_sequences=True))
        self.add(Dropout(0.2))
        self.add(CuDNNLSTM(100))
        self.add(Dense(total_words, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.summary())
