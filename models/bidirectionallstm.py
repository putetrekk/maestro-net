from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential


class BiDirectionalLSTMModel(Sequential):
    Name = 'BiDi-LSTM'

    def __init__(self, sequence_length, total_words):
        super().__init__()

        self.add(Embedding(input_dim=total_words, output_dim=10, input_length=sequence_length))
        self.add(Bidirectional(LSTM(150, return_sequences=True)))
        self.add(Dropout(0.2))
        self.add(Bidirectional(LSTM(100)))
        self.add(Dense(total_words, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.summary())
