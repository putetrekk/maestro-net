from tensorflow.keras.layers import SimpleRNN, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model


class LSTMModel(Model):
    def __init__(self, total_words, sequence_length):
        super(LSTMModel, self).__init__()

        model = self
        model.add(Embedding(total_words, 10, input_length=sequence_length))  # , input_length=max_sequence_len-1))
        model.add(LSTM(150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=True))
        model.add(SimpleRNN(100))
        model.add(Dense(total_words, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
