from tensorflow.keras.layers import SimpleRNN, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

import note_parser


def generate_model(notes, sequence_length):
	predictors, label, total_words = note_parser.parse(notes, sequence_length)
	model = create_model(predictors, label, sequence_length, total_words)

	return model


def create_model(predictors, label, sequence_length, total_words):
	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=sequence_length - 1))#, input_length=max_sequence_len-1))
	model.add(LSTM(150, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(100, return_sequences=True))
	model.add(SimpleRNN(100))
	model.add(Dense(total_words, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
	model.fit(predictors, label, epochs=100, verbose=1, callbacks=[early_stop])
	model.save_weights("weights_n250.h5")
	return model
