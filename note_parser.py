from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import numpy as np

tokenizer = Tokenizer()


def parse(data: list, sequence_length=100):
	#  tokenizing
	tokenizer.fit_on_texts(data)
	total_words = len(tokenizer.word_index) + 1

	# create input sequences using list of tokens
	input_sequences = []
	for line in data:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(sequence_length, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)

	# pad sequences
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=sequence_length, padding='pre'))

	# create predictors and label
	predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
	label = ku.to_categorical(label, num_classes=total_words)

	return predictors, label, total_words


def generate_text(model, seed_text, next_words, sequence_length, n=1):
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=sequence_length - 1, padding='pre')

		predictions = model.predict(token_list, verbose=0)[0]

		top_predictions = np.argsort(predictions)[-n:]
		top_probabilities = np.take(predictions, np.argsort(predictions)[-n:])
		top_probabilities /= top_probabilities.sum()
		next_note_index = np.random.choice(top_predictions, p=top_probabilities)

		seed_text += " " + tokenizer.index_word[next_note_index]
	return seed_text
