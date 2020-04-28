from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import numpy as np

from midiparser import MidiParser


def create_tokenizer():
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(MidiParser.vocabulary().keys())
	return tokenizer


def create_sequences(data: list, sequence_length=100, validation_split: float = 0.2):
	tokenizer = create_tokenizer()
	total_words = len(tokenizer.word_index) + 1

	# create input sequences using list of tokens
	sequences = []
	for line in data:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(sequence_length, len(token_list)):
			n_gram_sequence = token_list[i - sequence_length:i + 1]
			sequences.append(n_gram_sequence)

	# pad sequences
	sequences = np.array(pad_sequences(sequences, maxlen=sequence_length + 1, padding='pre'))

	# create predictors and label
	predictors, label = sequences[:, :-1], sequences[:, -1]
	label = ku.to_categorical(label, num_classes=total_words)

	v_num = int(len(predictors) * validation_split)

	train_x, train_y = predictors[v_num:], label[v_num:]
	validation_x, validation_y = predictors[0:v_num], label[0:v_num]

	return train_x, train_y, validation_x, validation_y

def split_input_output(all_data: list, words_per_section = 25):
	inputs, outputs = [], []
	for data in all_data:
		sep = ' '
		sectioned_notes = []

		groups = data.split(sep)
		while len(groups):
			sectioned_notes.append(sep.join(groups[:words_per_section]))
			groups = groups[words_per_section:]

		for i in range(len(sectioned_notes) - 1):
			inputs.append(sectioned_notes[i])
			outputs.append(sectioned_notes[i + 1])

	return inputs, outputs
