from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import numpy as np

from midiparser import MidiParser


def create_tokenizer():
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(MidiParser.vocabulary().keys())
	return tokenizer


def create_sequences(data: list, sequence_length=100):
	tokenizer = create_tokenizer()
	total_words = len(tokenizer.word_index)

	# create input sequences using list of tokens
	sequences = []
	for line in data:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(sequence_length, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			sequences.append(n_gram_sequence)

	# pad sequences
	sequences = np.array(pad_sequences(sequences, maxlen=sequence_length + 1, padding='pre'))

	# create predictors and label
	predictors, label = sequences[:, :-1], sequences[:, -1]
	label = ku.to_categorical(label, num_classes=total_words)

	return predictors, label
