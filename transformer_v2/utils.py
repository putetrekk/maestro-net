import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
import re

def textPreprocess(input_text):
	def removeAccents(input_text):
		strange = 'ąćęłńóśżź'
		ascii_replacements = 'acelnoszz'
		translator = str.maketrans(strange, ascii_replacements)
		return input_text.translate(translator)

	#  def removeSpecial(input_text):
	#      special='[^A-Za-z0-9 ]+'
	#      return re.sub(special, '', input_text)

	def removeTriplicated(input_text):
		return re.compile(r'(.)\1{2,}', re.IGNORECASE).sub(r'\1', input_text)

	return removeTriplicated(removeAccents(input_text.lower()))


# Build tokenizer for both questions and answers. Tokenize, filter pad sentences
def tokenizeAndFilter(inputs, outputs, max_length):
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(inputs + outputs, target_vocab_size=2 ** 13)
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Define start and end token to indicate the start and end of a sentence
	START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
	# Vocabulary size plus start and end token
	vocab_size = tokenizer.vocab_size + 2

	tokenized_inputs, tokenized_outputs = [], []

	for (sentence1, sentence2) in zip(inputs, outputs):
		# tokenize sentence
		sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
		sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
		# check tokenized sentence max length
		if len(sentence1) <= max_length and len(sentence2) <= max_length:
			print(f'len sentence1: {len(sentence1)} - sentence2: {len(sentence2)}')
			tokenized_inputs.append(sentence1)
			tokenized_outputs.append(sentence2)

	# pad tokenized sentences
	tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
		tokenized_inputs, maxlen=max_length, padding='post')
	tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
		tokenized_outputs, maxlen=max_length, padding='post')

	return tokenizer, tokenized_inputs, tokenized_outputs, vocab_size


def createDataset(inputs, outputs, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((
		{
			'inputs': inputs,
			'dec_inputs': outputs[:, :-1]
		},
		{
			'outputs': outputs[:, 1:]
		},
	))
	# For tf.data.Dataset
	BUFFER_SIZE = 20000
	return dataset.cache().shuffle(BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def scaledDotProductAttention(query, key, value, mask):
	"""Calculate the attention weights. """
	matmul_qk = tf.matmul(query, key, transpose_b=True)

	# scale matmul_qk
	depth = tf.cast(tf.shape(key)[-1], tf.float32)
	logits = matmul_qk / tf.math.sqrt(depth)

	# add the mask to zero out padding tokens
	if mask is not None:
		logits += (mask * -1e9)

	# softmax is normalized on the last axis (seq_len_k)
	attention_weights = tf.nn.softmax(logits, axis=-1)

	output = tf.matmul(attention_weights, value)

	return output



def createPaddingMask(x):
	mask = tf.cast(tf.math.equal(x, 0), tf.float32)
	# (batch_size, 1, 1, sequence length)
	return mask[:, tf.newaxis, tf.newaxis, :]


def createLookAheadMask(x):
	seq_len = tf.shape(x)[1]
	look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
	padding_mask = createPaddingMask(x)
	return tf.maximum(look_ahead_mask, padding_mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def split_test_train(x, y, validation_split=0.1):
	v_num = int(len(x) * validation_split)
	train_x, train_y = x[v_num:], y[v_num:]
	validation_x, validation_y = x[0:v_num], y[0:v_num]

	return train_x, train_y, validation_x, validation_y