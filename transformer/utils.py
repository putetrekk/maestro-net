import re

import numpy
import tensorflow as tf
import tensorflow_datasets as tfds

from transformer.TransformerTokenizer import TransformerTokenizer

def scaled_dot_product_attention(query, key, value, mask):
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



def create_padding_mask(x):
	mask = tf.cast(tf.math.equal(x, 0), tf.float32)
	# (batch_size, 1, 1, sequence length)
	return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
	seq_len = tf.shape(x)[1]
	look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
	padding_mask = create_padding_mask(x)
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


def get_accuracy_and_loss_functions(max_length):
	def accuracy(y_true, y_pred):
		# ensure labels have shape (batch_size, MAX_LENGTH - 1)
		y_true = tf.reshape(y_true, shape=(-1, max_length - 1))
		return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


	def loss_function(y_true, y_pred):
		y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

		loss = tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits=True, reduction='none')(y_true, y_pred)

		mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
		loss = tf.multiply(loss, mask)

		return tf.reduce_mean(loss)


	return accuracy, loss_function


def get_tokenizer(inputs, outputs):
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(inputs + outputs, target_vocab_size=2 ** 13)
	start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
	vocab_size = tokenizer.vocab_size + 2

	return tokenizer, start_token, end_token, vocab_size

def get_custom_tokenizer():
	tokenizer = TransformerTokenizer()
	start_token = tokenizer.start_token
	end_token = tokenizer.end_token
	vocab_size = tokenizer.vocab_size + 2

	return tokenizer, start_token, end_token, vocab_size


def get_tokenized_data(inputs, outputs, tokenizer, start_token, end_token, max_length):
	tokenized_inputs, tokenized_outputs = [], []
	for (sentence1, sentence2) in zip(inputs, outputs):
		# print(f'start_token = {start_token} +| {tokenizer.encode(sentence1)} |+ end_token = {end_token}')
		sentence1 = start_token + tokenizer.encode(sentence1) + end_token
		sentence2 = start_token + tokenizer.encode(sentence2) + end_token
		# check tokenized sentence max length
		if len(sentence1) <= max_length and len(sentence2) <= max_length:
			tokenized_inputs.append(sentence1)
			tokenized_outputs.append(sentence2)

	# pad tokenized sentences
	tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
		tokenized_inputs, maxlen=max_length, padding='post')
	tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
		tokenized_outputs, maxlen=max_length, padding='post')

	return tokenized_inputs, tokenized_outputs


def get_dataset_as_tenor_slices(tokenized_inputs, tokenized_outputs):
	dataset = tf.data.Dataset.from_tensor_slices((
		{
			'inputs': tokenized_inputs,
			'dec_inputs': tokenized_outputs[:, :-1]
		},
		{
			'outputs': tokenized_outputs[:, 1:]
		},
	))

	BATCH_SIZE = 256
	BUFFER_SIZE = 20000
	dataset = dataset.cache()
	dataset = dataset.shuffle(BUFFER_SIZE)
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
	return dataset

def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
	from transformer.utils import create_padding_mask, create_look_ahead_mask
	from transformer.encoder import encoder
	from transformer.decoder import decoder

	inputs = tf.keras.Input(shape=(None,), name="inputs")
	dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")



	enc_padding_mask = tf.keras.layers.Lambda(
		create_padding_mask, output_shape=(1, 1, None),
		name='enc_padding_mask')(inputs)
	# mask the future tokens for decoder inputs at the 1st attention block
	look_ahead_mask = tf.keras.layers.Lambda(
		create_look_ahead_mask,
		output_shape=(1, None, None),
		name='look_ahead_mask')(dec_inputs)
	# mask the encoder outputs for the 2nd attention block
	dec_padding_mask = tf.keras.layers.Lambda(
		create_padding_mask, output_shape=(1, 1, None),
		name='dec_padding_mask')(inputs)

	enc_outputs = encoder(
		vocab_size=vocab_size,
		num_layers=num_layers,
		units=units,
		d_model=d_model,
		num_heads=num_heads,
		dropout=dropout,
	)(inputs=[inputs, enc_padding_mask])

	dec_outputs = decoder(
		vocab_size=vocab_size,
		num_layers=num_layers,
		units=units,
		d_model=d_model,
		num_heads=num_heads,
		dropout=dropout,
	)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

	outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

	return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def get_eval_function(tokenizer, model, start_token, end_token, max_length):
	def evaluate(sentence):
		# print("Sentence")
		# print(f'Encoded Sentence: {tokenizer.encode(sentence)}')
		sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

		output = tf.expand_dims(start_token, 0)

		outputs = []
		for i in range(max_length):
			#print(f'---------- generating word {i}')
			predictions = model(inputs=[sentence, output], training=False)


			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]
			best_k = tf.math.top_k(predictions, k=3, sorted=False)
			# print(best_k)
			# print(f'best_k={best_k}')
			# print(f'best_k[0]={best_k[0]}')
			# print(f'best_k[1]={best_k[1]}')
			# print(f'best_k[1][0][0]={best_k[1][0][0]}')
			indexes = best_k[1][0][0]
			values = best_k[0][0][0]
			values = [v.numpy() for v in values]

			max_prob = sum(values)
			probabilities = [v/max_prob for v in values]
			predicted_id = numpy.random.choice(indexes, p=probabilities)

			outputs.append(predicted_id)

		return outputs

	return evaluate

def get_predict_function(tokenizer, evaluate):
	def predict(sentence):
		prediction = evaluate(sentence)
		predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
		# cleaned_predicted_sentence = re.sub(r'([0-9])([pwe])', r'\1 \2', predicted_sentence)

		return predicted_sentence
	return predict


def split_test_train(x, y, validation_split=0.1):
	v_num = int(len(x) * validation_split)

	train_x, train_y = x[v_num:], y[v_num:]
	validation_x, validation_y = x[0:v_num], y[0:v_num]

	return train_x, train_y, validation_x, validation_y
