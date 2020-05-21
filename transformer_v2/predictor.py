import pickle

import numpy
import tensorflow as tf


def get_note_id(predictions):
	predictions = predictions[:, -1:, :]
	best_k = tf.math.top_k(predictions, k=3, sorted=False)
	indexes = best_k[1][0][0]
	values = best_k[0][0][0]
	values = [v.numpy() for v in values]
	probabilities = [value / sum(values) for value in values]
	predicted_id = numpy.random.choice(indexes, p=probabilities)

	return predicted_id



class Predictor():
	def __init__(self, tokenizer=False, max_length=80, tokenizer_pickle_path=''):
		if tokenizer:
			self.tokenizer = tokenizer
		else:
			with open(tokenizer_pickle_path, 'rb') as handle:
				self.tokenizer = pickle.load(handle)

		self.start_token = [self.tokenizer.vocab_size]
		self.end_token = [self.tokenizer.vocab_size + 1]
		self.vocab_size = self.tokenizer.vocab_size + 2
		self.max_length = max_length

	# def evaluate(self, sentence, model):
	# 	sentence = tf.expand_dims(
	# 		self.start_token + self.tokenizer.encode(sentence) + self.end_token, axis=0)
	# 	output = tf.expand_dims(self.start_token, 0)
	#
	# 	for i in range(self.max_length):
	# 		predictions = model(inputs=[sentence, output], training=False)
	# 		# select the last word from the seq_len dimension
	# 		predictions = predictions[:, -1:, :]
	# 		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
	# 		# return the result if the predicted_id is equal to the end token
	# 		if tf.equal(predicted_id, self.end_token[0]):
	# 			break
	# 		# concatenated the predicted_id to the output which is given to the decoder
	# 		# as its input.
	# 		output = tf.concat([output, predicted_id], axis=-1)
	#
	# 	return tf.squeeze(output, axis=0)

	def evaluate(self, sentence, model):
		sentence = tf.expand_dims(
			self.start_token + self.tokenizer.encode(sentence) + self.end_token, axis=0)
		output = tf.expand_dims(self.start_token, 0)
		outputs = []
		for i in range(self.max_length):
			predictions = model(inputs=[sentence, output], training=False)
			# select the last word from the seq_len dimension
			predicted_id = get_note_id(predictions)
			if tf.equal(predicted_id, self.end_token[0]):
				break
			# concatenated the predicted_id to the output which is given to the decoder
			# as its input.
			outputs.append(predicted_id)

		return outputs

	def predict(self, sentence, model):
		prediction = self.evaluate(sentence, model)
		predicted_sentence = self.tokenizer.decode(
			[i for i in prediction if i < self.tokenizer.vocab_size])
		return predicted_sentence.lstrip()

	def generate_work(self, model):
		#  Initial Note(s)
		initial_notes = "wait50 wait50 wait29 v10 p55"
		notes = initial_notes
		music = initial_notes
		for _ in range(10):
			notes = self.predict(notes, model)
			music += notes + ' '

		return music


