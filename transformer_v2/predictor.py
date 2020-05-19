import pickle
import tensorflow as tf

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

	def evaluate(self, sentence, model):
		sentence = tf.expand_dims(
			self.start_token + self.tokenizer.encode(sentence) + self.end_token, axis=0)
		output = tf.expand_dims(self.start_token, 0)

		for i in range(self.max_length):
			predictions = model(inputs=[sentence, output], training=False)
			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]
			predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
			# return the result if the predicted_id is equal to the end token
			if tf.equal(predicted_id, self.end_token[0]):
				break
			# concatenated the predicted_id to the output which is given to the decoder
			# as its input.
			output = tf.concat([output, predicted_id], axis=-1)

		return tf.squeeze(output, axis=0)

	def predict(self, sentence, model):
		prediction = self.evaluate(sentence, model)
		predicted_sentence = self.tokenizer.decode(
			[i for i in prediction if i < self.tokenizer.vocab_size])
		return predicted_sentence.lstrip()


	def generate_work(self, model):
		#  Initial Note(s)
		initial_notes = "wait10 wait1 p76 wait10 wait1 endp76 p78 wait10 wait1 endp78 p79 wait10 wait1 endp79 p81 wait10 wait1 " \
		        "endp81 p83 wait10 wait1 p71 wait10 wait1 endp83 endp71 p75 wait10 wait1 p71 wait2 endp75 wait9 endp71 " \
		        "p64 p76 wait10 wait1 endp64 p66 wait10 wait1 endp66 p67 wait10 wait1 endp67 p69 wait10 wait1 endp69 p71 " \
		        "wait10 wait1 p59 wait2 endp71 wait9 endp59 endp76 p63 p78 wait10 wait1 p59 wait2 endp63 wait9 endp59 " \
		        "endp78 p64 p79 wait10 wait1 endp79 p76 wait10 wait1 endp76 endp64 p66 p81 wait10 wait1 endp81 p78 wait10 " \
		        "wait1 endp78 endp66 p67 p83 wait10 wait1 endp83 p79 wait10 wait1 endp79 endp67 p69 p78 wait10 wait1 " \
		        "endp78 p76 wait10 wait1 endp76 endp69 p71 p75 wait10 wait10 wait2 endp75 endp71 p71 p83 wait10 wait10 " \
		        "wait2 endp83 endp71 p71 p84 wait5 endp84 p83 wait5 endp83 p84 wait5 endp84 p83 wait5 endp83 endp71 p71 " \
		        "p84 wait5 endp84 p83 wait5 endp83 p84 wait5 endp84 p83 wait5 endp71 p67 p71 wait10 wait1 endp83 p81 " \
		        "wait10 wait1 endp81 endp71 endp67 p67 p71 p79 wait10 wait1 endp79 p78 wait10 wait1 endp78 endp71 endp67 " \
		        "p67 p71 p76 wait10 wait1 endp76"
		notes = initial_notes
		music = initial_notes
		for _ in range(10):
			notes = self.predict(notes, model)
			music += notes + ' '

		return music


