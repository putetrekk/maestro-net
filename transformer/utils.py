import tensorflow as tf
import tensorflow_datasets as tfds

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


def get_tokenized_data(inputs, outputs, tokenizer, start_token, end_token, max_length):
	tokenized_inputs, tokenized_outputs = [], []
	MAX_LENGTH = 80
	for (sentence1, sentence2) in zip(inputs, outputs):

		sentence1 = start_token + tokenizer.encode(sentence1) + end_token
		sentence2 = start_token + tokenizer.encode(sentence2) + end_token
		# check tokenized sentence max length
		if len(sentence1) <= MAX_LENGTH and len(sentence2) <= max_length:
			tokenized_inputs.append(sentence1)
			tokenized_outputs.append(sentence2)

	# pad tokenized sentences
	tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
		tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
	tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
		tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

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

	BATCH_SIZE = 64
	BUFFER_SIZE = 20000
	dataset = dataset.cache()
	dataset = dataset.shuffle(BUFFER_SIZE)
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

	return dataset

