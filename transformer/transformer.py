import tensorflow as tf
import re


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
		sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)

		output = tf.expand_dims(start_token, 0)
		for i in range(max_length):
			predictions = model(inputs=[sentence, output], training=False)

			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]
			predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)


			# concatenated the predicted_id to the output which is given to the decoder
			# as its input.
			output = tf.concat([output, predicted_id], axis=-1)

		return tf.squeeze(output, axis=0)

	return evaluate

def get_predict_function(tokenizer, evaluate):
	def predict(sentence):
		prediction = evaluate(sentence)
		predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

		cleaned_predicted_sentence = re.sub(r'([0-9])([pwe])', r'\1 \2', predicted_sentence)
		print('Input: {}'.format(sentence))
		print('Output: {}'.format(cleaned_predicted_sentence))

		return cleaned_predicted_sentence
	return predict

def transform(data):
	from preprocessing import split_input_output
	from transformer.utils import get_tokenizer
	from transformer.utils import get_tokenized_data
	from transformer.utils import CustomSchedule
	from transformer.utils import get_accuracy_and_loss_functions
	from transformer.utils import get_dataset_as_tenor_slices

	# Hyper-parameters
	MAX_LENGTH = 80
	NUM_LAYERS = 2
	D_MODEL = 256
	NUM_HEADS = 8
	UNITS = 512
	DROPOUT = 0.1
	WORDS_PER_SECTION = 25 # The notes in a sentence
	inputs, outputs = split_input_output(data, WORDS_PER_SECTION)
	tokenizer, start_token, end_token, vocab_size = get_tokenizer(inputs, outputs)
	tokenized_inputs, tokenized_outputs = get_tokenized_data(inputs, outputs, tokenizer, start_token, end_token, MAX_LENGTH)
	dataset = get_dataset_as_tenor_slices(tokenized_inputs, tokenized_outputs)

	# Make the model
	model = transformer(
		vocab_size=vocab_size,
		num_layers=NUM_LAYERS,
		units=UNITS,
		d_model=D_MODEL,
		num_heads=NUM_HEADS,
		dropout=DROPOUT)

	learning_rate = CustomSchedule(D_MODEL)

	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	# Curried functions
	accuracy, loss = get_accuracy_and_loss_functions(MAX_LENGTH)
	model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

	EPOCHS = 30
	model.fit(dataset, epochs=EPOCHS)

	# Curried functions
	evaluate_function = get_eval_function(tokenizer, model, start_token, end_token, MAX_LENGTH)
	predict = get_predict_function(tokenizer, evaluate_function)

	# feed the model with its previous output
	sentence = 'wait7 p76 wait10 endp76'
	total_output = ''
	for _ in range(20):
		sentence = predict(sentence)
		total_output += sentence + ' '
		print('')

	print(total_output)

	return total_output
