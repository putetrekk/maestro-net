import csv
import datetime

from midiparser import MidiParser
from preprocessing import split_input_output
from transformer.utils import get_tokenizer, get_eval_function, get_predict_function, transformer
from transformer.utils import get_tokenized_data
from transformer.utils import CustomSchedule
from transformer.utils import get_accuracy_and_loss_functions
from transformer.utils import get_dataset_as_tenor_slices
import tensorflow as tf

if __name__ == '__main__':
	train_data_folder = 'midi/scarlatti/train'
	validation_data_folder = 'midi/scarlatti/validation'
	dataset_size = float('inf')
	save_history = True
	EPOCHS = 10
	EPOCHS_BATCH_SIZE = 5
	# Hyper-parameters

	NUM_LAYERS = 2
	D_MODEL = 256
	NUM_HEADS = 8
	UNITS = 512
	DROPOUT = 0.2
	WORDS_PER_SECTION = 20                      # The notes in a sentence
	MAX_LENGTH = 80                             # Biggest notes: wait10,

	parser = MidiParser(train_data_folder)
	validation_parser = MidiParser(validation_data_folder)
	data = parser.get_data(dataset_size)
	validation_data = validation_parser.get_data(dataset_size)

	inputs, outputs = split_input_output(data, WORDS_PER_SECTION)
	validation_inputs, validation_outputs = split_input_output(validation_data, WORDS_PER_SECTION)
	tokenizer, start_token, end_token, vocab_size = get_tokenizer(inputs + validation_inputs, outputs + validation_outputs)
	tokenized_inputs, tokenized_outputs = get_tokenized_data(inputs, outputs, tokenizer, start_token, end_token, MAX_LENGTH)
	dataset = get_dataset_as_tenor_slices(tokenized_inputs, tokenized_outputs)

	print(f'Input length: {len(inputs)}')
	print(f'dataset: {dataset}')
	tokenized_validation_inputs, tokenized_validation_outputs = get_tokenized_data(validation_inputs, validation_outputs, tokenizer, start_token, end_token, MAX_LENGTH)
	validation_dataset = get_dataset_as_tenor_slices(tokenized_validation_inputs, tokenized_validation_outputs)

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
	evaluate_function = get_eval_function(tokenizer, model, start_token, end_token, MAX_LENGTH)
	predict = get_predict_function(tokenizer, evaluate_function)

	model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
	recorded_stats = []
	record_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
	for batch in range(EPOCHS // EPOCHS_BATCH_SIZE):
		# train
		epoch_start = batch * EPOCHS_BATCH_SIZE
		epoch_end = epoch_start + EPOCHS_BATCH_SIZE
		history = model.fit(dataset, validation_data=validation_dataset, initial_epoch=epoch_start,epochs=epoch_end, verbose=1)

		# Record metrics
		metrics = history.history
		metrics = [list(range(epoch_start, epoch_end))] + [metrics[key] for key in metrics if key in record_metrics]
		recorded_stats += zip(*metrics)

		print("Generate new song....")
		# Generate a new song
		initial_notes = 'p67 wait4 endp67'
		notes = initial_notes
		music = ''
		for _ in range(50):
			notes = predict(notes)
			music += notes + ' '

		print("Saving music...")
		parser.save_music(music, f'training_batch{batch}.mid')
		model.save_weights(f'weights/transformer/0_d{dataset_size}_e{epoch_end}of_{EPOCHS}_batch{batch}.h5')
	# model.save_weights(f'weights_s{sequence_length}_d{dataset_size}_e{epochs}_model_{model.Name}.h5')

	# save metrics history as csv
	if save_history:
		filename = f'history_transformer_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
		with open(filename, 'w', newline="\n") as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			headers = ['epoch'] + record_metrics
			writer.writerow(headers)
			for line in recorded_stats:
				writer.writerow(line)






	# history = model.fit(dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1)
	# metrics = history.history
	# print(metrics)
	# # model.load_weights(f'weights_transformer_epochs{EPOCHS}_words_per_s{WORDS_PER_SECTION}.h5')
	# model.save_weights(f'weights/transformer/t_epochs{EPOCHS}_words_per_s{WORDS_PER_SECTION}.h5')
	# # Curried functions
	#
	#
	# parser.save_music(total_output, f'transform_result.mid')
