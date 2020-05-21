import csv

import tensorflow as tf

# Maximum sentence length
from midiparser import MidiParser
from transformer_v2.predictor import Predictor
from transformer_v2.transformer import transformer
from transformer_v2.utils import createDataset, CustomSchedule, split_test_train, split_input_output, tokenize_data, \
	tokenize_and_filter
import os
import datetime

strategy = tf.distribute.get_strategy()

# For tf.data.Dataset
BATCH_SIZE = int(64 * strategy.num_replicas_in_sync)
BUFFER_SIZE = 20000

MAX_LENGTH = 80
# For Transformer
NUM_LAYERS = 2  # 6
D_MODEL = 256  # 512
NUM_HEADS = 8
UNITS = 512  # 2048
DROPOUT = 0.1

EPOCHS = 100

tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE


def loss_function(y_true, y_pred):
	y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

	loss = tf.keras.losses.SparseCategoricalCrossentropy(
		from_logits=True, reduction='none')(y_true, y_pred)

	mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
	loss = tf.multiply(loss, mask)

	return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
	# ensure labels have shape (batch_size, MAX_LENGTH - 1)
	y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
	return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def load_data(filename: str, take_num: int = 1000):
	with open(filename, 'r') as file:
		return file.read().split('\n')[0:take_num]


def prefix_timestamp(filename: str):
	now = datetime.datetime.now()
	timestamp = now.strftime('%Y%m%d_%H%M%S')
	if filename.startswith('/'):
		return timestamp + filename
	else:
		return timestamp + '_' + filename


if __name__ == '__main__':
	# clear backend
	print(f'Batch Size = {BATCH_SIZE}')
	tf.keras.backend.clear_session()
	epochs = 1_000
	epoch_batch_size = 2
	dataset_works = 1000
	WORDS_PER_SECTION = MAX_LENGTH-2
	output_dir = f'output/{prefix_timestamp("/")}'
	midi_dir = '../midi/'
	parser = MidiParser(midi_dir, output_dir)

	data_chopin = load_data('../data/chopin.txt', dataset_works)  # load from preparsed text-file
	data_bach = load_data('../data/bach.txt', dataset_works)  # load from preparsed text-file
	data_scarlatti = load_data('../data/scarlatti.txt', dataset_works)
	inputs, outputs = split_input_output(data_bach + data_chopin + data_scarlatti, WORDS_PER_SECTION)

	print(f'chopin_size={len(data_chopin)}')
	print(f'bach_size={len(data_bach)}')
	print(f'scarlatti_size={len(data_scarlatti)}')

	# tokenizer, questions, answers, vocab_size, = tokenize_and_filter(inputs, outputs, max_length=MAX_LENGTH)
	tokenizer, questions, answers, vocab_size,  = tokenize_data(inputs, outputs, max_length=MAX_LENGTH)
	train_x, train_y, valid_x, valid_y = split_test_train(questions, answers, 0.1)

	print(f'questions: {questions}')
	print('Vocab size: {}'.format(vocab_size))
	print('Number of samples: {}'.format(len(questions)))

	dataset = createDataset(train_x, train_y, BATCH_SIZE)
	validation_dataset = createDataset(valid_x, valid_y, BATCH_SIZE)

	learning_rate = CustomSchedule(D_MODEL)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

	with strategy.scope():
		model = transformer(
			vocab_size=vocab_size,
			num_layers=NUM_LAYERS,
			units=UNITS,
			d_model=D_MODEL,
			num_heads=NUM_HEADS,
			dropout=DROPOUT)
		model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

	recorded_stats = []
	record_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
	print(model.summary())
	print('\n')
	print("Train model")
	logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

	predictor = Predictor(tokenizer=tokenizer, max_length=MAX_LENGTH)
	for batch in range(epochs // epoch_batch_size):



		# train
		epoch_start = batch * epoch_batch_size
		epoch_end = epoch_start + epoch_batch_size
		history = model.fit(
			dataset,
			validation_data=validation_dataset,
			verbose=1,
			initial_epoch=epoch_start,
			epochs=epoch_end,
			callbacks=[tensorboard_callback])

		generated_music = predictor.generate_work(model)
		music_name = f'transformer_train_batch{epoch_end}.midi'
		parser.save_music(generated_music, music_name)

		model.save_weights(f'{output_dir}weights_{epoch_end}.h5')
		# Record metrics
		metrics = history.history
		metrics = [list(range(epoch_start, epoch_end))] + [metrics[key] for key in metrics if key in record_metrics]
		recorded_stats += zip(*metrics)

		filename = f'{output_dir}history_transformer_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
		with open(filename, 'w', newline="\n") as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			headers = ['epoch'] + record_metrics
			writer.writerow(headers)
			for line in recorded_stats:
				writer.writerow(line)

	print("\nModel weights saved!")
