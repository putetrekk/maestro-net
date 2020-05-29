import datetime
import pickle
import tensorflow as tf
from midiparser import MidiParser
from transformer_v2.predictor import Predictor
from transformer_v2.transformer import transformer
from transformer_v2.utils import split_input_output, tokenize_and_filter, CustomSchedule


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
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

# clear backend
tf.keras.backend.clear_session()
epochs = 1_000
epoch_batch_size = 5
dataset_works = 1000
WORDS_PER_SECTION = MAX_LENGTH-2
output_dir = f'output/{prefix_timestamp("/")}'
midi_dir = '../midi/'
parser = MidiParser(midi_dir, output_dir)

data_chopin = load_data('../data/chopin.txt', dataset_works)  # load from preparsed text-file
data_bach = load_data('../data/bach.txt', dataset_works)  # load from preparsed text-file
data_scarlatti = load_data('../data/scarlatti.txt', dataset_works)


print(f'chopin_size={len(data_chopin)}')
print(f'bach_size={len(data_bach)}')
print(f'scarlatti_size={len(data_scarlatti)}')

inputs, outputs = split_input_output(data_chopin, WORDS_PER_SECTION)

tokenizer, questions, answers, vocab_size, = tokenize_and_filter(inputs, outputs, max_length=MAX_LENGTH)


print(f'questions: {questions}')
print('Vocab size: {}'.format(vocab_size))
print('Number of samples: {}'.format(len(questions)))


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


predictor = Predictor(tokenizer=tokenizer, max_length=MAX_LENGTH)
model.load_weights('output/20200521_155132/weights_34.h5')