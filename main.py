import pathlib
from maestroutil import *
from music_generator import gen_music
from preprocessing import create_sequences, create_tokenizer
from midiparser import MidiParser
import csv, datetime

from models import *


def save_weights(model, output_dir:str):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_weights(output_dir + f'weights_s{sequence_length}_d{dataset_words}_e{epochs}_b{batch}_model_{model.Name}.h5')


def load_data(filename: str, take_num: int = 1000):
    with open(filename, 'r') as file:
        return file.read().split('\n')[0:take_num]


if __name__ == '__main__':
    midi_dir = 'midi/'
    output_dir = 'output/' + prefix_timestamp('/')
    result_length_characters = 200
    dataset_words = 10000  # when loading from midi files, the number of words to take
    dataset_songs = 555  # when loading from preparsed text-file, the number of songs to take
    sequence_length = 50

    epochs = 50
    epoch_batch_size = 5
    save_history = True

    parser = MidiParser(midi_dir, output_dir)
    # data = parser.get_data(dataset_size)  # load from midi files
    data = load_data('data/scarlatti_k1_555.txt', dataset_songs)  # load from preparsed text-file
    train_x, train_y, validation_x, validation_y = create_sequences(data, sequence_length)

    model = TFAttentionModel(sequence_length, len(parser.vocabulary()) + 1)

    # model.load_weights('output/20200425_144518/weights_s100_d10000_e5_b0_model_LSTM.h5')

    recorded_stats = []
    record_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    for batch in range(epochs // epoch_batch_size):
        # train
        epoch_start = batch * epoch_batch_size
        epoch_end = epoch_start + epoch_batch_size
        history = model.fit(
            train_x,
            train_y,
            validation_data=(validation_x, validation_y),
            verbose=1,
            initial_epoch=epoch_start,
            epochs=epoch_end)

        # Save weights
        save_weights(model, output_dir)

        # Record metrics
        metrics = history.history
        metrics = [list(range(epoch_start, epoch_end))] + [metrics[key] for key in metrics if key in record_metrics]
        recorded_stats += zip(*metrics)

        # generate a small sample every batch
        music = gen_music(model, 200, n=3)
        parser.save_music(music, f'training_batch{batch}.mid')

    # save metrics history as csv
    if save_history:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f'history_{model.Name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(output_dir + filename, 'w', newline="\n") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            headers = ['epoch'] + record_metrics
            writer.writerow(headers)
            for line in recorded_stats:
                writer.writerow(line)

    # generate some sample music! :D
    music = gen_music(model, result_length_characters, n=3)
    parser.save_music(music, f'generated_length{result_length_characters}.mid')
