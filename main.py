from music_generator import gen_music
from preprocessing import create_sequences, create_tokenizer
from midiparser import MidiParser
import csv, datetime

from models import *

if __name__ == '__main__':
    midi_folder = 'midi/'
    result_length_characters = 200
    dataset_size = 10000
    sequence_length = 50

    epochs = 500
    epoch_batch_size = 1
    save_history = True

    parser = MidiParser(midi_folder)
    data = parser.get_data(dataset_size)
    train_x, train_y = create_sequences(data, sequence_length)

    model = LSTMModel(sequence_length, len(create_tokenizer().word_index))

    # model.load_weights('weights.h5')

    recorded_stats = []
    for batch in range(epochs // epoch_batch_size):
        # train
        epoch_start = batch * epoch_batch_size
        epoch_end = epoch_start + epoch_batch_size
        history = model.fit(train_x, train_y, verbose=1, initial_epoch=epoch_start, epochs=epoch_end)

        # Record metrics
        metrics = [list(range(epoch_start, epoch_end))] + list(history.history.values())  # [epoch, loss, accuracy, ...]
        recorded_stats += zip(*metrics)

        # generate a small sample every batch
        music = gen_music(model, 200)
        parser.save_music(music, f'training_batch{batch}.mid')

    # model.save_weights(f'weights_s{sequence_length}_d{dataset_size}_e{epochs}_model_{model.Name}.h5')

    # save metrics history as csv
    if save_history:
        filename = f'history_{model.Name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        with open(filename, 'w', newline="\n") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            headers = ['epoch'] + model.metrics_names
            writer.writerow(headers)
            for line in recorded_stats:
                writer.writerow(line)

    # generate some sample music! :D
    music = gen_music(model, result_length_characters, n=2)
    parser.save_music(music, f'generated_length{result_length_characters}')
