from music_generator import gen_music
from preprocessing import create_sequences, create_tokenizer
from midiparser import MidiParser

from models import *

if __name__ == '__main__':
    midi_folder = 'midi/'
    result_length_characters = 200
    dataset_size = 10000
    sequence_length = 50

    epochs = 500
    epoch_batch_size = 1

    parser = MidiParser(midi_folder)
    data = parser.get_data(dataset_size)
    train_x, train_y = create_sequences(data, sequence_length)

    model = LSTMModel(sequence_length, len(create_tokenizer().word_index))

    # model.load_weights('weights.h5')

    for batch in range(epochs // epoch_batch_size):
        # train
        epoch_start = batch * epoch_batch_size
        epoch_end = epoch_start + epoch_batch_size
        model.fit(train_x, train_y, verbose=1, initial_epoch=epoch_start, epochs=epoch_end)

        # generate a small sample every batch
        music = gen_music(model, 200)
        parser.save_music(music, f'training_batch{batch}.mid')

    # model.save_weights(f'weights_s{sequence_length}_d{dataset_size}_e{epochs}.h5')

    # generate some sample music! :D
    music = gen_music(model, result_length_characters, n=2)
    parser.save_music(music, f'generated_length{result_length_characters}')
