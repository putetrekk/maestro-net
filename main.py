import note_parser
from midiparser import MidiParser
from model_generator import generate_model

if __name__ == '__main__':
    midi_folder = 'midi/'
    result_length_characters = 200
    dataset_size = 1000
    sequence_length = 5

    parser = MidiParser(midi_folder)
    notes = parser.prepare_notes(dataset_size)

    model = generate_model(notes, sequence_length)

    #  Initial Note(s)
    initial_wait_note = "wait4"
    result = note_parser.generate_text(model, initial_wait_note, result_length_characters, sequence_length, n=2)

    filename = f'generated_{dataset_size}.mid'
    print(result)
    parser.save_music(result, filename)
