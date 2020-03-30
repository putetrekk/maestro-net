import note_parser
from midiparser import MidiParser, generate_filename
from model_generator import generate_model
from note_preprocessor import prepare_notes

if __name__ == '__main__':
    midi_folder = 'midi/'
    num_sonatas_train = 20
    characters_per_sonata = 300
    result_length_characters = 200

    parser = MidiParser(midi_folder)
    notes = prepare_notes(parser, midi_folder, num_sonatas_train, characters_per_sonata)
    model, max_sequence_len = generate_model(notes)

    #  Initial Note(s)
    initial_wait_note = "wait4"
    result = note_parser.generate_text(model, initial_wait_note, result_length_characters, max_sequence_len)

    filename = generate_filename(num_sonatas_train, characters_per_sonata)
    print(result)
    parser.save_music(result, filename)
