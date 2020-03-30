import os
from midiparser import MidiParser


# Cuts the sonata off on the first whitespace after note on given index
def cut_sonata(number_of_characters: int):
    # Currying
    def cut_procedure(notes: str):
        notes = notes.lstrip()  # Remove whitespace in front
        index_to_cut_from = notes[number_of_characters:].find(" ")
        return notes[:number_of_characters + index_to_cut_from]

    return cut_procedure


def prepare_notes(midi_parser: MidiParser, midi_folder: str, number_of_sonatas: int, characters_per_sonata: int):
    sonata_files = [f for f in os.listdir(midi_folder) if f.endswith('.mid')][:number_of_sonatas]
    notes = list(map(lambda sonata_file: midi_parser.read_music(sonata_file), sonata_files))
    if characters_per_sonata > 0:
        notes = list(map(cut_sonata(characters_per_sonata), notes))

    return notes
