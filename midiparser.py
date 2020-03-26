from datetime import datetime
from os import path

import pretty_midi


class MidiParser:
    def __init__(self, folder: str):
        self.folder: str = folder

    def readMusic(self, file: str):
        file_path = path.join(self.folder, file)
        # We'll load in the example.mid file distributed with pretty_midi
        pm = pretty_midi.PrettyMIDI(file_path)

        piano_roll = pm.get_piano_roll(fs=1)

        for i in piano_roll:
            print(i)

        return