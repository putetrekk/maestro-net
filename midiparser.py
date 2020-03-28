from datetime import datetime
from os import path

import py_midicsv


class MidiParser:
    def __init__(self, folder: str):
        self.folder: str = folder

    def read_music(self, file: str):
        file_path = path.join(self.folder, file)

        csv = py_midicsv.midi_to_csv(file_path)

        print(csv)

        return csv
