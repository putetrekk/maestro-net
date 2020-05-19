from typing import Dict, List
import os
import pathlib
import csv
import py_midicsv as pm
import re
import numpy as np
from maestroutil import *
from transposer import scarlatti_get_offset

class MidiParser:
    TIME_CONSTANT = 10
    MAX_WAIT = 50

    def __init__(self, midi_dir: str, output_dir: str):
        self.midi_dir: str = midi_dir
        self.output_dir: str = output_dir

    def get_data(self, size: int) -> List[str]:
        files = [f for f in os.listdir(self.midi_dir) if f.endswith(('.mid', '.MID', '.midi', '.MIDI'))]
        np.random.shuffle(files)

        data = []

        word_count = 0
        for file in files:
            if word_count >= size:
                break

            words = self.read_music(file, normalize=False).split(' ')[1:]
            if word_count + len(words) > size:
                words = words[0: size - word_count - len(words)]
            word_count += len(words)
            words = ' '.join(words)

            data.append(words)

        if not self.__validate_data(data):
            exit()

        return data

    def __validate_data(self, data: List[str]) -> bool:
        vocab = self.vocabulary()
        success = True
        for d in data:
            words = d.split(' ')
            for word in words:
                if word not in vocab:
                    success = False
                    print(f'Stumbled upon a word that\'s not in the vocabulary: {word}')
        return success

    def time_to_waits(self, time_t):
        waits = [self.MAX_WAIT] * (time_t // self.MAX_WAIT) + [time_t % self.MAX_WAIT]
        return str.join(' ', ['wait' + str(w) for w in waits if w > 0])

    def read_music(self, file: str, normalize: bool = False) -> str:
        file_path = os.path.join(self.midi_dir, file)
        csv_rows = pm.midi_to_csv(file_path)
        csv_rows = list(csv.reader(csv_rows, delimiter=',', skipinitialspace=True))
        csv_rows = self.normalize_tempo(csv_rows)

        note_offset = 0
        if normalize:
            note_offset = scarlatti_get_offset(filename=file)

        encoded = ''
        time_prev = 0
        for row in csv_rows:
            m_track, m_time, m_type = row[0], int(row[1]), row[2]

            wait_t = round((m_time - time_prev) / self.TIME_CONSTANT)
            time_prev = m_time

            if m_type == 'Note_on_c' or m_type == 'Note_off_c':
                channel, note, velocity = row[3], row[4], row[5]
                note = str(int(note) + note_offset)

                if wait_t != 0:
                    encoded += ' ' + self.time_to_waits(wait_t)

                if m_type == 'Note_off_c' or (m_type == 'Note_on_c' and velocity == '0'):
                    encoded += ' endp' + note
                elif m_type == 'Note_on_c':
                    encoded += ' p' + note

        return encoded

    def save_music(self, encoded: str, filename: str) -> None:
        words = encoded.split(' ')

        csv_rows = []

        csv_rows.append(['0', '0', 'Header', '0', '1', '384'])
        csv_rows.append(['1', '0', 'Start_track'])
        csv_rows.append(['1', '0', 'Tempo', '500000'])

        m_track, m_channel, m_time = '1', '0', 0
        for word in words:
            wait = re.match(r'wait([0-9]*)', word)
            if wait:
                m_time += int(wait.group(1)) * self.TIME_CONSTANT

            note = re.match(r'(?P<end>end)?p(?P<note>[0-9]*)', word)
            if note:
                m_type = 'Note_off_c' if note.group('end') else 'Note_on_c'
                m_note = note.group('note')
                m_velocity = '127'
                csv_rows.append([m_track, str(m_time), m_type, m_channel, m_note, m_velocity])

        csv_rows.append(['1', str(m_time + 5000), 'End_track'])
        csv_rows.append(['0', '0', 'End_of_file'])

        csv_rows = [str.join(', ', row) for row in csv_rows]
        midi_object = pm.csv_to_midi(csv_rows)

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        filepath = os.path.join(self.output_dir, prefix_timestamp(filename))
        with open(filepath, 'wb') as file:
            midi_writer = pm.FileWriter(file)
            midi_writer.write(midi_object)

    @staticmethod
    def vocabulary() -> Dict[str, int]:
        waits = ['wait' + str(i) for i in range(1, MidiParser.MAX_WAIT + 1)]
        ps = ['p' + str(i) for i in range(109)]
        endps = ['endp' + str(i) for i in range(109)]

        vocabulary = waits + ps + endps
        return {vocabulary[i]: i for i in range(len(vocabulary))}

    @staticmethod
    def normalize_tempo(csv_rows):
        # Find the number of pulses (ticks) per quarter note (PPQ) and compute a normalization constant
        # based on generated files with PPQ=384
        header = next(row for row in csv_rows if row[2] == 'Header')
        normalization_constant = 384 / int(header[5])
        for row in csv_rows:
            row[1] = int(row[1]) * normalization_constant

        # Sort the rows by time
        csv_rows = sorted(csv_rows, key=lambda cr: int(cr[1]))

        # Normalize time for a default tempo=500_000
        norm_csv = []
        m_time_prev = 0
        m_adjusted_time = 0
        tempo_ratio = 1
        for row in csv_rows:
            m_track, m_time, m_type, msg = row[0], int(row[1]), row[2], row[3:]

            if m_type == 'Tempo':
                tempo_ratio = int(row[3]) / 500_000

            if m_type not in ['Note_on_c', 'Note_off_c']:
                continue  # skip all MIDI commands other than note_on and note_off

            rel_t = (m_time - m_time_prev)
            m_time_prev = m_time
            m_adjusted_time += rel_t * tempo_ratio

            new_msg = [m_track, str(int(m_adjusted_time)), m_type] + msg
            norm_csv.append(new_msg)

        return norm_csv
