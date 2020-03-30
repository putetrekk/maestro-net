import time
from os import path
import pathlib
import csv
import py_midicsv as pm
import re


def generate_filename(num_sonatas_trained: int, char_per_sonata: int):
    ts = round(time.time())
    filename = f'result_{num_sonatas_trained}_{char_per_sonata}_{ts}.midi'
    return filename


class MidiParser:
    TIME_CONSTANT = 10
    MAX_WAIT = 10

    def __init__(self, folder: str):
        self.folder: str = folder

    def time_to_waits(self, time_t):
        waits = [self.MAX_WAIT] * (time_t // self.MAX_WAIT) + [time_t % self.MAX_WAIT]
        return str.join(' ', ['wait' + str(w) for w in waits if w > 0])

    def read_music(self, file: str):
        file_path = path.join(self.folder, file)
        csv_rows = pm.midi_to_csv(file_path)

        encoded = ''
        time_prev = 0
        for row in csv.reader(csv_rows, delimiter=',', skipinitialspace=True):
            m_track, m_time, m_type = row[0], int(row[1]), row[2]

            wait_t = round((m_time - time_prev) / self.TIME_CONSTANT)
            time_prev = m_time

            if m_type == 'Note_on_c' or m_type == 'Note_off_c':
                channel, note, velocity = row[3], row[4], row[5]

                if wait_t != 0:
                    encoded += ' ' + self.time_to_waits(wait_t)

                if m_type == 'Note_on_c':
                    encoded += ' p' + note
                elif m_type == 'Note_off_c':
                    encoded += ' endp' + note

        return encoded

    def save_music(self, encoded: str, filename: str):
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

        folder = path.join(self.folder, 'output/')
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        filepath = path.join(folder, filename)
        with open(filepath, 'wb') as file:
            midi_writer = pm.FileWriter(file)
            midi_writer.write(midi_object)
