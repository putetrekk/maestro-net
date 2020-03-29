from os import path
import csv
import py_midicsv
import re


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
        csv_rows = py_midicsv.midi_to_csv(file_path)

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

    def decode_notwise(self, encoded: str):
        words = encoded.split(' ')

        csv_rows = []

        m_track, m_channel, m_time = 1, 0, 0
        for word in words:
            wait = re.match(r'wait([0-9]*)', word)
            if wait:
                m_time += int(wait.group(1)) * self.TIME_CONSTANT

            note = re.match(r'(?P<end>end)?p(?P<note>[0-9]*)', word)
            if note:
                m_type = 'Note_off_c' if note.group('end') else 'Note_on_c'
                m_note = int(note.group('note'))
                m_velocity = 127
                csv_rows.append([m_track, m_time, m_type, m_channel, m_note, m_velocity])

        return csv_rows


