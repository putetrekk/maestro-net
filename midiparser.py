from os import path
import csv
import py_midicsv

TIME_CONSTANT = 10
MAX_WAIT = 10


def time_to_waits(time_t):
    waits = [MAX_WAIT] * (time_t // MAX_WAIT) + [time_t % MAX_WAIT]
    return str.join(' ', ['wait' + str(w) for w in waits if w > 0])


class MidiParser:
    def __init__(self, folder: str):
        self.folder: str = folder

    def read_music(self, file: str):
        file_path = path.join(self.folder, file)
        csv_string = py_midicsv.midi_to_csv(file_path)

        encoded = ''
        time_prev = 0
        for row in csv.reader(csv_string, delimiter=',', skipinitialspace=True):
            m_track, m_time, m_type = row[0], int(row[1]), row[2]

            wait_t = (m_time - time_prev) // TIME_CONSTANT
            time_prev = m_time

            if m_type == 'Note_on_c' or m_type == 'Note_off_c':
                channel, note, velocity = row[3], row[4], row[5]

                if wait_t != 0:
                    encoded += ' ' + time_to_waits(wait_t)

                if m_type == 'Note_on_c':
                    encoded += " p" + note
                elif m_type == 'Note_off_c':
                    encoded += ' endp' + note

        return encoded
