from midiparser import MidiParser
import numpy as np

if __name__ == '__main__':
    parser = MidiParser('midi/')
    csv = parser.read_music('sonatas_k-001_(c)sankey.mid')

    with open("test.csv", "w") as text_file:
        text_file.write(str.join('', csv))
