from midiparser import MidiParser
import numpy as np

if __name__ == '__main__':
    parser = MidiParser('midi/')

    notewise_encoding = parser.read_music('sonatas_k-008_(c)sankey.mid')

    with open("test.txt", "w") as text_file:
        text_file.write(notewise_encoding)
