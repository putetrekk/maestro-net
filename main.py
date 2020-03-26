from midiparser import MidiParser
import numpy as np

if __name__ == '__main__':
    parser = MidiParser('midi/')
    piano_roll = parser.readMusic('sonatas_k-010_(c)sankey.mid')

