from midiparser import MidiParser

if __name__ == '__main__':
    parser = MidiParser('midi/')

    notewise_encoding = parser.read_music('sonatas_k-023_(c)sankey.mid')

    parser.save_music(notewise_encoding, 'output.midi')
