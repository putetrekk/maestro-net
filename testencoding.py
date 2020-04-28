# Run to test compression loss

import midiparser

if __name__ == '__main__':
    parser = midiparser.MidiParser('midi/', 'testencodeoutput')

    music = parser.read_music('some_midi_file.mid')
    parser.save_music(music, 'encoded_file.mid')