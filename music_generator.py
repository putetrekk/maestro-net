from tensorflow.keras import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from preprocessing import create_tokenizer


def gen_music(model: Model, length: int, n=1) -> str:
    print(f"Generating music (length: {length})")

    #  Initial Note(s)
    music = "wait4"

    for _ in range(length):
        tokenizer = create_tokenizer()
        token_list = tokenizer.texts_to_sequences([music])[0]
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]

        top_predictions = np.argsort(predictions)[-n:]
        top_probabilities = np.take(predictions, np.argsort(predictions)[-n:])
        top_probabilities /= top_probabilities.sum()
        next_note_index = np.random.choice(top_predictions, p=top_probabilities)

        music += " " + tokenizer.index_word[next_note_index]

    print(f"Generated: " + music)

    return music
