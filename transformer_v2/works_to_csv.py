import csv

from main import load_data


def split_input_output(all_data: list, words_per_section = 75):
	inputs, outputs = [], []
	for data in all_data:
		sep = ' '
		sectioned_notes = []

		groups = data.split(sep)
		while len(groups):
			sectioned_notes.append(sep.join(groups[:words_per_section]))
			groups = groups[words_per_section:]

		for i in range(len(sectioned_notes) - 1):
			inputs.append(sectioned_notes[i])
			outputs.append(sectioned_notes[i + 1])

	return inputs, outputs


data = load_data('../data/scarlatti_k1_555.txt', 100_000_000)
inputs, outputs = split_input_output(data)

with open('data/split_scarlatti.csv', 'w', encoding='utf-8') as f:
	writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
	writer.writerow([f'Input', f'Target'])

	for input, output in zip(inputs, outputs):
		writer.writerow([input, output])
