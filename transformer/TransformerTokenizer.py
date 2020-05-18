from preprocessing import create_tokenizer


class TransformerTokenizer():
	def __init__(self):
		self.tokenizer = create_tokenizer()
		self.vocab_size = len(self.tokenizer.word_index)
		self.start_token = [self.vocab_size]
		self.end_token = [len(self.tokenizer.word_index) + 1]

	# List of sentences
	def encode_data(self, data: list, sequence_length):
		sequences = []
		for line in data:
			token_list = self.tokenizer.texts_to_sequences([line])[0]
			for i in range(sequence_length, len(token_list)):
				n_gram_sequence = token_list[i - sequence_length:i + 1]
				sequences.append(n_gram_sequence)

		return sequences
	# String to Tokens
	def encode(self, sentence: str):
		token_list = self.tokenizer.texts_to_sequences([sentence])[0]
		return token_list

	# Tokens to string
	def decode(self, tokens: list):
		words = ''
		for token_index in tokens:
			word = self.tokenizer.index_word[token_index]
			words += word + " "

		return words
