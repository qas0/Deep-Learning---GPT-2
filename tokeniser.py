class CharacterTokeniser:
    def __init__(self, text):
        # sorting keeps the IDs consistent for the same set of characters
        self.characters = tuple(sorted(set(text)))
        self.char_to_id = {char: index for index, char in enumerate(self.characters)}

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids):
        characters = []
        for token_id in token_ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError("token ID is out of range")
            characters.append(self.characters[token_id])
        return "".join(characters)
