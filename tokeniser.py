from numbers import Integral


class CharacterTokeniser:
    def __init__(self, text):
        if not isinstance(text, str):
            raise TypeError("vocabulary text must be a string")
        if not text:
            raise ValueError("vocabulary text must not be empty")

        # sorting keeps the IDs consistent for the same set of characters
        self.characters = tuple(sorted(set(text)))
        self.char_to_id = {char: index for index, char in enumerate(self.characters)}

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as error:
            raise ValueError(f"character not in vocabulary: {error.args[0]!r}") from None

    def decode(self, token_ids):
        characters = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise TypeError("token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError("token ID is out of range")
            characters.append(self.characters[token_id])
        return "".join(characters)
