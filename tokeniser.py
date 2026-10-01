import re
from collections import Counter


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


class WordTokeniser:
    unknown_token = "<unk>"
    end_token = "<|endoftext|>"

    def __init__(self, text="", vocab_size=4096, vocabulary=None):
        if vocabulary is None:
            counts = Counter(self.tokenise(text))
            counts.pop(self.unknown_token, None)
            counts.pop(self.end_token, None)
            # keeps common training words, with stable IDs for frequency ties
            common = sorted(counts, key=lambda token: (-counts[token], token))[:vocab_size - 2]
            vocabulary = (self.unknown_token, self.end_token, *sorted(common))
        
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.unknown_id = self.token_to_id[self.unknown_token]
        self.end_id = self.token_to_id[self.end_token]

    @staticmethod
    def tokenise(text):
        text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        # keeps contractions together and gives punctuation its own IDs
        return re.findall(r"<\|endoftext\|>|<unk>|\w+(?:'\w+)*|[^\w\s]", text)

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def encode(self, text):
        return [self.token_to_id.get(token, self.unknown_id) for token in self.tokenise(text)]

    def decode(self, token_ids):
        text = " ".join(self.vocabulary[token_id] for token_id in token_ids)
        # joins punctuation without adding spaces inside double quotes or brackets
        text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r'"([^"]*)"', lambda match: '"' + match[1].strip() + '"', text)
        return "\n\n".join(story.strip() for story in text.split(self.end_token)).strip()
