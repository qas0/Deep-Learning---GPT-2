from pathlib import Path
from urllib.request import urlopen

import numpy as np

from tokeniser import CharacterTokeniser, WordTokeniser


DATA_PATH = Path(__file__).resolve().parent / "data" / "tiny_shakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def load_shakespeare():
    # keeps a local copy so it only needs downloading once
    if not DATA_PATH.exists():
        with urlopen(DATA_URL, timeout=30) as response:
            data = response.read()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_PATH.write_bytes(data)

    text = DATA_PATH.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    train_text, validation_text = text[:split], text[split:]
    tokeniser = CharacterTokeniser(train_text)
    train_ids = np.array(tokeniser.encode(train_text), dtype=np.int64)
    validation_ids = np.array(tokeniser.encode(validation_text), dtype=np.int64)
    return tokeniser, train_ids, validation_ids


def load_story_text(split, stories):
    path = DATA_PATH.parent / f"tinystories_{split}_{stories}.txt"
    if not path.exists():
        print(f"Downloading {stories:,} complete TinyStories {split} stories...", flush=True)
        lines = []
        completed = 0
        # stops at a story boundary instead of downloading the full dataset
        with urlopen(f"{STORIES_URL}/TinyStories-{split}.txt", timeout=30) as response:
            for line in response:
                lines.append(line)
                if line.strip() == b"<|endoftext|>":
                    completed += 1
                    if completed == stories:
                        break
        if completed != stories:
            raise RuntimeError("TinyStories download ended before the requested story count")
        text = b"".join(lines).decode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def load_tinystories(train_stories=10000, validation_stories=1000, vocab_size=4096):
    train_text = load_story_text("train", train_stories)
    validation_text = load_story_text("valid", validation_stories)
    # validation uses the training vocabulary, including its unknown-word ID
    tokeniser = WordTokeniser(train_text, vocab_size=vocab_size)
    train_ids = np.array(tokeniser.encode(train_text), dtype=np.int64)
    validation_ids = np.array(tokeniser.encode(validation_text), dtype=np.int64)
    return tokeniser, train_ids, validation_ids


def get_batch(token_ids, batch_size, context_length, rng):
    if not 0 < context_length < len(token_ids):
        raise ValueError("split must contain a full context and its next token")

    # samples within one split so sequences cannot cross into validation
    starts = rng.integers(0, len(token_ids) - context_length, size=batch_size)
    positions = starts[:, None] + np.arange(context_length)
    inputs = token_ids[positions]
    targets = token_ids[positions + 1]
    return inputs, targets


def main(dataset="tinystories"):
    seed = 7
    batch_size = 4
    context_length = 32
    loader = {"shakespeare": load_shakespeare, "tinystories": load_tinystories}[dataset]
    tokeniser, train_ids, validation_ids = loader()

    if isinstance(tokeniser, CharacterTokeniser):
        for char, token_id in tokeniser.char_to_id.items():
            print(repr(char), token_id)
    inputs, targets = get_batch(train_ids, batch_size, context_length, np.random.default_rng(seed))

    name = "TinyStories" if dataset == "tinystories" else "Tiny Shakespeare"
    unit = "word/punctuation tokens" if dataset == "tinystories" else "characters"
    print(f"{name}: {len(train_ids):,} training, {len(validation_ids):,} validation {unit}")
    print(f"Vocabulary: {tokeniser.vocab_size} {unit}")
    print(f"Seed: {seed}, batch size: {batch_size}, context length: {context_length}")
    print(f"Input shape: {inputs.shape}, target shape: {targets.shape}")
    print(f"Input:  {tokeniser.decode(inputs[0])!r}")
    print(f"Target: {tokeniser.decode(targets[0])!r}")


if __name__ == "__main__":
    main()
