from pathlib import Path
from urllib.request import urlopen

import numpy as np

from tokeniser import CharacterTokeniser


DATA_PATH = Path(__file__).resolve().parent / "data" / "tiny_shakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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


def get_batch(token_ids, batch_size, context_length, rng):
    if not 0 < context_length < len(token_ids):
        raise ValueError("split must contain a full context and its next character")

    # samples within one split so sequences cannot cross into validation
    starts = rng.integers(0, len(token_ids) - context_length, size=batch_size)
    positions = starts[:, None] + np.arange(context_length)
    inputs = token_ids[positions]
    targets = token_ids[positions + 1]
    return inputs, targets


def main():
    seed = 7
    batch_size = 4
    context_length = 32
    tokeniser, train_ids, validation_ids = load_shakespeare()
    inputs, targets = get_batch(train_ids, batch_size, context_length, np.random.default_rng(seed))

    print(f"Tiny Shakespeare: {len(train_ids):,} training, {len(validation_ids):,} validation characters")
    print(f"Vocabulary: {tokeniser.vocab_size} characters")
    print(f"Seed: {seed}, batch size: {batch_size}, context length: {context_length}")
    print(f"Input shape: {inputs.shape}, target shape: {targets.shape}")
    print(f"Input:  {tokeniser.decode(inputs[0])!r}")
    print(f"Target: {tokeniser.decode(targets[0])!r}")


if __name__ == "__main__":
    main()
