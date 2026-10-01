import gc
from pathlib import Path

import numpy as np

from model import load_model
from tokeniser import WordTokeniser


def generate(model, tokeniser, prompt, new_tokens=300, temperature=1.0, rng=None):
    """samples new tokens and returns them with the prompt"""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    rng = np.random.default_rng() if rng is None else rng
    token_ids = tokeniser.encode(prompt)
    word_tokens = isinstance(tokeniser, WordTokeniser)

    for step in range(new_tokens):
        # keeps the full output but only feeds the latest context into GPT
        context = token_ids[-model.context_length:]
        logits = model(context)
        scores = logits[-1] / temperature
        if word_tokens:
            # unknown words belong in training targets, not generated text
            scores.data[tokeniser.unknown_id] = -np.inf
        probabilities = scores.softmax().data
        next_token = int(rng.choice(tokeniser.vocab_size, p=probabilities))
        if word_tokens and next_token == tokeniser.end_id:
            gc.collect()
            break
        token_ids.append(next_token)
        if (step + 1) % 100 == 0 or step + 1 == new_tokens:
            gc.collect()

    return tokeniser.decode(token_ids)


def main(prompt="ROMEO:\n", new_tokens=300, temperature=1.0, seed=7, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "shakespeare_6000_steps.npz"
    path = Path(checkpoint_path)
    model, tokeniser = load_model(path)
    print(f"Model: {path}")
    print(f"Prompt: {prompt!r}")
    unit = "tokens (maximum)" if isinstance(tokeniser, WordTokeniser) else "characters"
    print(f"New {unit}: {new_tokens}, temperature: {temperature}, seed: {seed}")
    print(f"Context length: {model.context_length}")
    print()
    print(generate(model, tokeniser, prompt, new_tokens, temperature, np.random.default_rng(seed)))


# if __name__ == "__main__":
  #  main(temperature=0.7)
if __name__ == "__main__":
    main(
        prompt="Once upon a time,",
        new_tokens=100,
        temperature=0.7,
        checkpoint_path=Path(__file__).resolve().parent
        / "checkpoints"
        / "tinystories_ctx32_emb128_1000_steps_lr_decay_best.npz",
    )
