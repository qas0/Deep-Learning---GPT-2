import gc
from pathlib import Path

import numpy as np

from model import load_model


def generate(model, tokeniser, prompt, new_tokens=300, temperature=1.0, rng=None):
    """samples new characters and returns them with the prompt"""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    rng = np.random.default_rng() if rng is None else rng
    token_ids = tokeniser.encode(prompt)

    for step in range(new_tokens):
        # keeps the full output but only feeds the latest context into GPT
        context = token_ids[-model.context_length:]
        logits = model(context)
        probabilities = (logits[-1] / temperature).softmax().data
        next_token = int(rng.choice(tokeniser.vocab_size, p=probabilities))
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
    print(f"New characters: {new_tokens}, temperature: {temperature}, seed: {seed}")
    print(f"Context length: {model.context_length}")
    print()
    print(generate(model, tokeniser, prompt, new_tokens, temperature, np.random.default_rng(seed)))


# if __name__ == "__main__":
  #  main(temperature=0.7)
if __name__ == "__main__":
    main(
        temperature=0.7,
        checkpoint_path=Path(__file__).resolve().parent
        / "checkpoints"
        / "shakespeare_ctx32_emb64_10000_steps.npz",
    )
