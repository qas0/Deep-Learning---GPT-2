import gc
from pathlib import Path
from time import perf_counter

import numpy as np

from model import GPT, save_model, load_model
from nn import CrossEntropyLoss
from optim import Adam
from text_data import load_shakespeare, get_batch


def evaluate(model, token_ids, loss_function, batch_size, batches, seed):
    # repeats the same sampled windows so loss readings are comparable
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(token_ids, batch_size, model.context_length, rng)
        loss = loss_function(model(inputs), targets)
        losses.append(loss.data.item())
    mean_loss = float(np.mean(losses))
    if not np.isfinite(mean_loss):
        raise RuntimeError("non-finite evaluation loss")
    return mean_loss


def compare_contexts(checkpoint_paths, batches=100):
    _, _, validation_ids = load_shakespeare()
    models = [load_model(path)[0] for path in checkpoint_paths]
    context_length = max(model.context_length for model in models)
    rng = np.random.default_rng(10)
    loss_function = CrossEntropyLoss()
    losses = np.zeros(len(models))

    for batch in range(batches):
        inputs, targets = get_batch(validation_ids, 8, context_length, rng)
        for index, model in enumerate(models):
            logits = model(inputs[:, -model.context_length:])
            loss = loss_function(logits[:, -1, :], targets[:, -1])
            losses[index] += loss.data.item()
        if (batch + 1) % 10 == 0 or batch + 1 == batches:
            gc.collect()

    losses /= batches
    print(f"Matched validation: {batches * 8} next-character targets, seed: 10")
    for path, model, loss in zip(checkpoint_paths, models, losses):
        print(f"{path}: context={model.context_length}, loss={loss:.4f}")
    return losses


def main(steps=1000, save_path=None, context_length=32, batch_size=8):
    seed = 7
    embedding_dim = 128
    num_heads = 4
    num_layers = 2
    learning_rate = 0.001
    eval_interval = 100
    eval_batches = 10

    tokeniser, train_ids, validation_ids = load_shakespeare()
    model = GPT(
        tokeniser.vocab_size, context_length, embedding_dim, num_heads, num_layers,
        rng=np.random.default_rng(seed),
    )
    parameters = model.parameters()
    optimiser = Adam(parameters, lr=learning_rate)
    loss_function = CrossEntropyLoss()
    train_rng = np.random.default_rng(seed + 1)

    if save_path is None:
        save_path = (
            Path(__file__).resolve().parent / "checkpoints"
            / f"shakespeare_ctx{context_length}_emb{embedding_dim}_{steps}_steps.npz"
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = save_path.with_name(f"{save_path.stem}_best.npz")
    best_validation_loss = float("inf")
    best_step = 0

    print(f"Tiny Shakespeare: {len(train_ids):,} training, {len(validation_ids):,} validation characters")
    print(f"Vocabulary: {tokeniser.vocab_size} characters")
    print(f"GPT: layers={num_layers}, heads={num_heads}, embedding={embedding_dim}, context={context_length}")
    print(f"Parameters: {sum(p.data.size for p in parameters):,}")
    print(f"Adam: lr={learning_rate}, betas={optimiser.betas}, eps={optimiser.eps}")
    print(f"Seed: {seed}, steps: {steps}, batch size: {batch_size}")
    print(f"Evaluation: {eval_batches} fixed sampled batches per split, every {eval_interval} steps + final step")
    start = perf_counter()

    for step in range(steps + 1):
        if step % eval_interval == 0 or step == steps:
            train_loss = evaluate(
                model, train_ids, loss_function, batch_size, eval_batches, seed + 2,
            )
            validation_loss = evaluate(
                model, validation_ids, loss_function, batch_size, eval_batches, seed + 3,
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                save_model(model, tokeniser, best_path)

            # clears unused graphs before building up in training
            gc.collect()
            print(
                f"Step {step:4d}: train loss={train_loss:.4f} | "
                f"validation loss={validation_loss:.4f} | elapsed={perf_counter() - start:.1f}s",
                flush=True,
            )
        if step == steps:
            break

        inputs, targets = get_batch(train_ids, batch_size, context_length, train_rng)
        logits = model(inputs)
        loss = loss_function(logits, targets)
        if not np.isfinite(loss.data.item()):
            raise RuntimeError(f"non-finite training loss at step {step + 1}")
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Training and evaluation time: {perf_counter() - start:.2f}s")
    save_model(model, tokeniser, save_path)
    print(f"Saved final model: {save_path.resolve()}")
    print(f"Best validation loss: {best_validation_loss:.4f} at step {best_step}")
    print(f"Saved best model: {best_path.resolve()}")


if __name__ == "__main__":
    main(
        steps=20000,
        context_length=32,
        batch_size=8,
    )


    # compare_contexts([
    #   Path(__file__).resolve().parent / "checkpoints" / "shakespeare_6000_steps.npz",
    #     Path(__file__).resolve().parent / "checkpoints" / "shakespeare_ctx64_6000_steps.npz",
    #])
