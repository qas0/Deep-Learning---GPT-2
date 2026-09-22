from time import perf_counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tensor import Tensor
from nn import Module, Linear, ReLU, CrossEntropyLoss
from optim import Adam


class IrisClassifier(Module):
    def __init__(self, rng):
        self.hidden = Linear(4, 16, rng=rng)
        self.activation = ReLU()
        self.output = Linear(16, 3, rng=rng)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))


def load_data(seed):
    inputs, targets = load_iris(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(
        inputs, targets, test_size=0.2, stratify=targets, random_state=seed
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x, train_y, test_size=0.25, stratify=train_y, random_state=seed
    )

    # uses training statistics for every split so held out data doesnt affect scaling
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    return (
        ((train_x - mean) / scale, train_y),
        ((validation_x - mean) / scale, validation_y),
        ((test_x - mean) / scale, test_y),
    )


def evaluate(model, inputs, targets, loss_function):
    scores = model(Tensor(inputs))
    loss = float(loss_function(scores, targets).data)
    predictions = scores.data.argmax(axis=1)
    accuracy = float(np.mean(predictions == targets))
    return loss, accuracy, predictions


def main():
    seed = 7
    epochs = 200
    batch_size = 16
    learning_rate = 0.01
    rng = np.random.default_rng(seed)
    (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = load_data(seed)
    model = IrisClassifier(rng)
    parameters = model.parameters()
    optimiser = Adam(parameters, lr=learning_rate)
    loss_function = CrossEntropyLoss()

    print(f"Iris: {len(train_y)} training, {len(validation_y)} validation, {len(test_y)} test")
    print(f"Network: 4 -> 16 -> 3, parameters: {sum(p.data.size for p in parameters)}")
    print(f"Adam: lr={learning_rate}, betas={optimiser.betas}, eps={optimiser.eps}")
    print(f"Seed: {seed}, epochs: {epochs}, batch size: {batch_size}")
    initial_loss, initial_accuracy, _ = evaluate(model, train_x, train_y, loss_function)
    print(f"Before training: loss={initial_loss:.4f}, accuracy={initial_accuracy:.2%}")
    best_loss = float("inf")
    best_epoch = 0
    best_parameters = None
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_y))
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            optimiser.zero_grad()
            scores = model(Tensor(train_x[batch]))
            loss = loss_function(scores, train_y[batch])
            loss.backward()
            optimiser.step()

        validation_loss, validation_accuracy, _ = evaluate(
            model, validation_x, validation_y, loss_function
        )
        if not np.isfinite(validation_loss):
            raise RuntimeError(f"non-finite validation loss at epoch {epoch}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            # copies the best weights so later updates dont change them
            best_parameters = [p.data.copy() for p in parameters]

        if epoch == 1 or epoch % 20 == 0:
            train_loss, train_accuracy, _ = evaluate(model, train_x, train_y, loss_function)
            print(
                f"Epoch {epoch:3d}: train loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
                f"validation loss={validation_loss:.4f}, accuracy={validation_accuracy:.2%}"
            )

    for parameter, values in zip(parameters, best_parameters):
        parameter.data[...] = values

    print(f"Selected epoch: {best_epoch} (lowest validation loss)")
    for name, inputs, targets in (
        ("Training", train_x, train_y),
        ("Validation", validation_x, validation_y),
    ):
        loss, accuracy, _ = evaluate(model, inputs, targets, loss_function)
        print(f"{name}: loss={loss:.4f}, accuracy={accuracy:.2%}")

    
    test_loss, test_accuracy, predictions = evaluate(model, test_x, test_y, loss_function)
    print(f"Test: loss={test_loss:.4f}, accuracy={test_accuracy:.2%}")
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (test_y, predictions), 1)
    print("Test confusion matrix: rows = actual, columns = predicted")
    print("Class order: setosa, versicolor, virginica")
    print(confusion)
    print(f"Training and evaluation time: {perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
