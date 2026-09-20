import numpy as np

from tensor import Tensor


def numerical_gradient(function, inputs, input_number, epsilon=1e-6):
    """estimates gradient using numerical differentiation"""
    values = [np.array(value, dtype=np.float64, copy=True) for value in inputs]
    selected = values[input_number]
    gradient = np.zeros_like(selected)

    # slightly changes one value at a time + measures how much output changes
    for position in np.ndindex(selected.shape):
        original = selected[position]

        selected[position] = original + epsilon
        output_above = function(*values)

        selected[position] = original - epsilon
        output_below = function(*values)

        selected[position] = original
        gradient[position] = (output_above - output_below) / (2 * epsilon)

    return gradient


def check_gradients(name, function, inputs):
    """compare custom autograd against finite-difference gradients"""
    tensors = [Tensor(value, requires_grad=True) for value in inputs]
    output = function(*tensors)

    if output.data.size != 1:
        raise ValueError("gradient checks require a scalar output")

    output.backward()

    def evaluate(*values):
        arguments = [Tensor(value) for value in values]
        return function(*arguments).data.item()

    largest_error = 0.0

    for input_number, tensor in enumerate(tensors):
        estimated = numerical_gradient(evaluate, inputs, input_number)
        error = np.max(np.abs(tensor.grad - estimated))
        largest_error = max(largest_error, float(error))
        np.testing.assert_allclose(tensor.grad, estimated, rtol=1e-5, atol=1e-6)

    print(f"{name:<24} maximum error: {largest_error:.2e}")


def main():
    rng = np.random.default_rng(7)

    check_gradients(
        "arithmetic",
        lambda x, y: ((x**3 + 2 * x - y) / y).mean(),
        [rng.normal(size=(2, 3)), rng.uniform(0.5, 2.0, size=(2, 3))],
    )

    check_gradients(
        "broadcasting",
        lambda x, y: (x * y + y).sum(),
        [rng.normal(size=(2, 1)), rng.normal(size=(3,))],
    )

    check_gradients(
        "matrix multiplication",
        lambda x, y: ((x @ y.T) ** 2).mean(),
        [rng.normal(size=(2, 3)), rng.normal(size=(4, 3))],
    )

    indices = np.array([1, 1, 3])
    check_gradients(
        "reshape and indexing",
        lambda x: (x[indices].reshape(3, 2).transpose(1, 0) ** 2).mean(),
        [rng.normal(size=(5, 2))],
    )

    print("All gradient checks passed.")


if __name__ == "__main__":
    main()
