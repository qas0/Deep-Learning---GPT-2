import numpy as np
import torch

from tensor import Tensor


def compare_gradients(name, custom_function, torch_function, inputs):
    """compare forward results and gradients with PyTorch"""
    custom_inputs = [Tensor(value, requires_grad=True) for value in inputs]
    torch_inputs = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in inputs
    ]

    custom_output = custom_function(*custom_inputs)
    torch_output = torch_function(*torch_inputs)

    if custom_output.data.size != 1 or torch_output.numel() != 1:
        raise ValueError("comparisons require a scalar output")

    custom_output.backward()
    torch_output.backward()

    torch_value = torch_output.detach().cpu().numpy()
    output_error = float(np.max(np.abs(custom_output.data - torch_value)))
    largest_gradient_error = 0.0

    np.testing.assert_allclose(
        custom_output.data,
        torch_value,
        rtol=1e-9,
        atol=1e-10,
    )

    # checks each inputs gradients against the one from PyTorch
    for custom_input, torch_input in zip(custom_inputs, torch_inputs):
        torch_gradient = torch_input.grad.detach().cpu().numpy()
        gradient_error = np.max(np.abs(custom_input.grad - torch_gradient))
        largest_gradient_error = max(
            largest_gradient_error,
            float(gradient_error),
        )
        np.testing.assert_allclose(
            custom_input.grad,
            torch_gradient,
            rtol=1e-9,
            atol=1e-10,
        )

    print(
        f"{name:<24} "
        f"output error: {output_error:.2e}, "
        f"gradient error: {largest_gradient_error:.2e}"
    )


def main():
    rng = np.random.default_rng(7)

    arithmetic_inputs = [
        rng.normal(size=(2, 3)),
        rng.uniform(0.5, 2.0, size=(2, 3)),
    ]
    compare_gradients(
        "arithmetic",
        lambda x, y: ((x**3 + 2 * x - y) / y).mean(),
        lambda x, y: ((x**3 + 2 * x - y) / y).mean(),
        arithmetic_inputs,
    )

    broadcasting_inputs = [
        rng.normal(size=(2, 1)),
        rng.normal(size=(3,)),
    ]
    compare_gradients(
        "broadcasting",
        lambda x, y: (x * y + y).sum(),
        lambda x, y: (x * y + y).sum(),
        broadcasting_inputs,
    )

    matrix_inputs = [
        rng.normal(size=(2, 3)),
        rng.normal(size=(4, 3)),
    ]
    compare_gradients(
        "matrix multiplication",
        lambda x, y: ((x @ y.T) ** 2).mean(),
        lambda x, y: ((x @ y.T) ** 2).mean(),
        matrix_inputs,
    )

    custom_indices = np.array([1, 1, 3])
    torch_indices = torch.tensor([1, 1, 3], dtype=torch.long)
    indexing_inputs = [rng.normal(size=(5, 2))]
    compare_gradients(
        "reshape and indexing",
        lambda x: (x[custom_indices].reshape(3, 2).transpose(1, 0) ** 2).mean(),
        lambda x: (x[torch_indices].reshape(3, 2).permute(1, 0) ** 2).mean(),
        indexing_inputs,
    )

    print(f"all comparisons passed using PyTorch {torch.__version__}.")


if __name__ == "__main__":
    main()
