import numpy as np
import torch

from tensor import Tensor
from nn import Linear, ReLU, GELU


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


def compare_linear(rng):
    """check linear layer's outputs, input gradients and parameter gradients"""
    for shape in ((3,), (4, 3), (2, 4, 3)):
        layer = Linear(3, 2, rng=rng)
        reference = torch.nn.Linear(3, 2, dtype=torch.float64)

        # PyTorch stores linear weights as (out_features, in_features).
        with torch.no_grad():
            reference.weight.copy_(torch.tensor(layer.weight.data.T))
            reference.bias.copy_(torch.tensor(layer.bias.data))

        values = rng.normal(size=shape)
        x = Tensor(values, requires_grad=True)
        torch_x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        output = layer(x)
        torch_output = reference(torch_x)
        assert output.shape == shape[:-1] + (2,)
        np.testing.assert_allclose(
            output.data, torch_output.detach().numpy(), rtol=1e-9, atol=1e-10
        )

        (output**2).mean().backward()
        (torch_output**2).mean().backward()
        pairs = (
            (x.grad, torch_x.grad.numpy()),
            (layer.weight.grad, reference.weight.grad.numpy().T),
            (layer.bias.grad, reference.bias.grad.numpy()),
        )
        for actual, expected in pairs:
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)

        assert layer.parameters() == [layer.weight, layer.bias]
        largest_error = max(float(np.max(np.abs(a - b))) for a, b in pairs)
        layer.zero_grad()
        assert all(np.all(p.grad == 0) for p in layer.parameters())
        print(f"linear {str(shape):<17} gradient error: {largest_error:.2e}")


def compare_activations():
    """compare elementwise outputs and gradients, including zero and the tails"""
    values = np.array([-20, -8, -3, -1, -0.1, -1e-8, 0, 1e-8, 0.1, 1, 3, 20])
    for activation, reference in (
        (ReLU(), torch.nn.ReLU()),
        (GELU(), torch.nn.GELU(approximate="tanh")),
    ):
        largest_error = 0.0
        assert activation.parameters() == []
        for shape in ((), (12,), (3, 4), (2, 2, 3)):
            data = np.array(0.0) if not shape else values.reshape(shape)
            x = Tensor(data, requires_grad=True)
            torch_x = torch.tensor(data, dtype=torch.float64, requires_grad=True)
            output = activation(x)
            torch_output = reference(torch_x)
            assert output.shape == shape
            np.testing.assert_allclose(
                output.data, torch_output.detach().numpy(), rtol=1e-9, atol=1e-10
            )

            # Different incoming gradients check the chain rule element by element.
            weights = np.linspace(-2, 3, data.size).reshape(shape)
            (output * weights).sum().backward()
            (torch_output * torch.tensor(weights)).sum().backward()
            expected = torch_x.grad.numpy()
            np.testing.assert_allclose(x.grad, expected, rtol=1e-9, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(x.grad - expected))))

        print(f"{type(activation).__name__:<24} gradient error: {largest_error:.2e}")


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

    compare_linear(rng)
    compare_activations()

    print(f"all comparisons passed using PyTorch {torch.__version__}.")


if __name__ == "__main__":
    main()
