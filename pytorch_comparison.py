import numpy as np
import torch

from tensor import Tensor
from nn import Parameter, Linear, ReLU, GELU, Softmax, CrossEntropyLoss
from optim import SGD, Adam


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
    """checks activation outputs + gradients against PyTorch, including 0 and large positive or negative inputs"""
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

            # uses different incoming gradients to check the chain rule for each value
            weights = np.linspace(-2, 3, data.size).reshape(shape)
            (output * weights).sum().backward()
            (torch_output * torch.tensor(weights)).sum().backward()
            expected = torch_x.grad.numpy()
            np.testing.assert_allclose(x.grad, expected, rtol=1e-9, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(x.grad - expected))))

        print(f"{type(activation).__name__:<24} gradient error: {largest_error:.2e}")


def compare_probabilities(rng):
    """checks full probability arrays and weighted gradients along different axes"""
    cases = [
        (rng.normal(size=(4,)), -1),
        (rng.normal(size=(2, 3)), 0),
        (rng.normal(size=(2, 3, 4)), -1),
        (np.array([[1000., 1001., -1000.], [-1000., -999., -998.]]), -1),
        (np.zeros((2, 1)), -1),
    ]
    for values, axis in cases:
        weights = rng.normal(size=values.shape)
        for name in ("softmax", "log_softmax"):
            x = Tensor(values, requires_grad=True)
            tx = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            output = Softmax(axis)(x) if name == "softmax" else x.log_softmax(axis)
            expected = getattr(torch, name)(tx, dim=axis)
            assert output.shape == values.shape
            assert np.all(np.isfinite(output.data))
            np.testing.assert_allclose(
                output.data, expected.detach().numpy(), rtol=1e-9, atol=1e-10
            )
            probabilities = output.data if name == "softmax" else np.exp(output.data)
            np.testing.assert_allclose(probabilities.sum(axis=axis), 1.0)
            assert np.all(probabilities >= 0)
            (output * weights).sum().backward()
            (expected * torch.tensor(weights)).sum().backward()
            np.testing.assert_allclose(x.grad, tx.grad.numpy(), rtol=1e-9, atol=1e-10)

    print("softmax/log-softmax      outputs, gradients and stability passed")


def compare_cross_entropy(rng):
    """checks cross entropy loss + gradients against PyTorch"""
    cases = [
        (rng.normal(size=(3,)), np.array(1)),
        (rng.normal(size=(4, 3)), np.array([0, 1, 1, 2])),
        (rng.normal(size=(2, 2, 3)), np.array([[0, 2], [1, 0]])),
        (np.array([[1000., -1000., 0.], [-1000., 1000., 0.]]), np.array([1, 0])),
        (np.zeros((2, 1)), np.array([0, 0])),
    ]
    loss_function = CrossEntropyLoss()
    for values, targets in cases:
        compare_gradients(
            f"cross-entropy {values.shape}",
            lambda x: loss_function(x, targets),
            lambda x: torch.nn.functional.cross_entropy(
                x.reshape(-1, values.shape[-1]),
                torch.tensor(targets.reshape(-1), dtype=torch.long),
            ),
            [values],
        )
        loss = loss_function(Tensor(values), targets).data
        shifted_loss = loss_function(Tensor(values + 10000), targets).data
        assert np.isfinite(loss)
        np.testing.assert_allclose(loss, shifted_loss, rtol=1e-9, atol=1e-10)


def compare_optimisers(rng):
    """checks optimiser updates + gradient resets against PyTorch over several steps"""
    values = rng.normal(size=(8, 3))
    targets = values @ np.array([[2.0], [-1.0], [0.5]]) + 0.3
    x = Tensor(values)
    torch_x = torch.tensor(values, dtype=torch.float64)
    torch_targets = torch.tensor(targets, dtype=torch.float64)

    cases = (
        (SGD, torch.optim.SGD, {"lr": 0.0}),
        (SGD, torch.optim.SGD, {"lr": 0.05}),
        (Adam, torch.optim.Adam, {"lr": 0.0}),
        (Adam, torch.optim.Adam, {"lr": 0.05}),
        (Adam, torch.optim.Adam, {"lr": 0.01, "betas": (0.8, 0.95), "eps": 1e-6}),
    )
    for optimiser_class, torch_class, options in cases:
        lr = options["lr"]
        layer = Linear(3, 1, rng=rng)
        reference = torch.nn.Linear(3, 1, dtype=torch.float64)
        with torch.no_grad():
            reference.weight.copy_(torch.tensor(layer.weight.data.T))
            reference.bias.copy_(torch.tensor(layer.bias.data))

        optimiser = optimiser_class(layer.parameters(), **options)
        torch_optimiser = torch_class(reference.parameters(), **options)
        initial_loss = float(((layer(x) - targets) ** 2).mean().data)
        largest_error = 0.0

        for _ in range(10):
            optimiser.zero_grad()
            torch_optimiser.zero_grad(set_to_none=False)
            assert all(np.all(p.grad == 0) for p in layer.parameters())

            # builds a fresh graph using the updated weights each time
            loss = ((layer(x) - targets) ** 2).mean()
            torch_loss = ((reference(torch_x) - torch_targets) ** 2).mean()
            np.testing.assert_allclose(
                loss.data, torch_loss.detach().numpy(), rtol=1e-9, atol=1e-10
            )
            loss.backward()
            torch_loss.backward()
            optimiser.step()
            torch_optimiser.step()

            pairs = (
                (layer.weight.data, reference.weight.detach().numpy().T),
                (layer.bias.data, reference.bias.detach().numpy()),
            )
            for actual, expected in pairs:
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
                largest_error = max(largest_error, float(np.max(np.abs(actual - expected))))

        final_loss = float(((layer(x) - targets) ** 2).mean().data)
        if lr == 0:
            assert final_loss == initial_loss
        else:
            assert final_loss < initial_loss
        print(
            f"{optimiser_class.__name__} lr={lr:<14} update error: {largest_error:.2e}, "
            f"loss: {initial_loss:.4f} -> {final_loss:.4f}"
        )


def compare_adam_state():
    """checks Adam history with changing, zero + missing gradients"""
    parameters = [Parameter(2.0), Parameter([[1.0, -1.0], [0.5, -0.5]])]
    references = [torch.tensor(p.data, requires_grad=True) for p in parameters]
    optimiser = Adam(parameters)
    torch_optimiser = torch.optim.Adam(references)

    for step, scale in enumerate((1.0, -2.0, 0.0, 1e-10)):
        for index, (parameter, reference) in enumerate(zip(parameters, references)):
            # missing gradients skip the update, zero gradients still use past averages
            gradient = None if index == 1 and step == 1 else np.full(parameter.shape, scale)
            parameter.grad = gradient
            reference.grad = None if gradient is None else torch.tensor(gradient)

        optimiser.step()
        torch_optimiser.step()
        for index, (parameter, reference) in enumerate(zip(parameters, references)):
            state = torch_optimiser.state[reference]
            pairs = (
                (parameter.data, reference.detach().numpy()),
                (optimiser.first_moments[index], state["exp_avg"].numpy()),
                (optimiser.second_moments[index], state["exp_avg_sq"].numpy()),
            )
            for actual, expected in pairs:
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
            assert optimiser.steps[index] == state["step"].item()

        history = [value.copy() for value in optimiser.first_moments + optimiser.second_moments]
        steps = optimiser.steps.copy()
        optimiser.zero_grad()
        assert all(p.grad is None or np.all(p.grad == 0) for p in parameters)
        for actual, expected in zip(optimiser.first_moments + optimiser.second_moments, history):
            np.testing.assert_array_equal(actual, expected)
        assert optimiser.steps == steps

    print("Adam history            averages, step counts and gradient resets passed")


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
    compare_probabilities(rng)
    compare_cross_entropy(rng)
    compare_optimisers(rng)
    compare_adam_state()

    print(f"all comparisons passed using PyTorch {torch.__version__}.")


if __name__ == "__main__":
    main()
