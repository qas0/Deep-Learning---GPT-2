import numpy as np
import torch

from tensor import Tensor
from nn import Parameter, Linear, Embedding, LayerNorm, SelfAttention, FeedForward, TransformerBlock, ReLU, GELU, Softmax, CrossEntropyLoss
from optim import SGD, Adam
from model import GPT


def compare_gradients(name, custom_function, torch_function, inputs):
    """compare forward results and gradients with PyTorch"""
    custom_inputs = [Tensor(value, requires_grad=True) for value in inputs]
    torch_inputs = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in inputs
    ]

    custom_output = custom_function(*custom_inputs)
    torch_output = torch_function(*torch_inputs)

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


def compare_embedding(rng):
    layer = Embedding(6, 4, rng=rng)
    reference = torch.nn.Embedding(6, 4, dtype=torch.float64)
    with torch.no_grad():
        reference.weight.copy_(torch.tensor(layer.weight.data))
    assert layer.parameters() == [layer.weight]

    cases = (
        np.array(2),
        np.array([0, 2, 2, 5]),
        np.array([[2, 2, 0], [5, 2, 5]]),
        np.full((2, 2), 2),
        np.empty((0, 3), dtype=int),
    )
    largest_error = 0.0
    for ids in cases:
        layer.zero_grad()
        reference.zero_grad()
        output = layer(ids)
        expected = reference(torch.tensor(ids, dtype=torch.long))
        assert output.shape == ids.shape + (4,)
        np.testing.assert_array_equal(output.data, layer.weight.data[ids])
        np.testing.assert_array_equal(output.data, expected.detach().numpy())

        ids[...] = 1
        weights = rng.normal(size=output.shape)
        (output * weights).sum().backward()
        (expected * torch.tensor(weights)).sum().backward()
        expected_gradient = reference.weight.grad.numpy()
        np.testing.assert_allclose(
            layer.weight.grad, expected_gradient, rtol=1e-9, atol=1e-10
        )
        largest_error = max(
            largest_error, float(np.max(np.abs(layer.weight.grad - expected_gradient)))
        )

    print(f"embedding                gradient error: {largest_error:.2e}")


def compare_layer_norm(rng):
    """checks normalised outputs + input, scale and bias gradients"""
    cases = (
        (rng.normal(size=(4,)), 1e-5),
        (rng.normal(size=(3, 4)), 1e-5),
        (rng.normal(size=(2, 3, 4)), 1e-5),
        (np.full((2, 4), 3.0), 1e-5),
        (3.0 + rng.normal(scale=1e-7, size=(2, 4)), 1e-3),
        (rng.normal(size=(2, 1)), 1e-5),
    )
    largest_error = 0.0
    for values, eps in cases:
        layer = LayerNorm(values.shape[-1], eps=eps)
        reference = torch.nn.LayerNorm(values.shape[-1], eps=eps, dtype=torch.float64)
        assert layer.parameters() == [layer.weight, layer.bias]
        np.testing.assert_array_equal(layer.weight.data, np.ones(values.shape[-1]))
        np.testing.assert_array_equal(layer.bias.data, np.zeros(values.shape[-1]))

        # checks learned scale + bias too, instead of only their starting values
        layer.weight.data[:] = rng.normal(size=layer.weight.shape)
        layer.bias.data[:] = rng.normal(size=layer.bias.shape)
        with torch.no_grad():
            reference.weight.copy_(torch.tensor(layer.weight.data))
            reference.bias.copy_(torch.tensor(layer.bias.data))

        x = Tensor(values, requires_grad=True)
        torch_x = torch.tensor(values, requires_grad=True)
        output = layer(x)
        expected = reference(torch_x)
        assert output.shape == values.shape
        np.testing.assert_allclose(output.data, expected.detach().numpy(), rtol=1e-9, atol=1e-10)
        weights = rng.normal(size=values.shape)
        (output * weights).sum().backward()
        (expected * torch.tensor(weights)).sum().backward()
        for actual, target in (
            (x.grad, torch_x.grad.numpy()),
            (layer.weight.grad, reference.weight.grad.numpy()),
            (layer.bias.grad, reference.bias.grad.numpy()),
        ):
            np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(actual - target))))

    print(f"LayerNorm                gradient error: {largest_error:.2e}")


def check_causality(layer, x, output, weights, rng):
    for stop in range(1, x.shape[-2]):
        # changing future tokens must not affect earlier outputs
        changed = x.data.copy()
        changed[..., stop:, :] += rng.normal(size=changed[..., stop:, :].shape) * 10
        np.testing.assert_allclose(
            layer(Tensor(changed)).data[..., :stop, :],
            output.data[..., :stop, :], rtol=1e-9, atol=1e-10,
        )
        layer.zero_grad()
        x.grad.fill(0.0)
        (layer(x)[..., :stop, :] * weights[..., :stop, :]).sum().backward()
        np.testing.assert_array_equal(x.grad[..., stop:, :], 0.0)


def compare_self_attention(rng, causal=False, num_heads=1):
    """checks attention outputs + input and projection gradients"""
    largest_output_error = 0.0
    largest_gradient_error = 0.0
    for shape in ((3, 4), (2, 3, 4), (2, 1, 4), (2, 3, num_heads)):
        layer = SelfAttention(shape[-1], rng=rng, causal=causal, num_heads=num_heads)
        projections = (layer.query, layer.key, layer.value, layer.output)
        reference = torch.nn.MultiheadAttention(
            shape[-1], num_heads, dropout=0.0, batch_first=True, dtype=torch.float64
        )
        with torch.no_grad():
            for projection in projections:
                projection.bias.data[:] = rng.normal(scale=0.1, size=shape[-1])
            # PyTorch stores query, key + value parameters together
            reference.in_proj_weight.copy_(torch.tensor(np.concatenate(
                [p.weight.data.T for p in projections[:3]]
            )))
            reference.in_proj_bias.copy_(torch.tensor(np.concatenate(
                [p.bias.data for p in projections[:3]]
            )))
            reference.out_proj.weight.copy_(torch.tensor(layer.output.weight.data.T))
            reference.out_proj.bias.copy_(torch.tensor(layer.output.bias.data))
        assert layer.parameters() == [
            p for projection in projections for p in projection.parameters()
        ]

        values = rng.normal(size=shape)
        x = Tensor(values, requires_grad=True)
        torch_x = torch.tensor(values, requires_grad=True)
        output = layer(x)
        mask = torch.ones(shape[-2], shape[-2], dtype=torch.bool).triu(1) if causal else None
        expected, _ = reference(
            torch_x, torch_x, torch_x, attn_mask=mask, need_weights=False
        )
        assert output.shape == shape
        expected_values = expected.detach().numpy()
        np.testing.assert_allclose(output.data, expected_values, rtol=1e-9, atol=1e-10)
        largest_output_error = max(
            largest_output_error, float(np.max(np.abs(output.data - expected_values)))
        )

        
        if len(shape) == 3:
            np.testing.assert_allclose(
                output.data[0], layer(Tensor(values[0])).data, rtol=1e-9, atol=1e-10
            )
        if shape[-2] == 1:
            np.testing.assert_allclose(output.data, layer.output(layer.value(x)).data)

        weights = rng.normal(size=shape)
        (output * weights).sum().backward()
        (expected * torch.tensor(weights)).sum().backward()
        pairs = [(x.grad, torch_x.grad.numpy())]
        weight_grads = list(reference.in_proj_weight.grad.chunk(3)) + [reference.out_proj.weight.grad]
        bias_grads = list(reference.in_proj_bias.grad.chunk(3)) + [reference.out_proj.bias.grad]
        for projection, weight_grad, bias_grad in zip(projections, weight_grads, bias_grads):
            pairs.extend((
                (projection.weight.grad, weight_grad.numpy().T),
                (projection.bias.grad, bias_grad.numpy()),
            ))
        for actual, target in pairs:
            np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-10)
            largest_gradient_error = max(
                largest_gradient_error, float(np.max(np.abs(actual - target)))
            )

        if causal:
            np.testing.assert_allclose(
                output.data[..., 0, :], layer.output(layer.value(x)).data[..., 0, :]
            )
            check_causality(layer, x, output, weights, rng)

    name = "causal self-attention" if causal else "self-attention"
    print(
        f"{name}, heads={num_heads}  output error: {largest_output_error:.2e}, "
        f"gradient error: {largest_gradient_error:.2e}"
    )


def compare_feed_forward(rng):
    # checks the combined linear, GELU + projection calculation
    largest_error = 0.0
    for shape, hidden_dim in (((3,), None), ((4, 3), None), ((2, 4, 3), None), ((2, 4, 3), 5)):
        layer = FeedForward(3, hidden_dim=hidden_dim, rng=rng)
        width = 12 if hidden_dim is None else hidden_dim
        reference = torch.nn.Sequential(
            torch.nn.Linear(3, width, dtype=torch.float64),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(width, 3, dtype=torch.float64),
        )
        linears = ((layer.expansion, reference[0]), (layer.projection, reference[2]))
        with torch.no_grad():
            for custom, expected in linears:
                custom.bias.data[:] = rng.normal(scale=0.1, size=custom.bias.shape)
                expected.weight.copy_(torch.tensor(custom.weight.data.T))
                expected.bias.copy_(torch.tensor(custom.bias.data))
        assert layer.parameters() == [p for custom, _ in linears for p in custom.parameters()]

        values = rng.normal(size=shape)
        x = Tensor(values, requires_grad=True)
        torch_x = torch.tensor(values, requires_grad=True)
        output = layer(x)
        expected_output = reference(torch_x)
        assert output.shape == shape
        np.testing.assert_allclose(
            output.data, expected_output.detach().numpy(), rtol=1e-9, atol=1e-10
        )
        weights = rng.normal(size=shape)
        (output * weights).sum().backward()
        (expected_output * torch.tensor(weights)).sum().backward()
        pairs = [(x.grad, torch_x.grad.numpy())]
        for custom, expected in linears:
            pairs.extend((
                (custom.weight.grad, expected.weight.grad.numpy().T),
                (custom.bias.grad, expected.bias.grad.numpy()),
            ))
        for actual, target in pairs:
            np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(actual - target))))

    print(f"feed-forward             gradient error: {largest_error:.2e}")


def torch_block_reference(block):
    width = block.attention.embedding_dim
    reference = torch.nn.TransformerEncoderLayer(
        width, block.attention.num_heads,
        dim_feedforward=block.feed_forward.expansion.out_features,
        dropout=0.0, activation=torch.nn.GELU(approximate="tanh"),
        layer_norm_eps=block.norm1.eps, batch_first=True, norm_first=True, dtype=torch.float64,
    )
    modules = (
        (block.norm1, reference.norm1),
        (block.norm2, reference.norm2),
        (block.attention.output, reference.self_attn.out_proj),
        (block.feed_forward.expansion, reference.linear1),
        (block.feed_forward.projection, reference.linear2),
    )
    mappings = [
        (getattr(custom, name), getattr(expected, name), slice(None))
        for custom, expected in modules for name in ("weight", "bias")
    ]
    for index, projection in enumerate((block.attention.query, block.attention.key, block.attention.value)):
        rows = slice(index * width, (index + 1) * width)
        mappings.extend((
            (projection.weight, reference.self_attn.in_proj_weight, rows),
            (projection.bias, reference.self_attn.in_proj_bias, rows),
        ))
    with torch.no_grad():
        for custom, expected, rows in mappings:
            expected[rows].copy_(torch.tensor(custom.data.T))
    return reference, mappings


def compare_transformer_block(rng):
    
    largest_error = 0.0
    for shape, hidden_dim, eps in (((3, 4), None, 1e-5), ((2, 3, 4), 7, 1e-3), ((2, 1, 4), None, 1e-5)):
        block = TransformerBlock(4, num_heads=2, hidden_dim=hidden_dim, eps=eps, rng=rng)
        reference, mappings = torch_block_reference(block)
        assert len(block.parameters()) == 16

        x = Tensor(rng.normal(size=shape), requires_grad=True)
        torch_x = torch.tensor(x.data, requires_grad=True)
        output = block(x)
        mask = torch.ones(shape[-2], shape[-2], dtype=torch.bool).triu(1)
        expected_output = reference(torch_x, src_mask=mask)
        assert output.shape == shape
        np.testing.assert_allclose(
            output.data, expected_output.detach().numpy(), rtol=1e-9, atol=1e-10
        )
        weights = rng.normal(size=shape)
        (output * weights).sum().backward()
        (expected_output * torch.tensor(weights)).sum().backward()
        pairs = [(x.grad, torch_x.grad.numpy())]
        pairs.extend((custom.grad, expected.grad[rows].numpy().T) for custom, expected, rows in mappings)
        for actual, target in pairs:
            np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(actual - target))))
        check_causality(block, x, output, weights, rng)

        
        for layer in (block.attention.output, block.feed_forward.projection):
            layer.weight.data.fill(0.0)
            layer.bias.data.fill(0.0)
        block.zero_grad()
        x.grad.fill(0.0)
        identity = block(x)
        np.testing.assert_array_equal(identity.data, x.data)
        (identity * weights).sum().backward()
        np.testing.assert_allclose(x.grad, weights, rtol=1e-9, atol=1e-10)

    print(f"transformer block        gradient error: {largest_error:.2e}")


def compare_gpt(rng):
    
    largest_error = 0.0
    sequences = (
        np.array([1, 1, 2, 3, 4]),
        np.array([[1, 2, 1, 4], [3, 1, 3, 2]]),
        np.array([[2, 2]]),
    )
    for sequence in sequences:
        ids, targets = sequence[..., :-1], sequence[..., 1:]
        model = GPT(7, 4, embedding_dim=4, num_heads=2, num_layers=2, rng=rng)
        token_embedding = torch.nn.Embedding(7, 4, dtype=torch.float64)
        position_embedding = torch.nn.Embedding(4, 4, dtype=torch.float64)
        norm = torch.nn.LayerNorm(4, dtype=torch.float64)
        pairs = (
            (model.token_embedding.weight, token_embedding.weight),
            (model.position_embedding.weight, position_embedding.weight),
            (model.norm.weight, norm.weight),
            (model.norm.bias, norm.bias),
        )
        with torch.no_grad():
            for custom, expected in pairs:
                expected.copy_(torch.tensor(custom.data))
        references, mappings = [], []
        for block in model.blocks:
            reference, block_mappings = torch_block_reference(block)
            references.append(reference)
            mappings.extend(block_mappings)
        parameters = model.parameters()
        assert len(parameters) == len(pairs) + len(mappings)
        assert set(parameters) == (
            {custom for custom, _ in pairs} | {custom for custom, _, _ in mappings}
        )

        output = model(ids)
        torch_ids = torch.tensor(ids, dtype=torch.long)
        expected = token_embedding(torch_ids) + position_embedding(torch.arange(ids.shape[-1]))
        mask = torch.ones(ids.shape[-1], ids.shape[-1], dtype=torch.bool).triu(1)
        for reference in references:
            expected = reference(expected, src_mask=mask)
        expected = norm(expected) @ token_embedding.weight.T
        assert output.shape == ids.shape + (7,)
        np.testing.assert_allclose(output.data, expected.detach().numpy(), rtol=1e-9, atol=1e-10)

        loss = CrossEntropyLoss()(output, targets)
        expected_loss = torch.nn.functional.cross_entropy(
            expected.reshape(-1, 7), torch.tensor(targets.reshape(-1), dtype=torch.long)
        )
        np.testing.assert_allclose(loss.data, expected_loss.detach().numpy(), rtol=1e-9, atol=1e-10)
        loss.backward()
        expected_loss.backward()
        gradients = [(custom.grad, expected.grad.numpy()) for custom, expected in pairs]
        gradients.extend((custom.grad, expected.grad[rows].numpy().T) for custom, expected, rows in mappings)
        for actual, target in gradients:
            np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-10)
            largest_error = max(largest_error, float(np.max(np.abs(actual - target))))

        before = model.token_embedding.weight.data.copy()
        SGD(parameters, lr=0.01).step()
        np.testing.assert_allclose(
            model.token_embedding.weight.data,
            before - 0.01 * token_embedding.weight.grad.numpy(), rtol=1e-9, atol=1e-10,
        )
        output = model(ids)
        for stop in range(1, ids.shape[-1]):
            changed = ids.copy()
            changed[..., stop:] = (changed[..., stop:] + 1) % model.vocab_size
            np.testing.assert_allclose(
                model(changed).data[..., :stop, :], output.data[..., :stop, :],
                rtol=1e-9, atol=1e-10,
            )
            model.zero_grad()
            CrossEntropyLoss()(model(ids)[..., :stop, :], targets[..., :stop]).backward()
            np.testing.assert_array_equal(model.position_embedding.weight.grad[stop:], 0.0)

    print(f"GPT (tied weights)       logits, loss, gradients + shared update passed, error: {largest_error:.2e}")


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
    compare_embedding(rng)
    compare_layer_norm(rng)
    for num_heads in (1, 2, 4):
        compare_self_attention(rng, num_heads=num_heads)
        compare_self_attention(rng, causal=True, num_heads=num_heads)
    compare_feed_forward(rng)
    compare_transformer_block(rng)
    compare_gpt(rng)

    print(f"all comparisons passed using PyTorch {torch.__version__}.")


if __name__ == "__main__":
    main()
