import numpy as np

from tensor import Tensor


class Parameter(Tensor):
    """stores values that get updated during training"""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter({self.data!r})"


class Module:
    """neural network base class"""

    def parameters(self):
        """returns parameters"""
        parameters = []
        visited_parameters = set()
        visited_modules = set()

        def collect(value):
            if isinstance(value, Parameter):
                identity = id(value)
                if identity not in visited_parameters:
                    visited_parameters.add(identity)
                    parameters.append(value)
            elif isinstance(value, Module):
                identity = id(value)
                if identity in visited_modules:
                    return

                visited_modules.add(identity)
                for child in vars(value).values():
                    collect(child)
            elif isinstance(value, (list, tuple, set)):
                for child in value:
                    collect(child)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)

        # recursion is used here to find parameters inside modules + containers
        collect(self)
        return parameters

    def zero_grad(self):
        """sets parameter gradients to 0"""
        for parameter in self.parameters():
            parameter.grad.fill(0.0)

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Module):
    """transforms final input dimension using learned weights + bias"""

    def __init__(self, in_features, out_features, rng=None):
        for size in (in_features, out_features):
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
                raise TypeError("feature counts must be integers")
            if size <= 0:
                raise ValueError("feature counts must be positive")

        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng() if rng is None else rng

        # scales random weights by input width to limit initial output size
        limit = 1 / np.sqrt(in_features)
        self.weight = Parameter(
            rng.uniform(-limit, limit, size=(in_features, out_features))
        )
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, x):
        """map shape (..., in_features) to (..., out_features)"""
        if not x.shape or x.shape[-1] != self.in_features:
            raise ValueError(f"expected final input dimension {self.in_features}")

        
        return x @ self.weight + self.bias


class Embedding(Module):
    """maps integer IDs shaped (...) to vectors shaped (..., embedding_dim)"""

    def __init__(self, num_embeddings, embedding_dim, rng=None):
        for size in (num_embeddings, embedding_dim):
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
                raise TypeError("embedding sizes must be integers")
            if size <= 0:
                raise ValueError("embedding sizes must be positive")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng() if rng is None else rng
        self.weight = Parameter(
            rng.normal(0.0, 0.02, size=(num_embeddings, embedding_dim))
        )

    def forward(self, token_ids):
        # copies IDs so changing the input later doesnt redirect gradients
        token_ids = np.array(token_ids, copy=True)
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise TypeError("token IDs must be integers")
        if np.any(token_ids < 0) or np.any(token_ids >= self.num_embeddings):
            raise ValueError("token ID is out of range")

        return self.weight[token_ids]


class LayerNorm(Module):
    """normalises the final feature dimension with learned scale + bias"""

    def __init__(self, num_features, eps=1e-5):
        if isinstance(num_features, bool) or not isinstance(num_features, (int, np.integer)):
            raise TypeError("feature count must be an integer")
        if num_features <= 0:
            raise ValueError("feature count must be positive")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))

    def forward(self, x):
        if not x.shape or x.shape[-1] != self.num_features:
            raise ValueError(f"expected final input dimension {self.num_features}")

        # keeps each tokens statistics separate from the other tokens
        centred = x - x.mean(axis=-1, keepdims=True)
        variance = (centred**2).mean(axis=-1, keepdims=True)
        # eps stops division by zero when the features have no variation
        normalised = centred / (variance + self.eps)**0.5
        return normalised * self.weight + self.bias


class SelfAttention(Module):
    """mixes token information across attention heads"""

    def __init__(self, embedding_dim, rng=None, causal=False, num_heads=1):
        if isinstance(num_heads, bool) or not isinstance(num_heads, (int, np.integer)):
            raise TypeError("head count must be an integer")
        if num_heads <= 0:
            raise ValueError("head count must be positive")
        rng = np.random.default_rng() if rng is None else rng
        self.query = Linear(embedding_dim, embedding_dim, rng=rng)
        self.key = Linear(embedding_dim, embedding_dim, rng=rng)
        self.value = Linear(embedding_dim, embedding_dim, rng=rng)
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding dimension must be divisible by head count")
        self.output = Linear(embedding_dim, embedding_dim, rng=rng)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.causal = causal

    def forward(self, x):
        if len(x.shape) not in (2, 3) or x.shape[-2] == 0:
            raise ValueError("expected a sequence or batch with at least one token")

        batch = x.shape[0] if len(x.shape) == 3 else 1
        tokens = x.shape[-2]

        def split_heads(values):
            # gives each head its own features: (batch, heads, tokens, head_dim)
            return values.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = split_heads(self.query(x))
        keys = split_heads(self.key(x))
        values = split_heads(self.value(x))

        # swaps token and feature axes keeping batches and heads separate
        transposed_keys = keys.transpose(0, 1, 3, 2)
        # scales scores so larger feature counts dont make softmax too sharp
        scores = (queries @ transposed_keys) / self.head_dim**0.5
        if self.causal:
            # future scores become -inf so their softmax weights are 0
            mask = np.triu(np.full((tokens, tokens), -np.inf), k=1)
            scores = scores + mask
        weights = scores.softmax(axis=-1)
        # joins the heads back into each tokens feature vector
        combined = (weights @ values).transpose(0, 2, 1, 3).reshape(x.shape)
        return self.output(combined)


class ReLU(Module):
    """applies ReLU to each value, keeping positive values + replacing negatives with 0"""

    def forward(self, x):
        return x.relu()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class FeedForward(Module):
    # expands each tokens features, applies GELU + projects back

    def __init__(self, embedding_dim, hidden_dim=None, rng=None):
        hidden_dim = 4 * embedding_dim if hidden_dim is None else hidden_dim
        rng = np.random.default_rng() if rng is None else rng
        self.expansion = Linear(embedding_dim, hidden_dim, rng=rng)
        self.activation = GELU()
        self.projection = Linear(hidden_dim, embedding_dim, rng=rng)

    def forward(self, x):
        return self.projection(self.activation(self.expansion(x)))


class Softmax(Module):
    """applies softmax along axis to turn scores into probabilities"""

    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        return x.softmax(axis=self.axis)


class CrossEntropyLoss(Module):
    """averages loss for logits shaped (..., classes) + target class indices shaped (...)"""

    def forward(self, logits, targets):
        if not logits.shape or logits.data.size == 0:
            raise ValueError("logits must be nonempty and have a class dimension")

        targets = np.asarray(targets)
        if targets.shape != logits.shape[:-1]:
            raise ValueError("target shape must match logits without the class axis")
        if not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("targets must contain integer class indices")

        classes = logits.shape[-1]
        if np.any(targets < 0) or np.any(targets >= classes):
            raise ValueError("target class index is out of range")

        log_probabilities = logits.log_softmax().reshape(-1, classes)
        rows = np.arange(targets.size)
        return -log_probabilities[rows, targets.reshape(-1)].mean()
