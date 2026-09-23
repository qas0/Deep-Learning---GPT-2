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


class ReLU(Module):
    """applies ReLU to each value, keeping positive values + replacing negatives with 0"""

    def forward(self, x):
        return x.relu()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


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
