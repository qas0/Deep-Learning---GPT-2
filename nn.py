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
