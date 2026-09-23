import numpy as np


def _sum_to_shape(gradient, shape):
    """sums gradient back to the inputs original shape"""
    # sums over the extra dimensions that were added by broadcasting
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)

    # dimensions that were expanded from size 1 during forward pass get summed over
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)

    return gradient


class Tensor:
    """stores values + tracks operations to calculate gradients"""

    def __init__(self, data, requires_grad=False, _children=()):
        # copies the data so changes to the original array wont affect this tensor
        self.data = np.array(data, dtype=np.float64, copy=True)

        # stores the gradient and the links needed to traverse the graph backwards. 
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._children = set(_children)
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data!r})"

    # elementwise arithmetic operations

    def __add__(self, other):
        """adds a tensor or numeric value using NumPy to broadcast inputs (if needed)"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            self.data + other.data,
            requires_grad=requires_grad,
            _children=(self, other),
        )

        # the derivative of addition is 1 with respect to either input
        def _backward():
            if self.requires_grad:
                self.grad += _sum_to_shape(result.grad, self.shape)
            if other.requires_grad:
                other.grad += _sum_to_shape(result.grad, other.shape)

        result._backward = _backward
        return result

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        """negate every element"""
        result = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self.grad -= result.grad

        result._backward = _backward
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other):
        """multiplies each value using NumPy broadcasting (if needed)"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            self.data * other.data,
            requires_grad=requires_grad,
            _children=(self, other),
        )

        # mulitiplies the upstream gradient by the other input
        def _backward():
            if self.requires_grad:
                gradient = result.grad * other.data
                self.grad += _sum_to_shape(gradient, self.shape)
            if other.requires_grad:
                gradient = result.grad * self.data
                other.grad += _sum_to_shape(gradient, other.shape)

        result._backward = _backward
        return result

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        """raises every value to a given numeric power""" 
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be an int or float")

        result = Tensor(
            self.data**exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        # for y = x^n, the local derivative is n * x^(n - 1), uses power rule 
        def _backward():
            if self.requires_grad and exponent != 0:
                self.grad += (
                    exponent * self.data ** (exponent - 1) * result.grad
                )

        result._backward = _backward
        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    # activation functions

    def relu(self):
        """keeps positive values + replaces negative values with 0"""
        positive = self.data > 0
        result = Tensor(
            np.maximum(self.data, 0),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                # blocks incoming gradient if inputs consist of negative & 0
                self.grad += result.grad * positive

        result._backward = _backward
        return result

    def gelu(self):
        """applies tanh approximation of GELU""" 
        x = self.data
        scale = np.sqrt(2 / np.pi)
        tanh_value = np.tanh(scale * (x + 0.044715 * x**3))
        result = Tensor(
            0.5 * x * (1 + tanh_value),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                # uses product rule for x * (1 + tanh(u)) + chain rule through u
                inner_gradient = scale * (1 + 3 * 0.044715 * x**2)
                local_gradient = 0.5 * (1 + tanh_value) + (
                    0.5 * x * (1 - tanh_value**2) * inner_gradient
                )
                self.grad += result.grad * local_gradient

        result._backward = _backward
        return result

    # probabilities and log probabilities

    def softmax(self, axis=-1):
        """converts scores to probabilities that sum to 1 along chosen axis"""
        # subtracts the largest score to stop exp overflowing without changing the probabilities
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=axis, keepdims=True)
        result = Tensor(
            probabilities,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                weighted_gradient = (result.grad * probabilities).sum(
                    axis=axis, keepdims=True
                )
                self.grad += probabilities * (result.grad - weighted_gradient)

        result._backward = _backward
        return result

    def log_softmax(self, axis=-1):
        """calculates log probabilities directly to avoid taking log of zero"""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_total = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        log_probabilities = shifted - log_total
        probabilities = np.exp(log_probabilities)
        result = Tensor(
            log_probabilities,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self.grad += result.grad - probabilities * result.grad.sum(
                    axis=axis, keepdims=True
                )

        result._backward = _backward
        return result

    # handles matrix and shape operations 

    def __matmul__(self, other):
        """performs matrix multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(
            self.data @ other.data,
            requires_grad=requires_grad,
            _children=(self, other),
        )

        # calculates gradients using the incoming gradient and the other inputs transpose
        def _backward():
            left = self.data
            right = other.data
            upstream = result.grad

            left_matrix = left[np.newaxis, :] if left.ndim == 1 else left
            right_matrix = right[:, np.newaxis] if right.ndim == 1 else right

            if left.ndim == 1 and right.ndim == 1:
                upstream_matrix = upstream.reshape(1, 1)
            elif left.ndim == 1:
                upstream_matrix = np.expand_dims(upstream, axis=-2)
            elif right.ndim == 1:
                upstream_matrix = np.expand_dims(upstream, axis=-1)
            else:
                upstream_matrix = upstream

            if self.requires_grad:
                gradient = upstream_matrix @ np.swapaxes(right_matrix, -1, -2)
                if left.ndim == 1:
                    gradient = np.squeeze(gradient, axis=-2)
                self.grad += _sum_to_shape(gradient, self.shape)

            if other.requires_grad:
                gradient = np.swapaxes(left_matrix, -1, -2) @ upstream_matrix
                if right.ndim == 1:
                    gradient = np.squeeze(gradient, axis=-1)
                other.grad += _sum_to_shape(gradient, other.shape)

        result._backward = _backward
        return result

    def reshape(self, *shape):
        """returns the tensor with a new shape without changing its values"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        result = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        # reshape moves no values, so the gradient returns to the old shape
        def _backward():
            if self.requires_grad:
                self.grad += result.grad.reshape(self.shape)

        result._backward = _backward
        return result

    def transpose(self, *axes): 
        """reorders the tensors dimensions"""
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])

        result = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
        )
        axes = tuple(axis % self.data.ndim for axis in axes)
        inverse_axes = tuple(np.argsort(axes))

        # puts the grads dimensions back in their original order
        def _backward():
            if self.requires_grad:
                self.grad += result.grad.transpose(inverse_axes)

        result._backward = _backward
        return result

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        """gets values using an index or slice (NumPy indexing)"""
        result = Tensor(
            self.data[index],
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, index, result.grad) # add.at adds the gradients together if an index is used more than once

        result._backward = _backward
        return result

    # handles sum, mean, and backpropogation

    def sum(self):
        result = Tensor(
            self.data.sum(),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        # every input gets the same incoming grad
        def _backward():
            if self.requires_grad:
                self.grad += result.grad

        result._backward = _backward
        return result

    def mean(self):
        return self.sum() / self.data.size

    def backward(self):
        """works backwards from a single output val to calculate grad"""
        if self.data.size != 1:
            raise RuntimeError("backward() requires a scalar tensor")

        ordered = []
        visited = set()

        # goes through the inputs first, then adds the tensor to the list
        def build_order(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._children:
                    build_order(child)
                ordered.append(tensor)

        build_order(self)

        # clears intermediate gradients while keeping accumulated input grads
        for tensor in ordered:
            if tensor._children and tensor.grad is not None:
                tensor.grad.fill(0.0)

        # starts at 1 because the outputs derivative with respect to itself is 1
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += np.ones_like(self.data)

        # goes through the list backwards
        for tensor in reversed(ordered):
            tensor._backward()
