from math import isfinite

import numpy as np


class SGD:
    """updates parameters using their gradients + learning rate"""

    def __init__(self, parameters, lr=0.01):
        if not isfinite(lr) or lr < 0:
            raise ValueError("learning rate must be finite and nonnegative")

        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("optimiser needs at least one parameter")
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.fill(0.0)

    def step(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                # updates stored values directly so the update isnt added to the graph
                parameter.data -= self.lr * parameter.grad


class Adam:
    """updates parameters using running averages of gradients + squared gradients"""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        if not isfinite(lr) or lr < 0:
            raise ValueError("learning rate must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two values between 0 and 1, excluding 1")
        if not isfinite(eps) or eps <= 0:
            raise ValueError("epsilon must be finite and positive")

        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("optimiser needs at least one parameter")
        self.lr = lr
        self.betas = tuple(betas)
        self.eps = eps
        self.first_moments = [np.zeros_like(p.data) for p in self.parameters]
        self.second_moments = [np.zeros_like(p.data) for p in self.parameters]
        self.steps = [0 for p in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.fill(0.0)

    def step(self):
        beta1, beta2 = self.betas
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue

            # counts updates separately in case a parameter has no grad
            self.steps[index] += 1
            timestep = self.steps[index]
            gradient = parameter.grad
            first = self.first_moments[index]
            second = self.second_moments[index]
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * gradient**2

            # corrects the averages starting at 0 which makes early values too small
            corrected_first = first / (1 - beta1**timestep)
            corrected_second = second / (1 - beta2**timestep)
            parameter.data -= (
                self.lr * corrected_first / (np.sqrt(corrected_second) + self.eps)
            )
