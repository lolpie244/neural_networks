from neural_network.functions import Function
import numpy as np


class Relu(Function):
    def __call__(self, neurons: np.ndarray) -> np.ndarray:
        return np.maximum(0, neurons)

    def derivative(self, neurons: np.ndarray) -> np.ndarray:
        return np.where(neurons < 0, 0, 1)


class Sigmoid(Function):
    def __call__(self, neurons: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-neurons))

    def derivative(self, neurons: np.ndarray) -> np.ndarray:
        value = self(neurons)
        return value * (1 - value)

