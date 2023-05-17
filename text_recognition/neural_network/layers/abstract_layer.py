from abc import ABC, abstractmethod
from neural_network.activation_functions import ActivationFunction
import numpy as np


class Layer(ABC):
    def __init__(self, activation_function: ActivationFunction, size: int | tuple) -> None:
        self.activation_function = activation_function
        self.size = size

        if isinstance(size, int):
            size = (size, 1)

        self.weights: np.ndarray = np.random.rand(*size) # type: ignore
        self.bias = np.random.rand(1, *size[1:])

    @abstractmethod
    def forward(self, input_data: np.ndarray):
        return self.activation_function.function(np.dot(self.weights.T, input_data) + self.bias)

    @abstractmethod
    def backward(self, input_data: np.ndarray):
        pass
