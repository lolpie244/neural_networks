import numpy as np

from .abstract_layer import Layer
from neural_network.activation_functions import ActivationFunction
from neural_network.utils import Cache


class ClasificationLayer(Layer):
    def __init__(self, activation_function: ActivationFunction, neurons_count: int) -> None:
        self.activation_function = activation_function
        self.neurons_count = neurons_count

        self.weights: np.ndarray
        self.bias: float
        self.cache = Cache(
            linear=np.ndarray(0),
            input_data=np.ndarray(0),
        )

    def initialize_weights(self, input_data_count):
        self.weights: np.ndarray = np.random.rand(self.neurons_count, input_data_count)
        self.bias = np.random.rand()

    def get_linear(self, input_data):
        return np.dot(self.weights.T, input_data) + self.bias

    def forward(self, input_data: np.ndarray):
        if self.weights is None:
            self.initialize_weights(input_data.shape[0])

        linear = self.get_linear(input_data)

        self.cache.update(
            linear=linear,
            input_data=input_data,
        )
        return self.activation_function.function(linear)

    def backward(self, input_data: np.ndarray, alpha):
        m = input_data.shape[1]

        dz = input_data * self.activation_function.derivative(self.cache.linear)
        dw = np.dot(dz, self.cache.input_data.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        dA = np.dot(self.weights.T, dz)

        self.weights = self.weights - alpha * dw
        self.bias = self.bias - alpha * db

        return dA
