from typing import Optional
import numpy as np

from .abstract_layer import Layer
from neural_network.functions import Function
from neural_network.utils import Cache


class ClasificationLayer(Layer):
    def __init__(self, neurons_count: int, activation_function: Function) -> None:
        self.activation_function = activation_function
        self.neurons_count = neurons_count

        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        self.cache = Cache(
            linear=np.ndarray(0),
            input_data=np.ndarray(0),
        )

    def initialize_weights(self, input_data_count):
        self.weights: np.ndarray = np.random.rand(self.neurons_count, input_data_count)
        self.bias = np.random.rand(self.neurons_count, 1)

    def forward(self, input_data: np.ndarray):
        if self.weights is None:
            self.initialize_weights(input_data.shape[0])

        linear = np.dot(self.weights, input_data) + self.bias

        self.cache.update(
            linear=linear,
            input_data=input_data,
        )
        return self.activation_function(linear)

    def backward(self, input_data: np.ndarray, alpha):
        m = input_data.shape[1]

        dz = input_data * self.activation_function.derivative(self.cache.linear)
        dw = np.dot(dz, self.cache.input_data.T)
        db = np.sum(dz, axis=1, keepdims=True)

        dA = np.dot(self.weights.T, dz)

        self.weights -= alpha * dw
        self.bias -= alpha * db

        return dA

