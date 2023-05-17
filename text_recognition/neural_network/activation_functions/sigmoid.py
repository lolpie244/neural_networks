from abstract_function import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):
    @classmethod
    def function(cls, neurons: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-neurons))

    @classmethod
    def derivative(cls, neurons: np.ndarray) -> np.ndarray:
        value = cls.function(neurons)
        return value * (1 - value)

