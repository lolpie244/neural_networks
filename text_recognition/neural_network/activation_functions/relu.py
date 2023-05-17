from abstract_function import ActivationFunction
import numpy as np


class Relu(ActivationFunction):
    @classmethod
    def function(cls, neurons: np.ndarray) -> np.ndarray:
        return np.max(0, neurons)

    @classmethod
    def derivative(cls, neurons: np.ndarray) -> np.ndarray:
        return np.where(neurons < 0, 0, 1)

