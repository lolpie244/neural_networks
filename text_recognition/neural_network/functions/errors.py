import numpy as np

from neural_network.functions import Function


class CategoricalCrossEntropy(Function):
    def __call__(self, y_pred, y_real) -> np.ndarray:
        return -np.sum(y_real * np.log(y_pred))

    def derivative(self, y_pred, y_real) -> np.ndarray:
        return (y_pred - y_real) / y_pred.shape[0]


class MeanSquared(Function):
    def __call__(self, y_pred, y_real) -> np.ndarray:
        return np.mean(np.power((y_real - y_pred), 2), axis=0)

    def derivative(self, y_pred, y_real) -> np.ndarray:
        return 2 * (y_pred - y_real) / len(y_real)


class BinaryCrossEntropy(Function):
    def __call__(self, y_pred, y_real) -> np.ndarray:
        return np.mean(-(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)), axis=1)

    def derivative(self, y_pred, y_real) -> np.ndarray:
        return (y_pred - y_real) / (y_pred * (1 - y_pred))
