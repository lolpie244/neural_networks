import numpy as np
from .abstract_layer import Layer


class CnnLayser(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return super().forward(input_data)

    def backward(self, input_data: np.ndarray, alpha: float) -> np.ndarray:
        return super().backward(input_data, alpha)
