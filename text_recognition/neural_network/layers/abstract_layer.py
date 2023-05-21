from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self) -> None:
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        self.cache = Cache()

        super().__init__()

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, input_data: np.ndarray, alpha: float) -> np.ndarray:
        pass
