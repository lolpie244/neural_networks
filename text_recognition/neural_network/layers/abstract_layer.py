from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, input_data: np.ndarray):
        pass

    @abstractmethod
    def backward(self, input_data: np.ndarray):
        pass
