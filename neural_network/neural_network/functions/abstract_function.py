from abc import ABC, abstractmethod
import numpy as np


class Function(ABC):

    @abstractmethod
    def __call__(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, *args) -> np.ndarray:
        pass

