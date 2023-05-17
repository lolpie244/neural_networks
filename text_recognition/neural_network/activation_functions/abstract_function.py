from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):

    @abstractmethod
    @classmethod
    def function(cls, neurons: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    @classmethod
    def derivative(cls, neurons: np.ndarray) -> np.ndarray:
        pass

