from abc import ABC, abstractmethod
from typing import Any

from neural_network.utils.cache import Cache


class Normalize(ABC):
    @abstractmethod
    def normilize(self, dataset) -> Any:
        pass


class DataProcessor(ABC):
    @abstractmethod
    def normilize(self):
        pass

