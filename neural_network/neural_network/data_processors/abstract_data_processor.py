from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np

from neural_network.utils.cache import Cache


class Normalize(ABC):
    @abstractmethod
    def normilize(self, dataset) -> Any:
        pass


class DataProcessor(ABC):
    @abstractmethod
    def normilize(self, x_normilize, y_normalize) -> Cache:
        pass

    @classmethod
    def split_training(cls, cache, training_size, randomize=True) -> Tuple[Cache, Cache]:
        x, y = cache.x, cache.y

        m = int(x.shape[1] * training_size)

        if randomize:
            perm = np.random.permutation(x.shape[1])
            x = x[:, perm]
            y = y[:, perm]

        training = Cache(
            x=x[:, :m],
            y=y[:, :m],
        )

        test = Cache(
            x=x[:, m:],
            y=y[:, m:],
        )

        return (training, test)
