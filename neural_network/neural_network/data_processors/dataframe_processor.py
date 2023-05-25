from neural_network.utils.cache import Cache
from .abstract_data_processor import DataProcessor, Normalize
import pandas as pd
from neural_network.utils import Cache


class DataFrameProcessor(DataProcessor):
    def __init__(self, x_dataset, y_dataset):
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset

    @classmethod
    def from_file(cls, filename, y_column, delimiter=","):
        x_dataset = pd.read_csv(filename, delimiter=delimiter)
        y_dataset = x_dataset.pop(y_column)
        return DataFrameProcessor(x_dataset, y_dataset)

    def normilize(self, x_normalization_class: Normalize, y_normalization_class: Normalize | None = None):
        self.x_dataset: pd.DataFrame = x_normalization_class.normilize(self.x_dataset) 

        if y_normalization_class is not None:
            self.y_dataset = y_normalization_class.normilize(self.y_dataset)

        return Cache(
            x=self.x_dataset.values.astype(float).T,
            y=self.y_dataset.values.astype(float).T,
        )
