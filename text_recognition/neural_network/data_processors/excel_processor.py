from .abstract_data_processor import DataProcessor, Normalize
import pandas as pd
from neural_network.utils import Cache


class ExcelProcessor(DataProcessor):
    def __init__(self, filename, y_column, delimiter=","):
        self.x_dataset = pd.read_csv(filename, delimiter=delimiter)
        self.y_dataset = self.x_dataset.pop(y_column)

    def normilize(self, normalization_class: Normalize):
        self.x_dataset: pd.DataFrame = normalization_class.normilize(self.x_dataset)

    def to_vector(self) -> Cache:
        return Cache(
            x=self.x_dataset.values.astype(float).T,
            y=self.y_dataset.values.astype(float),
        )
