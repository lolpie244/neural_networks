from .abstract_data_processor import Normalize
import pandas as pd


class StructuredDataNormilize(Normalize):
    def __init__(self, normilize_coefs=None) -> None:
        self.normilize_coefs = normilize_coefs

    def normilize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if not self.normilize_coefs:
            numerical = dataset.select_dtypes(include=["float64", "int64"]).columns
            self.numerical_coefs = {column: (dataset[column].min(), dataset[column].max()) for column in numerical}

        new_dataset = pd.get_dummies(
            dataset,
            columns=[column for column in dataset.columns if column not in self.numerical_coefs],
        )

        for column, min_max in self.numerical_coefs.items():
            min, max = min_max
            new_dataset[column] = (dataset[column] - min) / (max - min)

        return new_dataset
