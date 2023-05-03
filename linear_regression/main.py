import pandas as pd
import numpy as np


class Normalize:
    def __init__(self, dataset: pd.DataFrame, numerical_coefs=None) -> None:
        """
        Save normalization coefficients for numerical columns
        """
        if numerical_coefs:
            self.numerical_coefs = numerical_coefs
        else:
            numerical = dataset.select_dtypes(include=["float64", "int64"]).columns
            self.numerical_coefs = {
                column: (dataset[column].min(), dataset[column].max())
                for column in numerical
            }

    def normilize(self, dataset):
        """
        Normalise dataset:
            For numerical columns:
                new_x = (x - x_min) / (x_max - x_min)

            For no numerical columns:
                | Column_A |     | Column_A | Column_B | Column_C |
                |    A     |     |    1     |    0     |    0     |
                |    B     |  => |    0     |    1     |    0     |
                |    C     |     |    0     |    0     |    1     |
        """
        new_dataset = pd.get_dummies(
            dataset,
            columns=[
                column
                for column in dataset.columns
                if column not in self.numerical_coefs
            ],
        )

        for column, min_max in self.numerical_coefs.items():
            min, max = min_max
            new_dataset[column] = (dataset[column] - min) / (max - min)

        return new_dataset


class TrainingExamples:
    training_examples_percentage = 0.8

    def __init__(self, filename, y_column, delimiter=","):
        """
        Fetch test data from file, saves in format:
            nx         - count of columns in dataset
            m_training - count of training examples
            m_test     - count of training examples

                [x_1(1)  ...   x_1(j)  ...]
                [...     ...   ...     ...]
            x - [x_i(1)  ...   x_i(j)  ...]  (i: i-th training expample)
                [...     ...   ...     ...]
                [x_nx(1) ...   x_nx(j) ...]

            y - [[y(1), y(2), ..., y(i), ... y(m_training)]] (i = i-th training expample)

            x_test, y_test similar to x and y
        """
        x_dataset = pd.read_csv(filename, delimiter=delimiter)
        y_dataset = x_dataset.pop(y_column)

        x_dataset = Normalize(x_dataset).normilize(x_dataset)

        self.m_training = int(x_dataset.shape[0] * self.training_examples_percentage)
        self.m_test = x_dataset.shape[0] - self.m_training

        self.x = x_dataset.head(self.m_training).values.astype(float).T
        self.y = y_dataset.head(self.m_training).values.astype(float)
        self.y = self.y.reshape((self.y.shape[0], 1)).T

        self.x_test = x_dataset.tail(self.m_test).values.astype(float).T
        self.y_test = y_dataset.tail(self.m_test).values.astype(float)

        self.nx = self.x.shape[0]


class LinearRegressionModel:
    def __init__(self, nx, w=None, b=None) -> None:
        """
        w  - weights
        b  - bias
        nx - count of neurons
        """
        self.w = w or np.zeros((nx, 1))
        self.b = b or 0
        self.nx = nx

    def sigmoid(self, z):
        """
        sigmoid function that contains values in range [0, 1]
        """
        return 1 / (1 + np.exp(-z))

    def propagate(self, X, Y):
        """
        direct distraction of gradient descent.

        Using for finding current cost and dw, db (that are used for descent in train method)
        """
        m = X.shape[0]

        y_hat = self.sigmoid(np.dot(self.w.T, X) + self.b)

        cost = np.sum(-(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))) / m

        dz = y_hat - Y
        dw = np.dot(X, dz.T) / m
        db = np.sum(dz) / m

        return dw, db, cost

    def train(self, X_train, Y_train, iterations=10000, alpha=0.001):
        """
        backward distraction of gradient descent

        Using, for descent further down on gradient descent function (for improve result)
        """
        cost = 0
        for _ in range(iterations):
            dw, db, cost = self.propagate(X_train, Y_train)

            self.w = self.w - alpha * dw
            self.b = self.b - alpha * db

        return cost

    def predict(self, X):
        return self.sigmoid(np.dot(self.w.T, X) + self.b)


# Load data and train model
training_data = TrainingExamples("dataset.csv", "diabetes")
model = LinearRegressionModel(training_data.nx)
model.train(training_data.x, training_data.y)


# Count error for test data
result = model.predict(training_data.x_test)
errors_count = 0
for i in range(training_data.m_test):
    errors_count += int(bool(training_data.y_test[i]) != (result[0][i] >= 0.5))

print(errors_count / training_data.m_test * 100, errors_count)
