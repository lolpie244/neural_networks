from neural_network.layers import Layer
from neural_network.functions import Function


class LayersConteiner:
    def __init__(self, *args: Layer) -> None:
        self.layers_list = args

    def train(self, X, Y, loss_function: Function, iterations=10000, alpha=0.001):
        for _ in range(iterations):
            current_activation = X

            for layer in self.layers_list:
                current_activation = layer.forward(current_activation)

            dA = loss_function.derivative(current_activation, Y)

            for layer in self.layers_list[::-1]:
                dA = layer.backward(dA, alpha)

    def predict(self, X):
        current_activation = X

        for layer in self.layers_list:
            current_activation = layer.forward(current_activation)

        return current_activation

