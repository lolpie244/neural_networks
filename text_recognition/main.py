from neural_network.data_processors.normalization import StructuredDataNormilize
from neural_network.layers_container import LayersConteiner
from neural_network.functions.activations import Relu, Sigmoid
from neural_network.functions.errors import MeanSquared
from neural_network.layers import ClasificationLayer
from neural_network.data_processors import DataFrameProcessor

from sk


neural_network = LayersConteiner(
    ClasificationLayer(5, Relu()),
    ClasificationLayer(3, Relu()),
    ClasificationLayer(3, Sigmoid()),
)

dataset_processor = DataFrameProcessor.from_file("dataset.csv", "test")

dataset = dataset_processor.normilize(StructuredDataNormilize())

neural_network.train(dataset.x, dataset.y, MeanSquared())
