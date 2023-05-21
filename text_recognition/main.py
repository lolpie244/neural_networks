from neural_network.data_processors.normalization import DataFrameNormalize, OnlyDummyNormalize
from neural_network.layers_container import LayersConteiner
from neural_network.functions.activations import Relu, Sigmoid
from neural_network.functions.errors import CrossEntropy
from neural_network.layers import ClasificationLayer
from neural_network.data_processors import DataFrameProcessor


from sklearn.datasets import load_iris


neural_network = LayersConteiner(
    ClasificationLayer(10, Sigmoid()),
    ClasificationLayer(10, Sigmoid()),
    ClasificationLayer(3, Sigmoid()),
)


X, Y = load_iris(return_X_y=True, as_frame=True)

# нормалізація даних
dataset_processor = DataFrameProcessor(X, Y)
dataset = dataset_processor.normilize(DataFrameNormalize(), OnlyDummyNormalize())

# розділювання на тестові
training_dataset, test_dataset = dataset_processor.split_training(dataset, 0.8)


neural_network.train(training_dataset.x, training_dataset.y, CrossEntropy(), alpha=0.01)
results = neural_network.predict(test_dataset.x)

print(*map(list, test_dataset.y), "#"*100, sep='\n')
row_format = "{:^4}|" * len(test_dataset.y[0])
for x in results:
    print(f"[{row_format.format(*map(lambda k: str(k), x))}]")
