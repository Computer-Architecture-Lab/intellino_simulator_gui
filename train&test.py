import numpy as np
import math
import cv2
from torchvision import datasets
from intellino.core.neuron_cell import NeuronCells
from torch.utils.data import DataLoader

train_mnist = datasets.MNIST("../mnist_data/",
                             download=True,
                             train=True)
test_mnist = datasets.MNIST("../mnist_data/",
                            download=True,
                            train=False)

number_of_neuron_cells = 1000
length_of_input_vector = 256
resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")

for data, label in train_mnist:
    dataloader = DataLoader(train_mnist, shuffle=True)
    numpy_image = np.array(data)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
    flatten_image = resized_image.reshape(1, -1).squeeze()
    is_finish = neuron_cells.train(vector=flatten_image, target=label)
    if is_finish == True:
        break

for idx, (data, label) in enumerate(test_mnist):
    if idx > 20:
        break
    numpy_image = np.array(data)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
    flatten_image = resized_image.reshape(1, -1).squeeze()
    predict_label = neuron_cells.inference(vector=flatten_image)
    print(f"label : {label}, predict_label : {predict_label}")