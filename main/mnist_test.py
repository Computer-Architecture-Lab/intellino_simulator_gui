import cv2
import numpy as np
import math
from torchvision import datasets
from mnist_train import neuron_cells


number_of_neuron_cells = 100
length_of_input_vector = 256
resize_size = int(math.sqrt(length_of_input_vector))


test_mnist = datasets.MNIST("../mnist_data/",
                            download=True,
                            train=False)

for i, (data, label) in enumerate(test_mnist):
    if i < 6 :
        numpy_image = np.array(data)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image .reshape(1,-1).squeeze()
        predict_label = neuron_cells.inference(vector=flatten_image)
        print(f"label : {label}, predict_label : {predict_label}", flush=True)


