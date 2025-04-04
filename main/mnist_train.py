import numpy as np
import math
import time
import cv2
from torchvision import datasets
from intellino.core.neuron_cell import NeuronCells


#-----------------------------------intellino train--------------------------------#


number_of_neuron_cells = 100
length_of_input_vector = 256

resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")



train_mnist = datasets.MNIST('../mnist_data/', download=True, train=True)

for i, (data, label) in enumerate(train_mnist):
    numpy_image = np.array(data)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
    flatten_image = resized_image .reshape(1, -1).squeeze()
    is_finish = neuron_cells.train(vector=flatten_image, target=label)

    # 학습 진행률
    if __name__ == "__main__":
        progress = int((i)/number_of_neuron_cells * 100)
        if i%4==0:
            print(f"progress : {progress}", flush=True)     # flush=True : 출력 내용을 바로 콘솔로 내보내게 함
            time.sleep(0.01)

        if is_finish == True:
            print("train finish")
            break



