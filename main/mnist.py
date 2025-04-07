import numpy as np
import math, time, cv2, sys, os, pickle
from torchvision import datasets
from intellino.core.neuron_cell import NeuronCells


#-----------------------------------intellino train--------------------------------#


number_of_neuron_cells = 100
length_of_input_vector = 256

resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")


# dataset
train_mnist = datasets.MNIST('../mnist_data/', download=True, train=True)
test_mnist = datasets.MNIST("../mnist_data/", download=True, train=False)
                            

def train():
    for i, (data, label) in enumerate(train_mnist):
        numpy_image = np.array(data)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image .reshape(1, -1).squeeze()
        is_finish = neuron_cells.train(vector=flatten_image, target=label)

        # 학습 진행률
        progress = int((i)/number_of_neuron_cells * 100)
        if i%4==0:
            print(f"progress : {progress}", flush=True)     # flush=True : 출력 내용을 바로 콘솔로 내보내게 함
            time.sleep(0.01)

        if is_finish == True:
            print("train finish")
            with open("trained_neuron.pkl","wb") as f:
                pickle.dump(neuron_cells, f)
            break


#-----------------------------------intellino test---------------------------------#


def infer():

    if not os.path.exists("trained_neuron.pkl"):
        print("[ERROR] 학습된 모델이 없습니다. 먼저 학습을 실행하세요.", flush=True)
        return
    
    with open("trained_neuron.pkl", "rb") as f:
        neuron_cells_loaded = pickle.load(f)

    for i, (data, label) in enumerate(test_mnist):
        if i < 6 :
            numpy_image = np.array(data)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
            flatten_image = resized_image .reshape(1,-1).squeeze()
            predict_label = neuron_cells_loaded.inference(vector=flatten_image)
            print(f"label : {label}, predict_label : {predict_label}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer()
    else:
        train()