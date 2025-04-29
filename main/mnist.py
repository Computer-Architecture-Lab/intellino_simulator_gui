import numpy as np
import math
import time
import cv2
import sys
import os
import pickle
from torchvision import datasets
from intellino.core.neuron_cell import NeuronCells
from torch.utils.data import DataLoader


#-----------------------------------intellino train--------------------------------#


number_of_neuron_cells = 1000
length_of_input_vector = 784

resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")


# dataset
train_mnist = datasets.MNIST('../mnist_data/', download=True, train=True)
test_mnist = datasets.MNIST("../mnist_data/", download=True, train=False)
                            

def train():
    dataloader = DataLoader(train_mnist, shuffle=True)
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

def preprocess_user_image(image_path):
    # 1. 이미지 읽기 (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to mount image.")

    # 2. 이미지 크기 조정
    image = cv2.resize(image, (resize_size, resize_size))

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 픽셀 반전 (흰 배경일 경우 평균 밝기 기준)
    if np.mean(image) > 127:
        image = 255 - image  # MNIST 스타일로 맞춤

    # 4. 디버깅 이미지 저장
    cv2.imwrite("preprocessed_debug.png", image)

    # 5. 평탄화 (flatten)
    return image.reshape(1, -1).squeeze()



def infer():
    if not os.path.exists("trained_neuron.pkl"):
        print("[ERROR] There is no learned model.Run learning first.", flush=True)
        return

    with open("trained_neuron.pkl", "rb") as f:
        neuron_cells_loaded = pickle.load(f)

    # ------------------------------
    # [NEW] 외부 이미지로부터 추론
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(f"[ERROR] File does not exist: {image_path}", flush=True)
            return

        try:
            flatten_image = preprocess_user_image(image_path)
            predict_label = neuron_cells_loaded.inference(vector=flatten_image)
            print(f"Input Image Inference Results: predict_label = {predict_label}", flush=True)
            return
        except Exception as e:
            print(f"[ERROR] Pre-processing failed: {e}", flush=True)
            return

    # ------------------------------
    # 기본 내장 MNIST 테스트셋으로 추론 (예전 방식)
    for i, (data, label) in enumerate(test_mnist):
        if i < 6:
            numpy_image = np.array(data)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
            flatten_image = resized_image.reshape(1, -1).squeeze()
            predict_label = neuron_cells_loaded.inference(vector=flatten_image)
            print(f"label : {label}, predict_label : {predict_label}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer()
    else:
        train()