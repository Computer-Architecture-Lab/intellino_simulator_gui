import numpy as np
import math
import time
import cv2
import sys
import os
import pickle
from tqdm import tqdm
from torchvision import datasets
from intellino.core.neuron_cell import NeuronCells
from torch.utils.data import DataLoader
from collections import defaultdict


#-----------------------------------intellino train--------------------------------#


number_of_neuron_cells = 1000
length_of_input_vector = 784
min_per_label = number_of_neuron_cells // 10

resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")


# dataset
train_mnist = datasets.MNIST('../mnist_data/', download=True, train=True)
test_mnist = datasets.MNIST("../mnist_data/", download=True, train=False)
                            

def train():
    dataloader = DataLoader(train_mnist, shuffle=True)
    label_counts = defaultdict(int)
    trained = 0

    for i, (data, label) in enumerate(train_mnist):
        if label_counts[label] >= min_per_label:
            continue

        numpy_image = np.array(data)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image.reshape(1, -1).squeeze()

        neuron_cells.train(vector=flatten_image, target=label)

        label_counts[label] += 1
        trained += 1

        if i % 4 == 0:
            progress = int(trained / number_of_neuron_cells * 100)
            print(f"progress : {progress}%", flush=True)
            time.sleep(0.01)

        # ✅ 라벨 10개 × 100개 학습되면 저장
        if trained >= number_of_neuron_cells:
            print("train finish")
            with open("trained_neuron.pkl", "wb") as f:
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

def inference_debug_external(neuron_cells, vector, result_widget=None, top_n=5):

    label_distances = defaultdict(list)  # 🔹 라벨별 거리 저장
    for cell in neuron_cells.cells:
        d = manhattan_distance(vector, cell._vector)
        distances.append((d, cell.target))
        label_distances[cell.target].append(d)  # 🔹 라벨별 그룹핑
    distances.sort(key=lambda x: x[0])

    # 👉 Top-5 뉴런 출력
    if result_widget:
        result_widget.append("[DEBUG] Distance to neurons (Top-5):")
        for i, (dist, label) in enumerate(distances[:top_n]):
            result_widget.append(f"  Top-{i+1}: label={label}, distance={dist:.2f}")
    else:
        print("[DEBUG] Distance to neurons (Top-5):")
        for i, (dist, label) in enumerate(distances[:top_n]):
            print(f"  Top-{i+1}: label={label}, distance={dist:.2f}")

    # 👉 라벨별 평균 거리 출력
    label_avg = {label: np.mean(dlist) for label, dlist in label_distances.items()}
    sorted_avg = sorted(label_avg.items(), key=lambda x: x[1])
    if result_widget:
        result_widget.append("[DEBUG] Mean distance per label:")
        for label, avg in sorted_avg:
            result_widget.append(f"  label={label}: mean distance={avg:.2f}")
    else:
        print("[DEBUG] Mean distance per label:")
        for label, avg in sorted_avg:
            print(f"  label={label}: mean distance={avg:.2f}")
    return distances[0][1]  # Top-1의 label 리턴

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
            predict_label = inference_debug_external(neuron_cells_loaded, flatten_image, top_n=10)
            print(f"[INFO] Inference Result: {predict_label}", flush=True)
            return
        except Exception as e:
            print(f"[ERROR] Pre-processing failed: {e}", flush=True)
            return

    # ------------------------------
    # 기본 내장 MNIST 테스트셋으로 추론 (예전 방식)
    cnt = 0
    correct = 0

    for data, label in tqdm(test_mnist):
        if cnt ==5:
            break
        cnt +=1
        numpy_image = np.array(data)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image.reshape(1, -1).squeeze()
        predict_label = neuron_cells_loaded.inference(vector=flatten_image)
        # print(f"label : {label}, predict_label : {predict_label}", flush=True)
        if predict_label==label:
            correct+=1
        print(neuron_cells.inference_distances(flatten_image))
        print(neuron_cells.inference_category(flatten_image))
        print(neuron_cells.inference(flatten_image))

    print(correct/cnt*100)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer()
    else:
        train()

    infer()