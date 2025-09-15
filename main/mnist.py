import numpy as np
import math, time, cv2, sys, os, pickle
from torchvision import datasets, transforms
from intellino.core.neuron_cell import NeuronCells
from torch.utils.data import DataLoader
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "trained_neuron.pkl")




#-----------------------------------intellino train--------------------------------#


number_of_neuron_cells = 1000
length_of_input_vector = 784
min_per_label = number_of_neuron_cells // 10

resize_size = int(math.sqrt(length_of_input_vector))
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")

transform = transforms.ToTensor()
# dataset
train_mnist = datasets.MNIST('../mnist_data/', download=True, train=True)
test_mnist = datasets.MNIST("../mnist_data/", download=True, train=False)


def train():  
    label_counts = defaultdict(int)
    train_num=0

    for i, (data, label) in enumerate(train_mnist):
        if label_counts[label] >= min_per_label:
            continue
        train_num+=1
        numpy_image = np.array(data)
        # numpy_image = data.squeeze().numpy().astype(np.uint8)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image .reshape(1, -1).squeeze()
        is_finish = neuron_cells.train(vector=flatten_image, target=label)

        label_counts[label] += 1

        # 학습 진행률
        progress = int((train_num/number_of_neuron_cells)*100)
        if train_num%(number_of_neuron_cells/25)==0:
            print(f"progress : {progress}", flush=True)     # flush=True : 출력 내용을 바로 콘솔로 내보내게 함
            time.sleep(0.01)

        # 지금 is_finish가 True가 안됨... 누가 해결좀 해줘
        if is_finish == True:
            print("train finish")
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(neuron_cells, f)
            break


#-----------------------------------intellino test---------------------------------#


def preprocess_user_image(image_path):
    # 0. 이미지 읽기 (컬러 -> 그레이)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to mount image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 약한 Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    cv2.imwrite("1.png", blurred)

    # 2. Threshold (Otsu 방식)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

    # 픽셀 반전 (흰 배경일 경우 평균 밝기 기준)
    if np.mean(binary) > 127:
        binary = 255 - binary  # MNIST 스타일로 맞춤

    # 3. 숫자 영역만 추출
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]

    # 4. Aspect Ratio 유지하면서 최대한 키우기
    target_size = 20
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. 중앙 배치 (28x28)
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # 6. 저장 (디버깅용)
    cv2.imwrite("preprocessed_user_image.png", padded)

    # 8. Flatten
    flatten = padded.reshape(1, -1).squeeze()

    return flatten


#    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#    ret , binary = cv2.threshold(src,170,255,cv2.THRESH_BINARY_INV)
#
#    # 픽셀 반전 (흰 배경일 경우 평균 밝기 기준)
#    if np.mean(binary) > 127:
#        binary = 255 - binary  # MNIST 스타일로 맞춤
#
#    myNum1 = np.asarray(cv2.resize(binary, dsize=(resize_size, resize_size), interpolation=cv2.INTER_AREA))
#    cv2.imwrite("preprocessed_user_image.png", myNum1)
#    myNum2 = myNum1/255
#
#    flatten_image = myNum2.reshape(1, -1).squeeze()
#
#    return flatten_image



def infer():

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] There is no learned model. Run learning first.", flush=True)
        return

    with open(MODEL_PATH, "rb") as f:
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
    else:
        # ------------------------------
        # 기본 내장 MNIST 테스트셋 전체로 추론
        correct = 0
        total = 0

        # 각 클래스(0~9)별로 맞춘 개수, 전체 개수 카운트
        num_classes = 10  # MNIST는 0~9, 10개 클래스
        class_correct = [0 for _ in range(num_classes)]
        class_total = [0 for _ in range(num_classes)]
        
        for i, (data, label) in tqdm(enumerate(test_mnist), total=len(test_mnist), desc="Infer Progress"):
            numpy_image = np.array(data)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(numpy_image, dsize=(resize_size, resize_size))
            flatten_image = resized_image.reshape(1, -1).squeeze()
        
            predict_label = neuron_cells_loaded.inference(vector=flatten_image)
        
            if predict_label == label:
                correct += 1
                class_correct[label] += 1  # 정답 개수 추가
            class_total[label] += 1  # 총 시도 개수 추가
            total += 1
        
        accuracy = correct / total * 100
        print(f"\n[Test Result] Total Accuracy: {accuracy:.2f}% ({correct}/{total})", flush=True)
        
        # 클래스별 정확도 출력
        print("\n[Class-wise Accuracy]")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i] * 100
                print(f"Label {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"Label {i}: No samples.")
                          
                      
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer()           
    else:                 
        train()           
        # infer()        

                          
                      
                          
                          
                          
                      
                          
                          