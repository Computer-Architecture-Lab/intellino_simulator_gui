import numpy as np
import math, time, cv2, sys, os, pickle
from torchvision import datasets
from intellino.core.neuron_cell import NeuronCells
from torch.utils.data import DataLoader
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
    for i, (data, label) in enumerate(train_mnist):
        numpy_image = np.array(data)
        resized_image = cv2.resize(numpy_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image.reshape(1, -1).squeeze()
        flatten_image = flatten_image / 255.0               # 정규화
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
    # 0. 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to mount image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 숫자 부분 crop
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]

    # 4. Aspect Ratio 유지해서 20x20 안에 넣기
    max_side = max(w, h)
    square_image = np.zeros((max_side, max_side), dtype=np.uint8)
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2
    square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    resized = cv2.resize(square_image, (20, 20), interpolation=cv2.INTER_AREA)

    # 5. 28x28 중앙 배치
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_pad = (28 - 20) // 2
    y_pad = (28 - 20) // 2
    padded[y_pad:y_pad+20, x_pad:x_pad+20] = resized

    # 6. 저장 (디버깅용)
    cv2.imwrite("preprocessed_user_image.png", padded)

    # 7. 정규화
    normalized = padded / 255.0

    # 8. Flatten
    flatten = normalized.reshape(1, -1).squeeze()

    return flatten

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

            # ------------------------------
            # 기본 내장 MNIST 테스트셋 전체로 추론
            correct = 0
            total = 0

            for i, (data, label) in tqdm(enumerate(test_mnist), total=len(test_mnist), desc="Infer Progress"):
                numpy_image = np.array(data)
                resized_image = cv2.resize(numpy_image, dsize=(resize_size, resize_size))
                flatten_image = resized_image.reshape(1, -1).squeeze()
                flatten_image = flatten_image / 255.0  # 정규화 (0~1)

                predict_label = neuron_cells_loaded.inference(vector=flatten_image)

                if predict_label == label:
                    correct += 1
                total += 1

            accuracy = correct / total * 100
            print(f"\n[Test Result] Accuracy: {accuracy:.2f}% ({correct}/{total})", flush=True)
            return

        except Exception as e:
            print(f"[ERROR] Pre-processing failed: {e}", flush=True)
            return

                          
                      
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer()           
    else:                 
        train()           
                      
                          
                      
                          
                          
                          
                      
                          
                          