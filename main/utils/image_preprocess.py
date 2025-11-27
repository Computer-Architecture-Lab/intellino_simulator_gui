# utils/image_preprocess.py
import cv2
import math
import numpy as np


def preprocess_user_image(image_path: str, length_of_input_vector=None):

    RESIZE_SIZE = int(math.sqrt(length_of_input_vector))
    # 1. 이미지 읽기 (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to mount image.")

    # 2. 이미지 크기 조정
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 픽셀 반전 (흰 배경일 경우 평균 밝기 기준)
    if np.mean(image) > 127:
        image = 255 - image  # MNIST 스타일로 맞춤

    # 4. 디버깅 이미지 저장
    cv2.imwrite("preprocessed_debug.png", image)

    # 5. 평탄화 (flatten)
    return image.reshape(1, -1).squeeze()
