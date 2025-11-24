# mnist.py  (GUI / exe 전용 버전)

import os
import sys
import time
import pickle
import math

import numpy as np
import cv2

from intellino.core.neuron_cell import NeuronCells


# ─────────────────────────────────────
# PyInstaller(onefile) + 개발환경 겸용 경로 헬퍼
# ─────────────────────────────────────
def resource_path(name: str) -> str:
    """
    onefile 실행 시:  sys._MEIPASS 기준
    개발 환경 실행 시: 이 파일(__file__) 기준
    두 경우 모두에서 name 파일을 찾도록 함.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, name),                         # e.g. trained_neuron.pkl
        os.path.join(base, "main", name),                 # onefile에서 ;main 구조
        os.path.join(os.path.dirname(__file__), name),    # 개발 환경
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 못 찾으면 일단 첫 번째 후보 리턴
    return candidates[0]


# 미리 학습해 둔 모델 파일 (PyInstaller에서 같이 패키징됨)
MODEL_PATH = resource_path("trained_neuron.pkl")

# intellino 입력 벡터 길이 (MNIST 28x28)
LENGTH_OF_INPUT_VECTOR = 784
RESIZE_SIZE = int(math.sqrt(LENGTH_OF_INPUT_VECTOR))


# ─────────────────────────────────────
# 1. GUI에서 사용하는 "가짜 학습" 함수
# ─────────────────────────────────────
def train(progress_cb=None, log_cb=None):
    """
    GUI에서 MNIST 버튼 눌렀을 때 호출되는 함수.
    - 실제 학습은 하지 않고,
    - progress_cb 를 통해 0 → 100% 까지 게이지만 채워줌.
    - log_cb 를 통해 텍스트 로그만 출력.

    기존 학습 속도와 비슷하게 보이도록 step 수 / 슬립 시간 조절 가능.
    """

    # 로그 출력 (기존 "MNIST 학습을 시작합니다..." 역할)
    if log_cb is not None:
        try:
            log_cb("MNIST training started (using pre-trained model).")
        except Exception:
            pass

    # 게이지를 몇 단계로 나눌지 (기존 25번 정도였으니 그대로 25 단계 사용)
    steps = 25

    # 한 스텝당 딜레이 (체감 속도 조절용)
    # 너무 빠르면 0.02, 너무 느리면 0.01 등으로 조정 가능
    SLEEP_SEC = 0.03

    for i in range(steps + 1):
        percent = int(100 * i / steps)

        # 진행률 콜백
        if progress_cb is not None:
            try:
                progress_cb(percent)
            except Exception:
                pass

        # 너무 순식간에 끝나지 않도록 약간 딜레이
        time.sleep(SLEEP_SEC)

    # 마지막 로그
    if log_cb is not None:
        try:
            log_cb("MNIST training completed.")
        except Exception:
            pass


# ─────────────────────────────────────
# 2. 이미지 전처리 + 단일 이미지 추론 함수
# ─────────────────────────────────────
def preprocess_user_image(image_path: str) -> np.ndarray:
    """
    기존에 사용하던 OpenCV 기반 전처리 그대로 유지:
      1) BGR → Gray
      2) Gaussian Blur
      3) OTSU threshold + 필요시 반전
      4) 숫자 영역 bounding box로 crop
      5) 긴 변 기준 20px로 리사이즈 (비율 유지)
      6) 28x28 중앙 배치
      7) flatten → (784,) float32
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to mount image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # OTSU threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

    # 배경이 흰색이고 숫자가 검정이면 MNIST 스타일로 맞추기 위해 반전
    if np.mean(binary) > 127:
        binary = 255 - binary

    # 숫자 영역만 crop
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y + h, x:x + w]

    # 긴 변 기준 20px로 리사이즈
    target_size = 20
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 28x28 중앙 배치
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # (784,) float32 벡터로 변환
    flatten = padded.reshape(1, -1).squeeze().astype(np.float32)
    return flatten


def infer_image(image_path: str) -> int:
    """
    GUI에서 사용할 단일 이미지 추론 함수.
    - 미리 학습된 trained_neuron.pkl 을 로드
    - preprocess_user_image()로 전처리
    - NeuronCells.inference() 호출
    """

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"There is no learned model. Expected at: {MODEL_PATH}")

    # NeuronCells 객체 로드
    with open(MODEL_PATH, "rb") as f:
        neuron_cells_loaded: NeuronCells = pickle.load(f)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File does not exist: {image_path}")

    flatten_image = preprocess_user_image(image_path)
    predict_label = neuron_cells_loaded.inference(vector=flatten_image)
    return int(predict_label)


# ─────────────────────────────────────
# 3. 단독 실행 디버그용 (선택 사항)
# ─────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        if len(sys.argv) < 3:
            print("Usage: python mnist.py infer <image_path>")
            sys.exit(1)
        label = infer_image(sys.argv[2])
        print(f"Input Image Inference Results: predict_label = {label}")
    else:
        # 콘솔에서 python mnist.py만 실행하면
        # progress 0~100% 보여주는 가짜 학습만 수행
        def _p(v):
            print(f"progress : {v}", flush=True)
        def _l(msg):
            print(msg, flush=True)
        train(progress_cb=_p, log_cb=_l)
