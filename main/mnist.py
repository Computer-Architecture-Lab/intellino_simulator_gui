# mnist.py  (GUI / exe 전용 버전)

import os
import sys
import time
import pickle
import math

import numpy as np
import cv2

from intellino.core.neuron_cell import NeuronCells

# 공통 유틸
from utils.resource_utils import resource_path
from utils.image_preprocess import preprocess_user_image


# 미리 학습해 둔 모델 파일 (PyInstaller에서 같이 패키징됨)
MODEL_PATH = resource_path("trained_neuron.pkl")

# intellino 입력 벡터 길이 (MNIST 28x28)
LENGTH_OF_INPUT_VECTOR = 784
RESIZE_SIZE = int(math.sqrt(LENGTH_OF_INPUT_VECTOR))


# ─────────────────────────────────────
# 1. GUI에서 사용하는 학습 함수
# ─────────────────────────────────────
def train(progress_cb=None, log_cb=None):
    """
    GUI에서 MNIST 버튼 눌렀을 때 호출되는 함수.
    - 실제 학습은 하지 않고,
    - progress_cb 를 통해 0 → 100% 까지 게이지만 채워줌.
    - log_cb 를 통해 텍스트 로그만 출력.
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

        # 딜레이
        time.sleep(SLEEP_SEC)

    # 마지막 로그
    if log_cb is not None:
        try:
            log_cb("MNIST training completed.")
        except Exception:
            pass

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

    flatten_image = preprocess_user_image(image_path, LENGTH_OF_INPUT_VECTOR)
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
