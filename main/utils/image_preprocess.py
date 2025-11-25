# utils/image_preprocess.py
import cv2
import numpy as np

def preprocess_digit_image(image_path: str) -> np.ndarray:
    """
    모든 숫자 이미지 전처리를 위한 통합 함수.
    custom_3, mnist 모두 이 함수 하나만 사용하도록 통일 가능.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # OTSU + 필요 시 반전
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:  
        binary = 255 - binary

    # crop
    coords = cv2.findNonZero(binary)
    if coords is None:
        cropped = binary
    else:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = binary[y:y+h, x:x+w]

    # 긴 변 기준 20px resize
    target = 20
    h2, w2 = cropped.shape[:2]
    if h2 == 0 or w2 == 0:
        resized = np.zeros((20,20), dtype=np.uint8)
    else:
        if w2 > h2:
            new_w, new_h = target, max(1, int(h2 * target / w2))
        else:
            new_h, new_w = target, max(1, int(w2 * target / h2))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 28×28 중앙 배치
    canvas = np.zeros((28,28), dtype=np.uint8)
    yo = (28 - resized.shape[0]) // 2
    xo = (28 - resized.shape[1]) // 2
    canvas[yo:yo+resized.shape[0], xo:xo+resized.shape[1]] = resized

    return (canvas.astype(np.float32) / 255.0).reshape(-1)
