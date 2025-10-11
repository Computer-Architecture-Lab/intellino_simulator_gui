# make_mnist_zip.py
import os, zipfile, numpy as np
from PIL import Image
from path_utils import get_dirs

CUSTOM_IMAGE_ROOT, NUMBER_DIR, _ = get_dirs(__file__)

def ensure_dirs():
    os.makedirs(NUMBER_DIR, exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(NUMBER_DIR, str(d)), exist_ok=True)

def save_samples(x, y, per_digit=10):
    counters = {d: 0 for d in range(10)}
    for img, lab in zip(x, y):
        d = int(lab)
        if counters[d] >= per_digit:
            if all(v >= per_digit for v in counters.values()):
                break
            else:
                continue
        im = Image.fromarray(img).convert("L").resize((28,28))
        dst = os.path.join(NUMBER_DIR, str(d), f"{d}_{counters[d]:02d}.png")
        im.save(dst)
        counters[d] += 1
    print("[save] per-digit counts:", counters)

def pack_zip(zip_name="number_image_10_per_digit.zip"):
    zpath = os.path.join(CUSTOM_IMAGE_ROOT, zip_name)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(NUMBER_DIR):
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.relpath(p, CUSTOM_IMAGE_ROOT)
                zf.write(p, arc)
    print("[zip] created:", zpath)

def main():
    ensure_dirs()
    try:
        # 1순위: tensorflow.keras
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()
        save_samples(x_train, y_train, per_digit=10)
    except Exception:
        # 2순위: torchvision
        try:
            from torchvision.datasets import MNIST
            ds = MNIST(root="./_mnist_cache", train=True, download=True)
            x=[]; y=[]
            for i in range(len(ds)):
                img, lab = ds[i]
                x.append(np.array(img)); y.append(lab)
            x = np.stack(x); y = np.array(y)
            save_samples(x, y, per_digit=10)
        except Exception as e:
            print("MNIST 다운로드/로딩 실패:", e)
            print("인터넷 연결 또는 라이브러리(tensorflow/torchvision) 확인 후 다시 시도해 주세요.")
            return
    pack_zip()

if __name__ == "__main__":
    main()
