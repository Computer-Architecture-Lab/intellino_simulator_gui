# custom_3.py — train/test split with accuracy + native file dialog
# 변경: 실험 상태(EXPERIMENT_STATE)에 [파라미터 라벨, 정확도] 기록

import sys, os, shutil, datetime, numpy as np, traceback, pickle, cv2
from PIL import Image

# 추론 경로 제어(기본 False: 어느 경로든 허용, 단 학습 파일 차단)
INFER_ONLY_FROM_TEST = False

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH   = os.path.join(ASSETS_DIR, "intellino_TM_transparent.png")
HOME_ICON_PATH = os.path.join(ASSETS_DIR, "home.png")

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QSizePolicy, QGraphicsOpacityEffect,
    QLineEdit, QFileDialog, QProgressBar, QTextBrowser, QMessageBox
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor, QTextCursor
from PySide2.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve

from path_utils import get_dirs
# ▶ 실험 상태 전역 객체(커스텀4에서 정의)
from custom_4 import EXPERIMENT_STATE

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

CUSTOM_IMAGE_ROOT, NUMBER_IMAGE_DIR, DEFAULT_OUTPUT_ROOT = get_dirs(__file__)
MODEL_BASENAME = "custom_model.pkl"

BUTTON_STYLE = """
    QPushButton {
        background-color:#ffffff; border:1px solid #ccc; border-radius:10px;
        padding:6px 12px; font-weight:600; font-size:13px;
    }
    QPushButton:hover { background-color:#e9ecef; }
    QPushButton:pressed { background-color:#adb5bd; color:white; }
"""

# ---------------------------
# 데이터 유틸
def load_images_from_dir(dir_path: str):
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))
             if str(f).lower().endswith(IMG_EXTS)]
    X, keep = [], []
    for p in files:
        try:
            img = Image.open(p).convert("L").resize((28,28))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.flatten()); keep.append(os.path.abspath(p))
        except Exception:
            pass
    if not X:
        return np.empty((0,784), dtype=np.float32), []
    return np.stack(X, axis=0), keep

def vectorize_like_training(path: str):
    try:
        img = Image.open(path).convert("L").resize((28,28))
        return (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1)
    except Exception:
        return None

def preprocess_user_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to open image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    coords = cv2.findNonZero(binary)
    if coords is None:
        cropped = binary
    else:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = binary[y:y+h, x:x+w]

    target = 20
    h, w = cropped.shape[:2]
    if w == 0 or h == 0:
        resized = np.zeros((20,20), dtype=np.uint8)
    else:
        if w > h:
            new_w, new_h = target, max(1, int(h * target / w))
        else:
            new_h, new_w = target, max(1, int(w * target / h))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28,28), dtype=np.uint8)
    yo = (28 - resized.shape[0]) // 2
    xo = (28 - resized.shape[1]) // 2
    canvas[yo:yo+resized.shape[0], xo:xo+resized.shape[1]] = resized

    return (canvas.astype(np.float32) / 255.0).reshape(-1)

# ---------------------------
# 간단 최근접-이웃 모델(L1)
class SimpleNearestModel:
    def __init__(self):
        self.vectors = None
        self.labels  = []

    def fit_from_root(self, root_dir: str):
        if not os.path.isdir(root_dir):
            raise ValueError(f"Train root not found: {root_dir}")
        vecs, labs = [], []
        subdirs = [d for d in sorted(os.listdir(root_dir))
                   if os.path.isdir(os.path.join(root_dir, d))]
        for d in subdirs:
            X, _ = load_images_from_dir(os.path.join(root_dir, d))
            if X.size == 0:
                continue
            vecs.append(X); labs += [d]*X.shape[0]
        if not vecs:
            raise RuntimeError("No images found for training.")
        self.vectors = np.concatenate(vecs, axis=0).astype(np.float32)
        self.labels  = labs

    def predict(self, vector: np.ndarray, top_k: int = 3):
        if self.vectors is None or not self.labels:
            raise RuntimeError("Model not trained.")
        dists = np.abs(self.vectors - vector[None,:]).sum(axis=1)
        idx = np.argsort(dists)[:max(1, top_k)]
        return [self.labels[i] for i in idx], [float(dists[i]) for i in idx]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "labels": self.labels}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.vectors = obj["vectors"].astype(np.float32)
        self.labels  = list(obj["labels"])

# ---------------------------
# UI 구성 요소
class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(50)
        self.setStyleSheet(
            "background-color:#f1f3f5;"
            "border-top-left-radius:15px; border-top-right-radius:15px;"
        )
        self.setAttribute(Qt.WA_StyledBackground, True)

        h = QHBoxLayout(self); h.setContentsMargins(15,0,15,0)
        logo = QLabel()
        pm = QPixmap(LOGO_PATH)
        if pm.isNull():
            logo.setText("intellino"); logo.setStyleSheet("font-weight:600;")
        else:
            logo.setPixmap(pm.scaled(65,65, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        home = QPushButton()
        home.setIcon(QIcon(HOME_ICON_PATH)); home.setIconSize(QSize(24,24))
        home.setFixedSize(34,34)
        home.setStyleSheet("QPushButton{border:none;background:transparent;} "
                           "QPushButton:hover{background:#dee2e6; border-radius:17px;}")
        home.clicked.connect(self._on_home)

        h.addWidget(logo); h.addStretch(); h.addWidget(home)
        self._offset = None

    def _on_home(self):
        app = QApplication.instance()
        if app: app.setStyleSheet("")
        if self._parent: self._parent.close()

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton: self._offset = e.pos()
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._offset is not None and e.buttons()==Qt.LeftButton and self._parent:
            self._parent.move(self._parent.pos() + e.pos() - self._offset)
    def mouseReleaseEvent(self, e: QMouseEvent):
        self._offset = None

class ProgressSection(QWidget):
    def __init__(self, title="6. Train"):
        super().__init__()
        g = QGroupBox(title); g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )
        g.setFixedHeight(70)
        row = QHBoxLayout()
        self.bar = QProgressBar(); self.bar.setTextVisible(False); self.bar.setValue(0); self.bar.setFixedHeight(8)
        self.bar.setStyleSheet("QProgressBar{border:1px solid #bbb;border-radius:3px;background:#f1f1f1;}"
                               "QProgressBar::chunk{background:#3b82f6;border-radius:3px;}")
        self.perc = QLabel("0%"); self.perc.setAlignment(Qt.AlignRight|Qt.AlignVCenter); self.perc.setFixedWidth(50)
        row.addWidget(self.bar); row.addWidget(self.perc); g.setLayout(row)
        v = QVBoxLayout(self); v.addWidget(g)
    def update(self, v:int):
        v = max(0, min(100, int(v))); self.bar.setValue(v); self.perc.setText(f"{v}%")

class ResultView(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(False)
        self.setReadOnly(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QTextBrowser{font-size:14px;border:1px solid #ccc;"
                           "border-radius:8px;padding:10px;background:#f8f9fa;}")
        self.document().setDefaultStyleSheet("""
            .blk { margin:6px 0; }
            .dim { color:#495057; }
            .ok  { color:#2b8a3e; }
            .info{ color:#1c7ed6; }
            .err { color:#c92a2a; }
            .hr  { height:1px; background:#e9ecef; margin:10px 0; }
            table.grid { border-collapse:collapse; font-size:12.5px; width:100%; }
            table.grid th { text-align:left; padding:4px 8px; background:#e9ecef; border:1px solid #dde2e6; }
            table.grid td { padding:4px 8px; border:1px solid #e9ecef; }
            .pred { font-weight:700; font-size:15px; }
            code  { background:#fff; border:1px solid #e9ecef; border-radius:4px; padding:1px 4px; }
        """)

    def add_block(self, html: str):
        self.moveCursor(QTextCursor.End)
        self.insertHtml(f"<p class='blk'>{html}</p>")
        self.append("")
        self.ensureCursorVisible()

    def add_hr(self):
        self.add_block("<div class='hr'></div>")

class InferenceSection(QWidget):
    def __init__(self):
        super().__init__()
        title = "8. Inference" if not INFER_ONLY_FROM_TEST else "8. Inference (use datasets/test)"
        g = QGroupBox(title); g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )
        h = QHBoxLayout()
        placeholder = "Select image (default: datasets/test)" if not INFER_ONLY_FROM_TEST \
                      else "Select test image (default: datasets/test)"
        self.file_input = QLineEdit(); self.file_input.setPlaceholderText(placeholder)
        self.file_input.setFixedHeight(35)
        self.file_input.setStyleSheet("QLineEdit{border:1px solid #ccc;border-radius:8px;padding-left:10px;font-size:13px;}")
        self.browse_btn = QPushButton("..."); self.browse_btn.setFixedSize(35,35)
        self.browse_btn.setStyleSheet("QPushButton{border:1px solid #ccc;border-radius:8px;background:#fff;font-weight:700;} QPushButton:hover{background:#e9ecef;}")
        self.start_btn = QPushButton("Start"); self.start_btn.setFixedSize(70,35); self.start_btn.setStyleSheet(BUTTON_STYLE)
        h.addWidget(self.file_input); h.addWidget(self.browse_btn); h.addWidget(self.start_btn)
        g.setLayout(h)
        v = QVBoxLayout(self); v.addWidget(g)

# ---------------------------
# 메인 창
class SubWindow(QWidget):
    def __init__(self, selection, samples_per_class: int = 1, prev_window=None, output_root=DEFAULT_OUTPUT_ROOT, exp_params=None):
        super().__init__()
        self.selection = selection
        self.samples_per_class = max(1, int(samples_per_class))
        self.output_root = output_root
        self.prev_window = prev_window

        self.exp_params = exp_params or {}

        self.num_categories = len(self.selection)
        self.model = SimpleNearestModel()

        self._last_save_root = ""
        self._datasets_root  = ""
        self._train_dir      = ""
        self._test_dir       = ""

        self._train_originals = set()
        self._train_copies    = set()
        self._test_items      = []
        self._last_accuracy   = None

        self._setup_ui()
        QTimer.singleShot(150, self._run_kmeans_and_train)

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setFixedSize(800,800)

        container = QWidget(self); container.setGeometry(0,0,800,800)
        container.setStyleSheet("background-color:white; border-radius:15px;")
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(30); shadow.setColor(QColor(0,0,0,100))
        self.setGraphicsEffect(shadow)

        self.title_bar = TitleBar(self); self.title_bar.setParent(container); self.title_bar.setGeometry(0,0,800,50)

        lay = QVBoxLayout(container); lay.setContentsMargins(20,60,20,20); lay.setSpacing(20)

        self.progress = ProgressSection("6. Train"); lay.addWidget(self.progress)

        out_g = QGroupBox("7. Output folder"); out_g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )
        row = QHBoxLayout()
        self.out_label = UILabel = QLabel("")
        self.out_label.setStyleSheet("font-size:13px;")
        self.open_btn = QPushButton("Open folder"); self.open_btn.setFixedSize(110,32); self.open_btn.setStyleSheet(BUTTON_STYLE)
        self.open_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self.out_label); row.addStretch(); row.addWidget(self.open_btn); out_g.setLayout(row)
        lay.addWidget(out_g)

        self.infer = InferenceSection(); lay.addWidget(self.infer)
        self.infer.browse_btn.clicked.connect(self._browse_infer_file)
        self.infer.start_btn.clicked.connect(self._start_inference)

        res_g = QGroupBox("9. Result"); res_g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )
        self.result = ResultView()
        rg_lay = QVBoxLayout(res_g); rg_lay.addWidget(self.result)
        lay.addWidget(res_g)

        btn_row = QHBoxLayout()
        self.next_btn = QPushButton("Next"); self.next_btn.setFixedSize(110,38); self.next_btn.setStyleSheet(BUTTON_STYLE)
        self.next_btn.clicked.connect(self._go_next); self.next_btn.setEnabled(False)
        btn_row.addStretch(); btn_row.addWidget(self.next_btn); lay.addLayout(btn_row)

    def _ok(self, text:str):   self.result.add_block(f"<span class='ok'>{text}</span>")
    def _info(self, text:str): self.result.add_block(f"<span class='info'>{text}</span>")
    def _hint(self, text:str): self.result.add_block(f"<span class='dim'>{text}</span>")
    def _err(self, text:str):  self.result.add_block(f"<span class='err'>{text}</span>")

    def _norm(self, p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    def _is_training_file(self, p: str) -> bool:
        npth = self._norm(p)
        return (npth in self._train_originals) or (npth in self._train_copies)

    def _make_param_label(self) -> str:
        c = self.exp_params.get("num_classes", self.num_categories)
        t = self.exp_params.get("samples_per_class", self.samples_per_class)
        v = self.exp_params.get("input_vec_len", None)
        m = self.exp_params.get("memory_kb", None)
        parts = []
        if v is not None: parts.append(f"V{v}")
        parts.append(f"C{c}")
        parts.append(f"T{t}")
        if m is not None: parts.append(f"M{m}K")
        return " / ".join(map(str, parts))

    def _run_kmeans_and_train(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(DEFAULT_OUTPUT_ROOT, ts)
        os.makedirs(save_root, exist_ok=True)
        self._last_save_root = save_root

        datasets_root = os.path.join(save_root, "datasets")
        train_root    = os.path.join(datasets_root, "train")
        test_root     = os.path.join(datasets_root, "test")
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)
        self._datasets_root = datasets_root
        self._train_dir     = train_root
        self._test_dir      = test_root

        self.out_label.setText(self._datasets_root)

        self._train_originals.clear()
        self._train_copies.clear()
        self._test_items.clear()

        total = max(1, len(self.selection))
        for i, item in enumerate(self.selection, start=1):
            dir_path = item["dir"]; label = str(item["label"])
            X, srcs = load_images_from_dir(dir_path)
            if len(srcs) == 0:
                self.progress.update(int(i/total*100)); continue

            k = min(self.samples_per_class, len(srcs))
            chosen_idx  = list(range(k))
            chosen_set  = set(chosen_idx)

            dst_train_label = os.path.join(train_root, label)
            os.makedirs(dst_train_label, exist_ok=True)
            for r, idx in enumerate(chosen_idx, start=1):
                src  = srcs[idx]; base = os.path.basename(src)
                dst  = os.path.join(dst_train_label, f"{label}_sel{r:02d}_{base}")
                try:
                    shutil.copy2(src, dst)
                    self._train_originals.add(self._norm(src))
                    self._train_copies.add(self._norm(dst))
                except Exception:
                    pass

            dst_test_label = os.path.join(test_root, label)
            os.makedirs(dst_test_label, exist_ok=True)
            for idx, src in enumerate(srcs):
                if idx in chosen_set:
                    continue
                base = os.path.basename(src)
                test_dst = os.path.join(dst_test_label, base)
                try:
                    shutil.copy2(src, test_dst)
                    self._test_items.append((test_dst, label))
                except Exception:
                    pass

            self.progress.update(int(i/total*100))

        self.progress.update(100)

        self.result.clear()
        self._ok("K‑means selection completed.")
        self._info("Training dataset prepared at <code>datasets/train/</code>.")
        if self._test_items:
            self._info("Test dataset prepared at <code>datasets/test/</code>.")
        else:
            self._hint("No test images were available; accuracy cannot be computed.")
        self.result.add_hr()

        try:
            self._info("Training on datasets/train …")
            self.model.fit_from_root(self._train_dir)
            self.model.save(os.path.join(save_root, MODEL_BASENAME))
            self._ok("Training completed.")
        except Exception as e:
            self._err(f"Training failed: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
            return

        try:
            if self._test_items:
                correct, total = 0, 0
                for p, true_lab in self._test_items:
                    vec = vectorize_like_training(p)
                    if vec is None:
                        continue
                    pred_lab = self.model.predict(vec, top_k=1)[0][0]
                    total += 1
                    if pred_lab == true_lab:
                        correct += 1
                if total > 0:
                    acc = 100.0 * correct / total
                    self._last_accuracy = float(acc)
                    # ▶ 실험 상태에 누적
                    EXPERIMENT_STATE.add_run(self._make_param_label(), float(acc))

                    self.result.add_block("<b>Test evaluation</b>")
                    self.result.add_block(
                        f"Accuracy: <b>{acc:.2f}%</b> "
                        f"(<code>{correct}</code>/<code>{total}</code>) on <code>datasets/test/</code>"
                    )
                else:
                    self._hint("Test dataset exists but no readable images; accuracy cannot be computed.")
            self.result.add_hr()
            if INFER_ONLY_FROM_TEST:
                self._hint("You can now run inference below (section 8). Only <code>datasets/test</code> files are allowed.")
            else:
                self._hint("You can now run inference below (section 8). You may choose images from anywhere; training files are blocked.")
            self.result.add_hr()
            self.next_btn.setEnabled(True)
        except Exception as e:
            self._err(f"Evaluation failed: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")

    # ---------- Inference ----------
    def _browse_infer_file(self):
        start_dir = self._test_dir if (self._test_dir and os.path.isdir(self._test_dir)) else self._datasets_root
        title = "Select test image (only from datasets/test)" if INFER_ONLY_FROM_TEST else "Select image"
        while True:
            file_path, _ = QFileDialog.getOpenFileName(self, title, start_dir, "Images (*.png *.jpg *.jpeg *.bmp)")
            if not file_path:
                return
            rp = os.path.realpath(file_path)
            is_training = self._is_training_file(rp)
            if INFER_ONLY_FROM_TEST:
                test_root = os.path.realpath(self._test_dir) if self._test_dir else ""
                in_test_root = bool(test_root and os.path.commonpath([rp, test_root]) == test_root)
                if in_test_root and not is_training:
                    self.infer.file_input.setText(file_path); return
                if not in_test_root:
                    QMessageBox.warning(self, "Not allowed", "허용되지 않은 경로입니다.\n추론 이미지는 datasets/test 폴더에서 선택해 주세요.")
                elif is_training:
                    QMessageBox.warning(self, "Not allowed", "이 파일은 학습에 사용되었습니다. 다른 파일을 선택해 주세요.")
            else:
                if not is_training:
                    self.infer.file_input.setText(file_path); return
                QMessageBox.warning(self, "Not allowed", "이 파일은 학습에 사용되었습니다. 다른 파일을 선택해 주세요.")

    def _start_inference(self):
        image_path = self.infer.file_input.text().strip()
        if not image_path:
            self._hint("Please choose an image file first."); return

        if self._is_training_file(image_path):
            QMessageBox.warning(self,"Not allowed","This file was used for training. Please choose a different file.")
            return

        if self.model.vectors is None or not self.model.labels:
            model_path = os.path.join(self._last_save_root, MODEL_BASENAME)
            if os.path.exists(model_path):
                try: self.model.load(model_path)
                except Exception as e:
                    self._err(f"Failed to load model: {e}")
                    self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
                    return
            else:
                self._err("No trained model. Run train first."); return

        try:
            vec = preprocess_user_image(image_path)
            top_labels, top_dists = self.model.predict(vec, top_k=3)
            img_name = os.path.basename(image_path)
            pred = top_labels[0]

            self.result.add_block("🔎 <b>Inference</b>")
            if self._last_accuracy is not None:
                self._hint(f"Last test accuracy (datasets/test): {self._last_accuracy:.2f}%")
            self.result.add_block(f"Input: <code>{img_name}</code>")
            self.result.add_block(f"Prediction: <span class='pred'>{pred}</span>")

            rows = "".join(
                f"<tr><td>{i}</td><td>{lab}</td><td>{dist:.3f}</td></tr>"
                for i, (lab, dist) in enumerate(zip(top_labels, top_dists), start=1)
            )
            self.result.add_block(
                f"<div class='dim' style='font-weight:600;margin-top:4px;margin-bottom:2px;'>Top‑{len(top_labels)} nearest</div>"
                f"<table class='grid'>"
                f"<tr><th>Rank</th><th>Label</th><th>Distance</th></tr>{rows}</table>"
            )
        except Exception as e:
            self._err(f"Inference failed: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")

    # ---------- etc ----------
    def _open_output_folder(self):
        p = self._datasets_root.strip()
        if not p: p = self._last_save_root or ""
        if not p: return
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                __import__("subprocess").Popen(["open", p])
            else:
                __import__("subprocess").Popen(["xdg-open", p])
        except Exception:
            pass

    def _go_next(self):
        try:
            from custom_4 import ExperimentWindow as Window4
        except Exception as e:
            self._err(f"Failed to import next window: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
            return

        try:
            win4 = Window4(num_categories=getattr(self, "num_categories", 0))
            try: win4.setGeometry(self.geometry())
            except Exception: win4.move(self.pos())
            win4.show(); win4.raise_(); win4.activateWindow()

            snap = self.grab()
            overlay = QLabel(win4); overlay.setPixmap(snap)
            overlay.setGeometry(0,0,self.width(), self.height())
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            overlay.raise_(); overlay.show()

            eff = QGraphicsOpacityEffect(overlay); overlay.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(180); anim.setStartValue(1.0); anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.InOutQuad)

            def _done(): overlay.deleteLater(); self.hide()
            self._overlay_anim = anim; anim.finished.connect(_done); anim.start()
        except Exception as e:
            self._err(f"Failed to open next window: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SubWindow(selection=[], samples_per_class=1)
    w.show()
    sys.exit(app.exec_())
