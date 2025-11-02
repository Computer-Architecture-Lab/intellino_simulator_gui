# custom_3.py — clean log with guaranteed line breaks + inference below Output folder
import sys, os, shutil, datetime, numpy as np, traceback, pickle, cv2
from PIL import Image

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH   = os.path.join(ASSETS_DIR, "intellino_TM_transparent.png")
HOME_ICON_PATH = os.path.join(ASSETS_DIR, "home.png")

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QSizePolicy, QGraphicsOpacityEffect,
    QLineEdit, QFileDialog, QProgressBar, QTextBrowser
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor, QTextCursor
from PySide2.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve

from path_utils import get_dirs

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# 프로젝트 상대 경로 확보
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
            X.append(arr.flatten()); keep.append(p)
        except Exception:
            pass
    if not X:
        return np.empty((0,784), dtype=np.float32), []
    return np.stack(X, axis=0), keep

def preprocess_user_image(image_path: str) -> np.ndarray:
    """외부 이미지 -> 28x28(0~1) float 벡터 (중심 정렬)"""
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
        self.vectors = None  # (N,784) float32
        self.labels  = []    # 길이 N, 각 항목은 str 라벨

    def fit_from_root(self, root_dir: str):
        if not os.path.isdir(root_dir):
            raise ValueError(f"Train root not found: {root_dir}")
        vecs, labs = [], []
        subdirs = [d for d in sorted(os.listdir(root_dir))
                   if os.path.isdir(os.path.join(root_dir, d))]
        for d in subdirs:
            X, keep = load_images_from_dir(os.path.join(root_dir, d))
            if X.size == 0: continue
            vecs.append(X); labs += [d]*len(keep)
        if not vecs:
            raise RuntimeError("No images found for training.")
        self.vectors = np.concatenate(vecs, axis=0).astype(np.float32)
        self.labels  = labs

    def predict(self, vector: np.ndarray, top_k: int = 3):
        if self.vectors is None or not self.labels:
            raise RuntimeError("Model not trained.")
        dists = np.abs(self.vectors - vector[None,:]).sum(axis=1)  # L1
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
    """
    QTextBrowser + 전용 CSS + <p class="blk">…</p> + 강제 개행으로
    어떤 환경에서도 줄바꿈이 확실히 보이도록 함.
    """
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
            .ok  { color:#2b8a3e; font-weight:700; }
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
        # 항상 <p>로 싸고, Qt의 append("")로 빈 문단을 추가해 줄바꿈을 강제
        self.moveCursor(QTextCursor.End)
        self.insertHtml(f"<p class='blk'>{html}</p>")
        self.append("")  # 빈 문단(새 줄) — 환경과 상관없이 줄바꿈 보장
        self.ensureCursorVisible()

    def add_hr(self):
        self.add_block("<div class='hr'></div>")

class InferenceSection(QWidget):
    def __init__(self):
        super().__init__()
        g = QGroupBox("8. Inference"); g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )
        h = QHBoxLayout()
        self.file_input = QLineEdit(); self.file_input.setPlaceholderText("Put in the file to infer")
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
    def __init__(self, selection, samples_per_class: int = 1, prev_window=None, output_root=DEFAULT_OUTPUT_ROOT):
        super().__init__()
        self.selection = selection
        self.samples_per_class = max(1, int(samples_per_class))
        self.output_root = output_root
        self.prev_window = prev_window

        self.num_categories = len(self.selection)
        self.model = SimpleNearestModel()
        self._last_save_root = ""

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
        self.out_label = QLabel(self.output_root); self.out_label.setStyleSheet("font-size:13px;")
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

    # ---------- Log helpers ----------
    def _ok(self, text:str):   self.result.add_block(f"✅ <span class='ok'>{text}</span>")
    def _info(self, text:str): self.result.add_block(f"ℹ️ <span class='info'>{text}</span>")
    def _hint(self, text:str): self.result.add_block(f"<span class='dim'>{text}</span>")
    def _err(self, text:str):  self.result.add_block(f"❌ <span class='err'>{text}</span>")

    # ---------- K-means -> Train ----------
    def _run_kmeans_and_train(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(self.output_root, ts)
        os.makedirs(save_root, exist_ok=True)
        self.out_label.setText(save_root)
        self._last_save_root = save_root

        total = max(1, len(self.selection))
        for i, item in enumerate(self.selection, start=1):
            dir_path = item["dir"]; label = str(item["label"])
            X, srcs = load_images_from_dir(dir_path)
            if len(srcs) == 0:
                self.progress.update(int(i/total*100)); continue

            k = min(self.samples_per_class, len(srcs))
            chosen_idx = list(range(k))  # 간단 샘플 선택: 앞에서 k개

            dst_dir = os.path.join(save_root, label); os.makedirs(dst_dir, exist_ok=True)
            for r, idx in enumerate(chosen_idx, start=1):
                src = srcs[idx]; base = os.path.basename(src)
                dst = os.path.join(dst_dir, f"{label}_sel{r:02d}_{base}")
                try: shutil.copy2(src, dst)
                except Exception: pass
            self.progress.update(int(i/total*100))

        self.progress.update(100)

        # 정돈된 메시지 (각 문장별로 독립 블록이라 줄바꿈 확실)
        self.result.clear()
        self._ok("K‑means selection completed.")
        self._info("Selected files have been saved to the output folder.")
        self.result.add_hr()

        # 즉시 훈련
        try:
            self._info("Training on selected dataset…")
            self.model.fit_from_root(save_root)
            self.model.save(os.path.join(save_root, MODEL_BASENAME))
            self._ok("Training completed.")
            self._hint("You can now run inference below (section 8).")
            self.result.add_hr()
            self.next_btn.setEnabled(True)
        except Exception as e:
            self._err(f"Training failed: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")

    # ---------- Inference ----------
    def _browse_infer_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if p: self.infer.file_input.setText(p)

    def _start_inference(self):
        image_path = self.infer.file_input.text().strip()
        if not image_path:
            self._hint("Please choose an image file first."); return

        if self.model.vectors is None or not self.model.labels:
            model_path = os.path.join(self._last_save_root, MODEL_BASENAME)
            if os.path.exists(model_path):
                try: self.model.load(model_path)
                except Exception as e:
                    self._err(f"Failed to load model: {e}"); return
            else:
                self._err("No trained model. Run train first."); return

        try:
            vec = preprocess_user_image(image_path)
            top_labels, top_dists = self.model.predict(vec, top_k=3)
            img_name = os.path.basename(image_path)
            pred = top_labels[0]

            rows = "".join(
                f"<tr><td>{i}</td><td>{lab}</td><td>{dist:.3f}</td></tr>"
                for i, (lab, dist) in enumerate(zip(top_labels, top_dists), start=1)
            )

            self.result.add_block("🔎 <b>Inference</b>")
            self.result.add_block(f"Input: <code>{img_name}</code>")
            self.result.add_block(f"Prediction: <span class='pred'>{pred}</span>")
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
        p = self.out_label.text().strip()
        if not p: return
        try:
            if sys.platform.startswith("win"): os.startfile(p)  # type: ignore
            elif sys.platform == "darwin": __import__("subprocess").Popen(["open", p])
            else: __import__("subprocess").Popen(["xdg-open", p])
        except Exception: pass

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
