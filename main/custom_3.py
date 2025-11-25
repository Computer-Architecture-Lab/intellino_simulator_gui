# custom_3.py — train/test split with accuracy + native file dialog


import sys, os, shutil, datetime, numpy as np, traceback, pickle, cv2
from PIL import Image  # (남겨두지만 학습·평가·추론은 모두 OpenCV 기반 전처리 사용)
from utils.resource_utils import resource_path
from utils.image_preprocess import preprocess_digit_image
from utils.ui_common import TitleBar, BUTTON_STYLE


# 추론 경로 제어(기본 False: 어느 경로든 허용, 단 학습 파일 차단)
INFER_ONLY_FROM_TEST = False


from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QSizePolicy, QGraphicsOpacityEffect,
    QLineEdit, QFileDialog, QProgressBar, QTextBrowser, QMessageBox, QStyle
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor, QTextCursor
from PySide2.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve

from path_utils import get_dirs
# ▶ 실험 상태 전역 객체(커스텀4에서 정의)
from custom_4 import EXPERIMENT_STATE

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# 원래 경로 계산
CUSTOM_IMAGE_ROOT, NUMBER_IMAGE_DIR, DEFAULT_OUTPUT_ROOT = get_dirs(__file__)

# exe에서 경로가 없을 때(또는 구조가 다른 경우) 보정 시도
try:
    main_dir = os.path.dirname(resource_path("intellino_TM_transparent.png"))  # 보통 '_MEIPASS/main'
    if not os.path.isdir(CUSTOM_IMAGE_ROOT):
        cand = os.path.join(main_dir, "custom_image")
        if os.path.isdir(cand):
            CUSTOM_IMAGE_ROOT = cand
    if not os.path.isdir(NUMBER_IMAGE_DIR):
        cand = os.path.join(main_dir, "custom_image", "number_image")
        if os.path.isdir(cand):
            NUMBER_IMAGE_DIR = cand
except Exception:
    pass

def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test.tmp")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False

def _resolve_output_root(preferred: str) -> str:
    """
    저장 루트가 패키지 내부(읽기 전용)일 수 있으니,
    쓰기 가능한 사용자 폴더로 자동 폴백.
    우선순위: preferred -> %LOCALAPPDATA%/iCore -> 사용자 홈/iCore_runs
    """
    candidates = []
    if preferred:
        candidates.append(preferred)
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        candidates.append(os.path.join(localapp, "iCore"))
    candidates.append(os.path.join(os.path.expanduser("~"), "iCore_runs"))

    for c in candidates:
        if _is_writable_dir(c):
            return c
    # 마지막 최후의 보루: 현재 작업 디렉터리
    fallback = os.path.join(os.getcwd(), "iCore_runs")
    os.makedirs(fallback, exist_ok=True)
    return fallback

# K-MEANS clustering function
def kmeans_clustering(vectors: np.ndarray, num_select: int,
                          max_iter: int = 15, random_state: int = 0):
    """
    vectors : (N, D) float32 배열 (각 이미지 벡터)
    num_select : 뽑고 싶은 샘플 개수 (클러스터 개수)
    return : 선택된 인덱스 리스트 (길이 num_select)
    """
    if vectors is None or vectors.size == 0:
        return []

    n_samples = vectors.shape[0]
    if num_select >= n_samples:
        # 전체가 필요한 개수 이하이면 그냥 전부 사용
        return list(range(n_samples))

    k = num_select
    rs = np.random.RandomState(random_state)

    # 1) 초기 centroid: 임의로 k개 골라 사용
    init_idx = rs.choice(n_samples, size=k, replace=False)
    centroids = vectors[init_idx].copy()

    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # 2) 각 샘플을 가장 가까운 centroid에 할당 (L2 거리)
        dists = np.linalg.norm(
            vectors[:, None, :] - centroids[None, :, :],
            axis=2
        )  # shape (N, k)
        new_labels = dists.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

        # 3) centroid 업데이트 (각 클러스터 평균)
        for c in range(k):
            mask = (labels == c)
            if not np.any(mask):
                # 비어 있는 클러스터는 임의 샘플 하나로 재초기화
                centroids[c] = vectors[rs.randint(0, n_samples)]
            else:
                centroids[c] = vectors[mask].mean(axis=0)

    # 4) 각 클러스터에서 centroid에 가장 가까운 샘플 1개씩 대표로 선택
    chosen = []
    for c in range(k):
        mask = (labels == c)
        if not np.any(mask):
            continue
        sub_idx = np.where(mask)[0]
        sub_vecs = vectors[sub_idx]
        diff = sub_vecs - centroids[c]
        d2 = np.einsum("ij,ij->i", diff, diff)  # 제곱거리
        best_local = sub_idx[d2.argmin()]
        chosen.append(int(best_local))

    # 혹시 어떤 이유로 k개 못 뽑았으면 나머지는 아무거나 채우기
    if len(chosen) < k:
        remain = [i for i in range(n_samples) if i not in chosen]
        chosen += remain[: (k - len(chosen))]

    return chosen

MODEL_BASENAME = "custom_model.pkl"

# ---------------------------
# 전처리(단일 파이프라인)
# → utils.image_preprocess.preprocess_digit_image 를 사용합니다.

# ---------------------------
def load_images_from_dir(dir_path: str):
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))
             if str(f).lower().endswith(IMG_EXTS)]
    X, keep = [], []
    for p in files:
        try:
            vec = preprocess_digit_image(p)
            X.append(vec); keep.append(os.path.abspath(p))
        except Exception:
            pass
    if not X:
        return np.empty((0,784), dtype=np.float32), []
    return np.stack(X, axis=0), keep


def vectorize_like_training(path: str):
    """과거 함수명 유지: 공통 숫자 전처리 래퍼"""
    try:
        return preprocess_digit_image(path)
    except Exception:
        return None

# 과거 함수명 유지(외부 코드 의존 대비)
def preprocess_user_image(image_path: str) -> np.ndarray:
    """공통 숫자 전처리 래퍼"""
    return preprocess_digit_image(image_path)

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

class ProgressSection(QWidget):
    def __init__(self, title="7. Train"):
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
        title = "9. Inference" if not INFER_ONLY_FROM_TEST else "8. Inference (use datasets/test)"
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
        # 출력 루트는 반드시 쓰기 가능한 곳으로 해석
        self.output_root = _resolve_output_root(output_root)
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

    # ------------ helpers ------------
    def _ok(self, text:str):   self.result.add_block(f"<span class='ok'>{text}</span>")
    def _info(self, text:str): self.result.add_block(f"<span class='info'>{text}</span>")
    def _hint(self, text:str): self.result.add_block(f"<span class='dim'>{text}</span>")
    def _err(self, text:str):  self.result.add_block(f"<span class='err'>{text}</span>")

    def _norm(self, p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

    def _is_training_file(self, p: str) -> bool:
        npth = self._norm(p)
        return (npth in self._train_originals) or (npth in self._train_copies)

    def _is_subpath(self, child: str, parent: str) -> bool:
        """child 가 parent 하위 경로인지 안전하게 확인(드라이브 상이 예외 대응)."""
        try:
            child_real  = os.path.realpath(child)
            parent_real = os.path.realpath(parent)
            return os.path.commonpath([child_real, parent_real]) == parent_real
        except Exception:
            return False

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

    # ------------ UI ------------
    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setFixedSize(800,800)

        container = QWidget(self); container.setGeometry(0,0,800,800)
        container.setStyleSheet("background-color:white; border-radius:15px;")
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(30); shadow.setColor(QColor(0,0,0,100))
        self.setGraphicsEffect(shadow)

        self.title_bar = TitleBar(self); self.title_bar.setParent(container); self.title_bar.setGeometry(0,0,800,50)

        lay = QVBoxLayout(container); lay.setContentsMargins(20,60,20,20); lay.setSpacing(20)

        self.progress = ProgressSection("7. Train"); lay.addWidget(self.progress)

        out_g = QGroupBox("8. Output folder"); out_g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )
        row = QHBoxLayout()
        self.out_label = QLabel("")
        self.out_label.setStyleSheet("font-size:13px;")
        self.open_btn = QPushButton("Open folder"); self.open_btn.setFixedSize(110,32); self.open_btn.setStyleSheet(BUTTON_STYLE)
        self.open_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self.out_label); row.addStretch(); row.addWidget(self.open_btn); out_g.setLayout(row)
        lay.addWidget(out_g)

        self.infer = InferenceSection(); lay.addWidget(self.infer)
        self.infer.browse_btn.clicked.connect(self._browse_infer_file)
        self.infer.start_btn.clicked.connect(self._start_inference)

        res_g = QGroupBox("10. Result"); res_g.setStyleSheet(
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

    # ------------ Train/Test split + Train + Eval ------------
    def _run_kmeans_and_train(self):
        # 저장 루트(쓰기 가능) 확보
        output_base = _resolve_output_root(self.output_root)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(output_base, ts)
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
            # 로깅: 라벨-디렉터리 매핑 및 수량
            self._info(f"[{i}] label='{label}' dir='{dir_path}' → images={len(srcs)}")

            n_samples = len(srcs)
            if n_samples == 0:
                self.progress.update(int(i/total*100))
                continue

            # 이 클래스에서 뽑을 개수 k
            k = min(self.samples_per_class, n_samples)

            # K-means로 대표 인덱스 선택
            chosen_idx = kmeans_clustering(X, k)
            chosen_set = set(chosen_idx)

            # k-means 선별된 train data 복사
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

            self._info(f"    → train {k} / test {len(srcs)-k}")
            self.progress.update(int(i/total*100))

        self.progress.update(100)

        self.result.clear()
        self._ok("K-means selection completed.")
        self._info("Training dataset prepared at <code>datasets/train/</code>.")
        if self._test_items:
            self._info("Test dataset prepared at <code>datasets/test/</code>.")
        else:
            self._hint("No test images were available; accuracy cannot be computed.")
        self.result.add_hr()

        # --- Train ---
        try:
            self._info("Training on datasets/train …")
            self.model.fit_from_root(self._train_dir)
            self.model.save(os.path.join(save_root, MODEL_BASENAME))
            self._ok("Training completed.")
        except Exception as e:
            self._err(f"Training failed: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
            return

        # --- Eval on test ---
        try:
            if self._test_items:
                correct, total = 0, 0
                for p, true_lab in self._test_items:
                    vec = vectorize_like_training(p)  # 공통 전처리 래퍼 사용
                    if vec is None:
                        continue
                    pred_lab = self.model.predict(vec, top_k=1)[0][0]
                    total += 1
                    if pred_lab == true_lab:
                        correct += 1
                if total > 0:
                    acc = 100.0 * correct / total
                    self._last_accuracy = float(acc)
                    # 실험 상태에 누적
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
                in_test_root = bool(test_root and self._is_subpath(rp, test_root))
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
            vec = preprocess_user_image(image_path)  # 공통 전처리 래퍼
            top_labels, top_dists = self.model.predict(vec, top_k=3)
            img_name = os.path.basename(image_path)
            pred = top_labels[0]

            self.result.add_block("🔎 <b>Inference</b>")
            if self._last_accuracy is not None:
                self._hint(f"Last test accuracy (datasets/test): {self._last_accuracy:.2f}%")
            self.result.add_block(f"Input: <code>{img_name}</code>")
            self.result.add_block(f"Prediction: <span class='pred'>{pred}</span>")

            # 개별 추론 정오 즉시 표시 (datasets/test/<label>/... 에서 선택한 경우)
            test_root = os.path.realpath(self._test_dir) if self._test_dir else ""
            rp = os.path.realpath(image_path)
            if test_root and self._is_subpath(rp, test_root):
                gt = os.path.basename(os.path.dirname(rp))  # 폴더명이 정답 라벨
                ok = (pred == gt)
                self.result.add_block(
                    f"Ground truth: <b>{gt}</b> → {'Correct' if ok else 'Wrong'}"
                )

            rows = "".join(
                f"<tr><td>{i}</td><td>{lab}</td><td>{dist:.3f}</td></tr>"
                for i, (lab, dist) in enumerate(zip(top_labels, top_dists), start=1)
            )
            self.result.add_block(
                f"<div class='dim' style='font-weight:600;margin-top:4px;margin-bottom:2px;'>Top-{len(top_labels)} nearest</div>"
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
        """
        ★ 변경 포인트: 이전 창 스냅샷 대신 '흰색 오버레이'로 페이드아웃
        → 새 창(ExperimentWindow) 위에 백색 위젯을 덮고 서서히 투명하게 만들어
           이전 화면 텍스트가 비칠 여지를 없앰.
        """
        try:
            from custom_4 import ExperimentWindow as Window4
        except Exception as e:
            self._err(f"Failed to import next window: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
            return

        try:
            win4 = Window4(num_categories=getattr(self, "num_categories", 0))
            try:
                win4.setGeometry(self.geometry())
            except Exception:
                win4.move(self.pos())
            win4.show(); win4.raise_(); win4.activateWindow()

            # ── 흰색 오버레이(스냅샷 사용 안 함) ─────────────────────────────
            overlay = QWidget(win4)
            overlay.setStyleSheet("background:#ffffff;")  # 완전 흰색
            overlay.setGeometry(win4.rect())             # 새 창 전체 덮기
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            overlay.raise_(); overlay.show()

            eff = QGraphicsOpacityEffect(overlay); overlay.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(180)                         # 필요시 0~300 범위로 조절
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.InOutQuad)

            def _done():
                overlay.deleteLater()
                self.hide()  # 이전 창 감추기

            self._overlay_anim = anim
            anim.finished.connect(_done)
            anim.start()
            # ────────────────────────────────────────────────────────────────

        except Exception as e:
            self._err(f"Failed to open next window: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SubWindow(selection=[], samples_per_class=1)
    w.show()
    sys.exit(app.exec_())
