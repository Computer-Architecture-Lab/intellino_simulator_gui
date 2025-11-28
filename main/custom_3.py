import sys, os, shutil, datetime, numpy as np, traceback, pickle, cv2
import math
import time
from PIL import Image  # (남겨두지만 학습·평가·추론은 모두 OpenCV 기반 전처리 사용)
from utils.resource_utils import resource_path
from utils.image_preprocess import preprocess_user_image
from utils.ui_common import TitleBar, BUTTON_STYLE


from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QSizePolicy, QGraphicsOpacityEffect,
    QProgressBar, QTextBrowser
)
from PySide2.QtGui import QColor, QTextCursor, QPainter, QPen, QBrush
from PySide2.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from utils.path_utils import get_dirs

from intellino.core.neuron_cell import NeuronCells
#=======================================================================================================#

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# 원래 경로 계산
CUSTOM_IMAGE_ROOT, NUMBER_IMAGE_DIR, DEFAULT_OUTPUT_ROOT = get_dirs(__file__)

# exe에서 경로가 없을 때(또는 구조가 다른 경우) 보정 시도
try:
    main_dir = os.path.dirname(resource_path("image/intellino_TM_transparent.png"))  # 보통 '_MEIPASS/main'
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
#=======================================================================================================#
#                                              function                                                 #
#=======================================================================================================#

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

# 전처리(단일 파이프라인)
# → utils.image_preprocess.preprocess_digit_image 를 사용합니다.
def load_images_from_dir(dir_path: str, length_of_input_vector: int):
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))
             if str(f).lower().endswith(IMG_EXTS)]
    X, keep = [], []
    for p in files:
        try:
            vec = preprocess_user_image(p, length_of_input_vector)
            X.append(vec); keep.append(os.path.abspath(p))
        except Exception:
            pass
    if not X:
        return np.empty((0,784), dtype=np.float32), []
    return np.stack(X, axis=0), keep

# Intellino train
def train(neuron_cells,
          train_samples,
          number_of_neuron_cells,
          length_of_input_vector,
          save_path: str = "temp/trained_neuron.pkl",
          progress_callback=None):
    resize_size = int(math.sqrt(length_of_input_vector))
    total = min(len(train_samples), number_of_neuron_cells)

    trained = 0
    is_finish_flag = False

    for idx, (img_path, label) in enumerate(train_samples):
        if trained >= number_of_neuron_cells:
            break

        # data 대신 이미지 파일에서 PIL Image 로드
        data = Image.open(img_path).convert("RGB")
        numpy_image = np.array(data)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
        flatten_image = resized_image.reshape(1, -1).squeeze()

        # label 타입 정리 (문자열 "0" 같은 것도 int로 변환 시도)
        try:
            target_label = int(label)
        except (ValueError, TypeError):
            target_label = label

        is_finish = neuron_cells.train(vector=flatten_image, target=target_label)

        trained += 1

        # 학습 진행률 (실제로 학습한 개수 기준)
        progress = int(trained / number_of_neuron_cells * 100)
        if progress_callback is not None:
            # GUI(progress bar) 업데이트용 콜백
            progress_callback(progress)
        else:
            # 콜백이 없으면 기존처럼 콘솔 출력
            if trained % 4 == 0:
                print(f"progress : {progress}", flush=True)
                time.sleep(0.01)

        # 이거 is_finish가 True가 안될수도 있을거같은데...
        if is_finish:
            print("train finish")
            is_finish_flag = True
            break

    # 학습 완료 후 모델 저장
    if save_path:
        # temp 폴더 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(neuron_cells, f)
        print(f"neuron_cells saved to: {save_path}")

    return is_finish_flag

# Intellino infer
def infer(neuron_cells,
          image_path: str,
          length_of_input_vector: int):
   
    if neuron_cells is None:
        raise RuntimeError("NeuronCells is not initialized.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File does not exist: {image_path}")

    resize_size = int(math.sqrt(length_of_input_vector))

    # train()과 동일한 전처리
    data = Image.open(image_path).convert("RGB")
    numpy_image = np.array(data)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(opencv_image, dsize=(resize_size, resize_size))
    flatten_image = resized_image.reshape(1, -1).squeeze()

    # Intellino 추론
    predict_label = neuron_cells.inference(vector=flatten_image)
    return predict_label

#=======================================================================================================#
#                                               UI 구성                                                  #
#=======================================================================================================#

# 그래프
class ExperimentState:
    def __init__(self):
        self.runs = []

    def add_run(self, label, acc):
        self.runs.append((label, acc))

    def clear(self):
        self.runs.clear()
# 지워도 됨
    def get_all(self):
        return self.runs

EXPERIMENT_STATE = ExperimentState()

class ProgressSection(QWidget):
    def __init__(self, title="7. Storange & inference"):
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


class BarChartWidget(QWidget):
    """
    간단한 막대 그래프 위젯
    - x축: label 리스트 (S 값, samples per class)
    - y축: value 리스트 (accuracy %, 값이 없을 때는 None)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels = []
        self._values = []
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, labels, values, global_max=None):
        self._labels = labels
        self._values = values
        self._global_max = global_max
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 그래프를 약간 아래로 내리기 위해 top=20, bottom=25 정도로 조정
        rect = self.rect().adjusted(40, 20, -20, -25)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        # y축 범위는 항상 0 ~ 100 고정
        min_val = 0.0
        max_val = 100.0

        # 축 그리기 기본 설정
        axis_pen = QPen(QColor(80, 80, 80))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)

        x0 = rect.left()
        y0 = rect.bottom()
        x1 = rect.right()
        y1 = rect.top()

        # y축 (0 ~ 100)
        painter.drawLine(x0, y0, x0, y1)
        # x축
        painter.drawLine(x0, y0, x1, y0)

        # y축 눈금(0, 100)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(x0 - 30, y0 + 2, 28, 12, Qt.AlignRight | Qt.AlignVCenter, "0")
        painter.drawText(x0 - 30, y1 - 6, 28, 12, Qt.AlignRight | Qt.AlignVCenter, "100")

        n = len(self._labels)
        if n == 0:
            return

        bar_space = rect.width() / max(n, 1)
        bar_width = bar_space * 0.5

        # x축 라벨(S 값)
        painter.setPen(QPen(QColor(60, 60, 60)))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        for i, label in enumerate(self._labels):
            cx = x0 + (i + 0.5) * bar_space
            text = f"S={label}"
            painter.drawText(cx - 20, y0 + 4, 40, 16, Qt.AlignHCenter | Qt.AlignTop, text)

        # 막대 및 퍼센트 텍스트 그리기
        bar_brush = QBrush(QColor(59, 130, 246))  # 파란색 계열
        max_brush  = QBrush(QColor(38, 220, 38))  # 최고값: 초록색

        valid_values = [v for v in self._values if v is not None]
        max_value = max(valid_values) if valid_values else None

        for i, (label, val) in enumerate(zip(self._labels, self._values)):
            if val is None:
                continue  # 데이터 없는 S

            # 값에 비례한 높이 (0~100 기준)
            ratio = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.0
            ratio = max(0.0, min(1.0, ratio))
            h = ratio * rect.height()

            cx = x0 + (i + 0.5) * bar_space
            top = y0 - h
            left = cx - bar_width / 2

            # 막대
            painter.setPen(Qt.NoPen)
            if self._global_max is not None and val == self._global_max:
                painter.setBrush(max_brush)   # 전역 최고 정확도 → 초록색
            else:
                painter.setBrush(bar_brush)   # 나머지 → 파란색
            painter.drawRect(left, top, bar_width, h)

            # 막대 위에 퍼센트 텍스트 (좌우 안 잘리게 bar_space 기준으로 넓게)
            painter.setPen(QPen(QColor(30, 30, 30)))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            txt = f"{val:.1f}%"

            text_width = bar_space * 0.9
            text_left = cx - text_width / 2
            painter.drawText(
                text_left,
                top - 16,
                text_width,
                14,
                Qt.AlignHCenter | Qt.AlignBottom,
                txt,
            )

        # y축 단위 라벨 "A"
        #   - 살짝 왼쪽(x0-25), 살짝 위(y1+10) 쪽으로 조정
        painter.setPen(QPen(QColor(60, 60, 60)))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(
            x0 - 25,   # 더 왼쪽
            y1 + 10,   # 약간 위로
            20,
            16,
            Qt.AlignRight | Qt.AlignVCenter,
            "A",
        )

class ExperimentGraphSection(QWidget):
    """
    6. Experiment graph
    - 벡터 길이별(64,128,256,512) × samples_per_class(2,4,6,8,10) 정확도 표시
    - 막대 그래프로 표시 (표 대신)
    - x축: S (samples per class), y축: accuracy (%)
    """
    def __init__(self):
        super().__init__()
        g = QGroupBox("6. Experiment graph")
        g.setStyleSheet(
            "QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;"
            "margin-top:10px;padding:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}"
        )

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(g)

        inner = QVBoxLayout()
        g.setLayout(inner)

        from PySide2.QtWidgets import QGridLayout
        grid = QGridLayout()
        self.graph_boxes = {}  # vec_len -> BarChartWidget

        vecs = [64, 128, 256, 512]
        for idx, v in enumerate(vecs):
            box = QGroupBox(f"Vector length = {v}")
            box_layout = QVBoxLayout(box)

            chart = BarChartWidget()
            chart.setMinimumHeight(200)  # 세로축 더 크게
            chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            box_layout.addWidget(chart)
            self.graph_boxes[v] = chart

            r = idx // 2
            c = idx % 2
            grid.addWidget(box, r, c)

        inner.addLayout(grid)

        # 🔽🔽🔽 이 부분만 이렇게 변경
        self.memory_label = QLabel(
            "Selected memory size: (not specified)   A: accuracy   S: sample dataset"
        )
        self.memory_label.setStyleSheet("font-size: 13px; color: #444; margin-top: 8px;")
        self.memory_label.setAlignment(Qt.AlignLeft)
        inner.addWidget(self.memory_label)

        self.log_view = ResultView()
        self.log_view.setVisible(False)

    def update_graph(self, results: dict, vec_lengths, k_list):
        """
        results[(vec_len, k)] = accuracy(%)

        - 무조건 S = 2,4,6,8,10 다섯 개 기준으로 x축 폭을 맞춘다.
        - 각 vec_len 그래프마다 S 라벨은 항상 5개가 다 보이게 하고,
          해당 S에 대한 결과가 없으면 value 는 None으로 둔다.
        """

        all_values = []
        for v in results.values():
            if v is not None:
                all_values.append(v)
        global_max = max(all_values) if all_values else None

        all_s = [2, 4, 6, 8, 10]  # S 값 다섯 개 고정

        for v in vec_lengths:
            chart = self.graph_boxes.get(v)
            if chart is None:
                continue

            labels = [str(s) for s in all_s]
            values = []

            for s in all_s:
                key = (v, s)
                if key in results:
                    values.append(results[key])   # accuracy 값
                else:
                    values.append(None)           # 데이터 없음 → 막대/퍼센트 없음

            chart.set_data(labels, values, global_max=global_max)

#=======================================================================================================#
#                                                 main                                                  #
#=======================================================================================================#
class SubWindow(QWidget):
    def __init__(self, selection, samples_per_class: int = 1, prev_window=None, output_root=DEFAULT_OUTPUT_ROOT, exp_params=None):
        super().__init__()
        self.selection = selection
        self.samples_per_class = max(1, int(samples_per_class))
        # 출력 루트는 반드시 쓰기 가능한 곳으로 해석
        self.output_root = _resolve_output_root(output_root)
        self.prev_window = prev_window

        self.exp_params = exp_params or {}

        self.num_classes = int(self.exp_params.get("num_classes", len(self.selection)))

        # 입력 벡터 길이 
        self.length_of_input_vector = int(self.exp_params.get("input_vec_len", 784))
        # 클래스당 샘플 수 
        self.samples_per_class = int(self.exp_params.get("samples_per_class", self.samples_per_class))
        # number_of_neuron_cells = num_classes × samples_per_class
        self.number_of_neuron_cells = self.num_classes * self.samples_per_class
        self.num_categories = len(self.selection)

        self.train_samples = []
        self.neuron_cells = None
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

#=======================================================================================================#
#                                              function                                                 #
#=======================================================================================================#
    def _ok(self, text:str):   self.result.add_block(f"<span class='ok'>{text}</span>")
    def _info(self, text:str): self.result.add_block(f"<span class='info'>{text}</span>")
    def _hint(self, text:str): self.result.add_block(f"<span class='dim'>{text}</span>")
    def _err(self, text:str):  self.result.add_block(f"<span class='err'>{text}</span>")

    def _norm(self, p: str) -> str:
        return os.path.normcase(os.path.abspath(p))

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


    # UI setup
    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setFixedSize(800,800)

        container = QWidget(self); container.setGeometry(0,0,800,800)
        container.setStyleSheet("background-color:white; border-radius:15px;")
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(30); shadow.setColor(QColor(0,0,0,100))
        self.setGraphicsEffect(shadow)

        self.title_bar = TitleBar(self); self.title_bar.setParent(container); self.title_bar.setGeometry(0,0,800,50)

        lay = QVBoxLayout(container); lay.setContentsMargins(20,60,20,20); lay.setSpacing(20)

        # 5. Train (progress bar)
        self.progress = ProgressSection("5. Storage & Inference"); lay.addWidget(self.progress)

        # 6. Experiment graph (그래프 + 숨겨진 로그)
        self.graph_section = ExperimentGraphSection()
        lay.addWidget(self.graph_section)

        # 로그용 핸들 (기존 self.result를 그래프 섹션의 log_view로 매핑)
        # log_view 는 화면에는 보이지 않음
        self.result = self.graph_section.log_view

        # 아래에 Next 버튼만 남김
        btn_row = QHBoxLayout()
        self.next_btn = QPushButton("Next"); self.next_btn.setFixedSize(110,38); self.next_btn.setStyleSheet(BUTTON_STYLE)
        self.next_btn.clicked.connect(self._go_next); self.next_btn.setEnabled(False)
        btn_row.addStretch(); btn_row.addWidget(self.next_btn); lay.addLayout(btn_row)

    # select dataset + sample/test split + K-means + Train + Eval (20개 실험 + 메모리 체크)
    def _run_kmeans_and_train(self):
        # 저장 루트(쓰기 가능) 확보
        output_base = _resolve_output_root(self.output_root)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(output_base, ts)
        os.makedirs(save_root, exist_ok=True)
        self._last_save_root = save_root

        datasets_root = os.path.join(save_root, "datasets")
        sample_root   = os.path.join(datasets_root, "sample")
        train_root    = os.path.join(datasets_root, "train")
        test_root     = os.path.join(datasets_root, "test")
        os.makedirs(sample_root, exist_ok=True)
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root,  exist_ok=True)
        self._datasets_root = datasets_root
        self._train_dir     = train_root
        self._test_dir      = test_root

        # 공통 상태 초기화
        self._train_originals.clear()
        self._train_copies.clear()
        self._test_items.clear()
        self.train_samples.clear()

        # ────────────────── 1단계: 각 클래스 폴더 → sample(8), test(2) 비율로 먼저 나누기 ──────────────────
        total = max(1, len(self.selection))

        for i, item in enumerate(self.selection, start=1):
            dir_path = item["dir"]
            label    = str(item["label"])

            # 원본 폴더에서 이미지 경로 수집
            try:
                srcs = [
                    os.path.join(dir_path, f)
                    for f in sorted(os.listdir(dir_path))
                    if str(f).lower().endswith(IMG_EXTS)
                ]
            except Exception as e:
                self._err(f"[{i}] Failed to read directory: {dir_path} ({e})")
                continue

            n_samples = len(srcs)
            self._info(f"[{i}] label='{label}' dir='{dir_path}' → images={n_samples}")

            if n_samples == 0:
                continue

            # 8:2 비율로 인덱스 split (최소 1장 sample, 1장 test 되도록 보정)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            sample_cnt = int(round(n_samples * 0.8))
            if sample_cnt <= 0:
                sample_cnt = 1
            if sample_cnt >= n_samples and n_samples > 1:
                sample_cnt = n_samples - 1
            test_cnt = n_samples - sample_cnt

            sample_idx = indices[:sample_cnt]
            test_idx   = indices[sample_cnt:]

            dst_sample_label = os.path.join(sample_root, label)
            dst_test_label   = os.path.join(test_root, label)
            os.makedirs(dst_sample_label, exist_ok=True)
            os.makedirs(dst_test_label,   exist_ok=True)

            # 8비율 → sample 폴더로 복사 (훈련 후보)
            for idx_ in sample_idx:
                src  = srcs[idx_]
                base = os.path.basename(src)
                dst  = os.path.join(dst_sample_label, base)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    self._hint(f"Failed to copy sample: {src} ({e})")

            # 2비율 → test 폴더로 복사 (평가용)
            for idx_ in test_idx:
                src  = srcs[idx_]
                base = os.path.basename(src)
                dst  = os.path.join(dst_test_label, base)
                try:
                    shutil.copy2(src, dst)
                    self._test_items.append((dst, label))
                except Exception as e:
                    self._hint(f"Failed to copy test: {src} ({e})")

            self._info(f"    → sample {sample_cnt} / test {test_cnt} (≈8:2 split)")

        # split 끝났으니 게이지를 0으로 초기화
        self.progress.update(0)
        self.result.clear()
        self._ok("Sample/Test split (≈8:2) completed for each category.")
        if not self._test_items:
            self._hint("No test images were available; accuracy may not be computed.")
        self.result.add_hr()

        # ────────────────── 2단계: 20개 실험 (vector_length × samples_per_class) ──────────────────
        vec_lengths = [64, 128, 256, 512]
        k_list      = [2, 4, 6, 8, 10]

        num_classes = self.num_classes
        memory_kb   = self.exp_params.get("memory_kb", None)
        memory_bytes = memory_kb * 1024 if memory_kb is not None else None

        any_run = False  # 실제로 학습/평가가 한 번이라도 수행됐는지 여부

        # 그래프용 결과 저장: 실제로 학습+평가한 조합만 기록
        results = {}

        # 전체에서 메모리 조건을 만족하는 실험 개수 미리 계산
        valid_pairs = []
        for vec_len in vec_lengths:
            for k in k_list:
                if memory_bytes is not None:
                    required = num_classes * vec_len * k
                    if required > memory_bytes:
                        continue
                valid_pairs.append((vec_len, k))

        total_experiments = max(1, len(valid_pairs))
        done_experiments = 0

        for vec_len in vec_lengths:
            for k in k_list:
                # ── 메모리 부등식: num_classes × vector_length × samples_per_class ≤ memory_size ──
                if memory_bytes is not None:
                    required = num_classes * vec_len * k
                    if required > memory_bytes:
                        self._hint(
                            f"Skip V{vec_len}, S{k}: "
                            f"{num_classes}×{vec_len}×{k}={required} > memory({memory_bytes} bytes)"
                        )
                        # 이 조합은 학습/추론 생략 (게이지에도 포함 안 됨)
                        continue

                any_run = True
                experiment_attempted = True

                try:
                    self._info(f"=== Experiment: V{vec_len}, S{k} ===")

                    # 현재 실험 파라미터 반영
                    self.length_of_input_vector = vec_len
                    self.samples_per_class      = k
                    self.number_of_neuron_cells = self.num_classes * self.samples_per_class

                    # 이전 실험의 train 관련 상태 초기화
                    self._train_originals.clear()
                    self._train_copies.clear()
                    self.train_samples.clear()

                    # ── 2-1) sample 폴더에서만 K-means로 k개 선별 → train 폴더에 저장 ──
                    for i, item in enumerate(self.selection, start=1):
                        label = str(item["label"])
                        src_sample_label = os.path.join(sample_root, label)
                        if not os.path.isdir(src_sample_label):
                            continue

                        # sample 폴더에서 이미지 벡터 로드
                        X, sample_paths = load_images_from_dir(src_sample_label, length_of_input_vector=self.length_of_input_vector)
                        n_samples = len(sample_paths)
                        if n_samples == 0:
                            continue

                        k_eff = min(self.samples_per_class, n_samples)
                        if k_eff <= 0:
                            continue

                        # K-means로 대표 인덱스 선택
                        chosen_idx = kmeans_clustering(X, k_eff)

                        dst_train_label = os.path.join(train_root, f"V{vec_len}_S{k}", label)
                        os.makedirs(dst_train_label, exist_ok=True)

                        for r, idx in enumerate(chosen_idx, start=1):
                            src  = sample_paths[idx]   # sample 폴더 안의 이미지
                            base = os.path.basename(src)
                            dst  = os.path.join(dst_train_label, f"{label}_sel{r:02d}_{base}")
                            try:
                                shutil.copy2(src, dst)
                                self._train_originals.add(self._norm(src))
                                self._train_copies.add(self._norm(dst))
                                self.train_samples.append((dst, label))
                            except Exception as e:
                                self._hint(f"Failed to copy train (V{vec_len},S{k}): {src} ({e})")

                        self._info(
                            f"[V{vec_len},S{k}] label='{label}' → train(selected from sample) {k_eff} / sample total {n_samples}"
                        )

                    # 학습에 쓸 샘플이 없으면 이 조합은 스킵
                    if not self.train_samples:
                        self._hint(f"No train samples for V{vec_len}, S{k}; skip training.")
                        self.result.add_hr()
                        continue

                    # ── 2-2) Train (Intellino) ──
                    try:
                        self._info(f"Training Intellino (V{vec_len}, S{k}) …")

                        # 1) NeuronCells 인스턴스 생성
                        self.neuron_cells = NeuronCells(
                            number_of_neuron_cells=self.number_of_neuron_cells,
                            length_of_input_vector=self.length_of_input_vector,
                            measure="manhattan"
                        )

                        # train 내부 진행률은 전체 게이지와 의미가 안 맞으니 콜백 제거
                        train(
                            neuron_cells=self.neuron_cells,
                            train_samples=self.train_samples,
                            number_of_neuron_cells=self.number_of_neuron_cells,
                            length_of_input_vector=self.length_of_input_vector,
                            progress_callback=None
                        )
                    except Exception as e:
                        self._err(f"Training failed (V{vec_len}, S{k}): {e}")
                        self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
                        self.result.add_hr()
                        continue

                    # ── 2-3) Eval on test ──
                    try:
                        acc = 0.0
                        if self._test_items and (self.neuron_cells is not None):
                            correct, total = 0, 0
                            for p, true_lab in self._test_items:
                                try:
                                    pred_lab = infer(
                                        neuron_cells=self.neuron_cells,
                                        image_path=p,
                                        length_of_input_vector=self.length_of_input_vector,
                                    )
                                except Exception:
                                    continue

                                total += 1
                                if str(pred_lab) == str(true_lab):
                                    correct += 1

                            if total > 0:
                                acc = 100.0 * correct / total
                                self._last_accuracy = float(acc)

                                memory_kb = self.exp_params.get("memory_kb", None)
                                if memory_kb is not None:
                                    param_label = f"V{vec_len} / C{num_classes} / S{k} / M{memory_kb}K"
                                else:
                                    param_label = f"V{vec_len} / C{num_classes} / S{k}"

                                EXPERIMENT_STATE.add_run(param_label, float(acc))

                                self.result.add_block(f"<b>Test evaluation (Intellino) — {param_label}</b>")
                                self.result.add_block(
                                    f"Accuracy: <b>{acc:.2f}%</b> "
                                    f"(<code>{correct}</code>/<code>{total}</code>)"
                                )
                            else:
                                self._hint(f"(V{vec_len},S{k}) Test dataset exists but no readable images; accuracy cannot be computed.")
                        else:
                            self._hint(f"(V{vec_len},S{k}) No test items or model; evaluation skipped.")

                        # 그래프용 결과 저장: 실제로 학습+평가까지 수행된 조합만 기록
                        results[(vec_len, k)] = acc
                        self.result.add_hr()
                    except Exception as e:
                        self._err(f"Evaluation failed (V{vec_len}, S{k}): {e}")
                        self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
                        self.result.add_hr()
                        continue

                finally:
                    # 이 (vec_len, k) 실험이 한 번 시도되었으면, 전체 진행률 갱신
                    if experiment_attempted and (vec_len, k) in valid_pairs:
                        done_experiments += 1
                        global_p = int(done_experiments * 100 / total_experiments)
                        self.progress.update(global_p)

        # ────────────────── 그래프 업데이트 ──────────────────
        self.graph_section.update_graph(results, vec_lengths, k_list)
        memory_kb = self.exp_params.get("memory_kb", None)
        if memory_kb is not None:
            self.graph_section.memory_label.setText(
                f"Selected memory size: {memory_kb} KB   A: accuracy   S: sample dataset"
            )
        else:
            self.graph_section.memory_label.setText(
                "Selected memory size: (not specified)   A: accuracy   S: sample dataset"
            )

        # ────────────────── 3단계: 전체 실험 요약 ──────────────────
        if not any_run:
            self._hint("No experiment satisfied memory constraint; no training/inference was performed.")
        self.result.add_hr()
        # 모든 실험 끝난 시점에 사실상 100%에 도달해 있음
        self.progress.update(100)
        self.next_btn.setEnabled(True)

         # ───────────────── 4단계: best_results 생성 ──────────────
        best_root = os.path.join(self._last_save_root, "best_results")
        os.makedirs(best_root, exist_ok=True)

        # 4-1) 이번 실행에서 나온 실험 중에서만 "전역 최고 정확도" 찾기
        best_acc = None
        best_pairs = []  # [(vec_len, k), ...]

        for (vec_len, k), acc in results.items():
            # acc가 0.0일 수도 있으니 None 체크만 한다
            if acc is None:
                continue
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_pairs = [(vec_len, k)]
            elif acc == best_acc:
                best_pairs.append((vec_len, k))

        # 이번 실행에서 유효한 실험이 하나도 없으면 종료
        if best_acc is None or not best_pairs:
            self._hint("No valid experiment result; best_results is empty.")
            return

        num_classes = self.num_classes
        memory_kb   = self.exp_params.get("memory_kb", None)

        # 4-2) best_pairs 에 해당하는 train 이미지들만 best_results로 복사
        for (vec_len, k) in best_pairs:
            # 폴더 이름 문자열 생성 (예: "V64 / C5 / S4 / M8K")
            if memory_kb is not None:
                label_str = f"V{vec_len} / S{k} / C{num_classes} / M{memory_kb}K"
            else:
                label_str = f"V{vec_len} / S{k} / C{num_classes}"

            folder_label = label_str.replace(" ", "").replace("/", "_")
            dst_folder = os.path.join(best_root, folder_label)

            # 기존 폴더가 있으면 통째로 삭제 후 다시 생성 (동일 실행 내 재실행 대비)
            if os.path.isdir(dst_folder):
                shutil.rmtree(dst_folder)
            os.makedirs(dst_folder, exist_ok=True)

            # 이 실험에서 사용된 train 폴더 경로
            train_src = os.path.join(
                self._train_dir,
                f"V{vec_len}_S{k}"
            )

            # train_src 내부에 각 label(0~9 등) 디렉토리가 있음
            if os.path.isdir(train_src):
                for label_dir in sorted(os.listdir(train_src)):
                    src_label_path = os.path.join(train_src, label_dir)
                    if not os.path.isdir(src_label_path):
                        continue

                    dst_label_path = os.path.join(dst_folder, label_dir)
                    os.makedirs(dst_label_path, exist_ok=True)

                    for f in os.listdir(src_label_path):
                        src_path = os.path.join(src_label_path, f)
                        dst_path = os.path.join(dst_label_path, f)
                        try:
                            shutil.copy2(src_path, dst_path)
                        except Exception as e:
                            self._hint(f"Failed to copy to best_results: {src_path} ({e})")

        # 최종 best_results 경로 저장
        self._best_results_root = best_root
 
    def _go_next(self):
        try:
            from custom_4 import ExperimentWindow as Window4
        except Exception as e:
            self._err(f"Failed to import next window: {e}")
            self.result.add_block(f"<pre class='dim'>{traceback.format_exc()}</pre>")
            return

        try:
            win4 = Window4(
                num_categories=getattr(self, "num_categories", 0),
                best_results_root=self._best_results_root
            )
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
