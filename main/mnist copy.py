# existing_mode_window.py
import sys
import os
import subprocess
import threading
import queue
import runpy

from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, \
    QGraphicsDropShadowEffect, QGroupBox, QProgressBar, QLineEdit, QFileDialog, QTextEdit, QSizePolicy, QStyle
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QTimer, Signal, QPoint

# ──────────────────────────────────────────────
# exe/개발 겸용 리소스 경로 헬퍼
def resource_path(relative_path: str) -> str:
    """
    PyInstaller(onefile) 실행 시 임시 폴더(sys._MEIPASS)와
    개발 환경(__file__ 기준)을 모두 커버.
    빌드 때 dest를 'main'으로 넣은 경우도 자동 탐색.
    """
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
        cand1 = os.path.join(base, relative_path)  # --add-data "...;."
        if os.path.exists(cand1):
            return cand1
        parent = os.path.basename(os.path.dirname(__file__))      # 보통 'main'
        cand2 = os.path.join(base, parent, relative_path)         # --add-data "...;main"
        if os.path.exists(cand2):
            return cand2
    # 개발(스크립트) 환경
    return os.path.join(os.path.dirname(__file__), relative_path)
# ──────────────────────────────────────────────

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = resource_path("intellino_TM_transparent.png")
HOME_ICON_PATH = resource_path("home.png")

# 1. Dataset 섹션 (클래스 분리)
class DatasetSection(QWidget):
    # 어떤 데이터셋이 선택되었는지 메인에 알리기 위한 신호
    mnist_clicked = Signal()
    cifar_clicked = Signal()
    speech_clicked = Signal()

    def __init__(self):
        super().__init__()

        dataset_group = QGroupBox("1. Dataset")
        dataset_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)

        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(10, 5, 10, 5)
        dataset_layout.setSpacing(10)
        dataset_layout.addStretch()

        button_style = """
            QPushButton {
                background-color: #ffffff; border: 1px solid #ccc; border-radius: 10px;
                padding: 5px 5px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background-color: #e9ecef; }
            QPushButton:pressed { background-color: #adb5bd; color: white; }
        """

        self.mnist_btn = QPushButton("MNIST")
        self.mnist_btn.setMinimumSize(QSize(90, 35))
        self.mnist_btn.setStyleSheet(button_style)
        self.mnist_btn.clicked.connect(self.mnist_clicked.emit)

        self.cifar_btn = QPushButton("CIFAR-10")
        self.cifar_btn.setMinimumSize(QSize(110, 35))
        self.cifar_btn.setStyleSheet(button_style)
        self.cifar_btn.clicked.connect(self.cifar_clicked.emit)

        self.speech_btn = QPushButton("Speech Commands")
        self.speech_btn.setMinimumSize(QSize(180, 35))
        self.speech_btn.setStyleSheet(button_style)
        self.speech_btn.clicked.connect(self.speech_clicked.emit)

        dataset_layout.addWidget(self.mnist_btn)
        dataset_layout.addWidget(self.cifar_btn)
        dataset_layout.addWidget(self.speech_btn)
        dataset_layout.addStretch()

        dataset_group.setLayout(dataset_layout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(dataset_group)


# 2. Train 섹션 (최소 높이)
class TrainSection(QWidget):
    def __init__(self):
        super().__init__()

        train_group = QGroupBox("2. Train")
        train_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)
        train_group.setFixedHeight(70)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #bbb; border-radius: 3px; background-color: #f1f1f1; }
            QProgressBar::chunk { background-color: #3b82f6; border-radius: 3px; }
        """)
        self.percent_label = QLabel("0%")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.percent_label.setFixedWidth(40)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.percent_label)
        train_group.setLayout(progress_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(train_group)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.percent_label.setText(f"{value}%")


# 3. Inference 섹션 (최소 높이)
class InferenceSection(QWidget):
    inference_requested = Signal(str)  # 파일 경로를 전달

    def __init__(self):
        super().__init__()

        inference_group = QGroupBox("3. Inference")
        inference_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)
        inference_group.setFixedHeight(80)

        inference_layout = QHBoxLayout()

        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Put in the file to infer")
        self.file_input.setFixedHeight(35)
        self.file_input.setStyleSheet("""
            QLineEdit { border: 1px solid #ccc; border-radius: 8px; padding-left: 10px; font-size: 13px; }
        """)

        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(35, 35)
        browse_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #ccc; border-radius: 8px; background-color: #ffffff; font-weight: bold;
            }
            QPushButton:hover { background-color: #e9ecef; }
        """)
        browse_btn.clicked.connect(self.browse_file)

        start_btn = QPushButton("Start")
        start_btn.setFixedSize(60, 35)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff; border: 1px solid #ccc; border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #e9ecef; }
            QPushButton:pressed { background-color: #adb5bd; color: white; }
        """)
        start_btn.clicked.connect(self.startFunction)

        inference_layout.addWidget(self.file_input)
        inference_layout.addWidget(browse_btn)
        inference_layout.addWidget(start_btn)
        inference_group.setLayout(inference_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(inference_group)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            self.file_input.setText(file_path)

    def startFunction(self):
        file_path = self.file_input.text()
        print(f"[DEBUG] Start button clicked, file = {file_path}")
        self.inference_requested.emit(file_path)


# 4. Result 섹션 (가변 크기 + 스크롤 가능)
class ResultSection(QWidget):
    def __init__(self):
        super().__init__()

        result_group = QGroupBox("4. Result")
        result_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)

        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setText("")
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-size: 14px; border: 1px solid #ccc; border-radius: 8px;
                padding: 10px; background-color: #f8f9fa;
            }
        """)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(result_group)


# ──────────────────────────────────────────────
# Subprocess 대체용: exe 환경에서 모듈을 같은 프로세스에서 실행
class _QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q
    def write(self, s: str):
        if s:
            # 줄 단위로 쪼개서 넣음
            for line in s.splitlines():
                if line.strip():
                    self.q.put(line)
    def flush(self):  # 호환용
        pass


# 메인 창
class SubWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)  # 테두리제거
        self.setAttribute(Qt.WA_TranslucentBackground)           # 창 배경에 투명 적용
        self.setFixedSize(800, 800)                              # 창 크기 고정

        self._is_frozen = bool(getattr(sys, "frozen", False))
        self._train_queue = None
        self._infer_queue = None

        shadow = QGraphicsDropShadowEffect(self)                 # 그림자 효과
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        container = QWidget(self)                                # 둥근 흰색 컨테이너 위젯
        container.setStyleSheet("""
            QWidget { background-color: white; border-radius: 15px; }
        """)
        container.setGeometry(0, 0, 800, 800)

        # 커스텀 타이틀 바
        title_bar = QWidget(container)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet(
            "background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pm = QPixmap(LOGO_PATH)
        if not pm.isNull():
            logo_label.setPixmap(pm.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # 파일이 없을 때 대비
            logo_label.setText("intellino")
            logo_label.setStyleSheet("font-weight:600;")

        close_btn = QPushButton()
        home_icon = QIcon(HOME_ICON_PATH)
        if home_icon.isNull():
            # 표준 홈 아이콘으로 폴백
            home_icon = self.style().standardIcon(QStyle.SP_DirHomeIcon)
        close_btn.setIcon(home_icon)
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34, 34)
        close_btn.setStyleSheet("""
            QPushButton { border: none; background-color: transparent; }
            QPushButton:hover { background-color: #dee2e6; border-radius: 17px; }
        """)

        close_btn.clicked.connect(self.close)

        title_layout.addWidget(logo_label)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 60, 10, 10)

        # 1. Dataset (클래스로 분리된 섹션 사용)
        self.dataset_section = DatasetSection()
        layout.addWidget(self.dataset_section)

        # 2. Train (최소)
        self.train_section = TrainSection()
        layout.addWidget(self.train_section)

        # 3. Inference (최소)
        self.inference_section = InferenceSection()
        layout.addWidget(self.inference_section)
        self.inference_section.inference_requested.connect(self.run_inference)

        # 4. Result (가변 스크롤)
        self.result_section = ResultSection()
        layout.addWidget(self.result_section)

        # Dataset 신호 연결 (지금은 MNIST만 동작 연결)
        self.dataset_section.mnist_clicked.connect(self.mnistFunction)
        # self.dataset_section.cifar_clicked.connect(self.cifarFunction)
        # self.dataset_section.speech_clicked.connect(self.speechFunction)

        self.offset = None
        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

    # ---------- 공통: mnist 모듈 실행(개발=서브프로세스, exe=인프로세스) ----------
    def _run_mnist_dev(self, args, for_infer=False):
        """개발 환경: 기존처럼 subprocess로 실행"""
        script_path = os.path.join(os.path.dirname(__file__), "mnist.py")
        cmd = [sys.executable, script_path] + args
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        if for_infer:
            self.infer_process = proc
            self.result_section.result_text.clear()
            self.infer_timer = QTimer()
            self.infer_timer.timeout.connect(self.read_inference_output)
            self.infer_timer.start(10)
        else:
            self.process = proc
            self.timer = QTimer()
            self.timer.timeout.connect(self.check_progress_output)
            self.timer.start(10)

    def _run_mnist_frozen(self, args, for_infer=False):
        """exe 환경: 같은 프로세스 안에서 'mnist' 모듈을 __main__으로 실행"""
        q = queue.Queue()
        writer = _QueueWriter(q)

        def worker():
            old_out, old_err, old_argv = sys.stdout, sys.stderr, sys.argv
            try:
                sys.stdout = writer
                sys.stderr = writer
                # 'python mnist.py [args...]'와 동일하게 동작하도록 argv 설정
                sys.argv = ["mnist"] + list(args)
                # PyInstaller에 'mnist' 모듈이 포함되어 있어야 함(--hidden-import mnist)
                runpy.run_module("mnist", run_name="__main__")
            except Exception as e:
                print(f"[ERROR] mnist run failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                sys.stdout, sys.stderr, sys.argv = old_out, old_err, old_argv
                q.put(None)  # 종료 신호

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        if for_infer:
            self._infer_queue = q
            self.result_section.result_text.clear()
            self.infer_timer = QTimer()
            self.infer_timer.timeout.connect(self._read_infer_queue)
            self.infer_timer.start(10)
        else:
            self._train_queue = q
            self.timer = QTimer()
            self.timer.timeout.connect(self._read_train_queue)
            self.timer.start(10)

    # ---------- Inference ----------
    def run_inference(self, file_path):
        print(f"[DEBUG] run_inference called with: {file_path}")
        if self._is_frozen:
            # exe: 같은 프로세스에서 mnist 수행
            self._run_mnist_frozen(["infer", file_path], for_infer=True)
        else:
            # 개발: subprocess
            self._run_mnist_dev(["infer", file_path], for_infer=True)

    def _read_infer_queue(self):
        if not self._infer_queue:
            return
        try:
            while True:
                item = self._infer_queue.get_nowait()
                if item is None:
                    self.infer_timer.stop()
                    break
                self.result_section.result_text.append(str(item))
        except queue.Empty:
            pass

    def read_inference_output(self):
        if self.infer_process and self.infer_process.stdout:
            line = self.infer_process.stdout.readline()
            if line:
                self.result_section.result_text.append(line.strip())
            if self.infer_process.poll() is not None:
                self.infer_timer.stop()

    # ---------- Train ----------
    def mnistFunction(self):
        if self._is_frozen:
            self._run_mnist_frozen([], for_infer=False)
        else:
            self._run_mnist_dev([], for_infer=False)

    def _read_train_queue(self):
        if not self._train_queue:
            return
        try:
            while True:
                item = self._train_queue.get_nowait()
                if item is None:
                    self.timer.stop()
                    break
                line = str(item)
                if "progress :" in line:
                    try:
                        percent = int(line.strip().split("progress :")[1])
                        self.train_section.update_progress(percent)
                    except Exception:
                        pass
                # 진행 로그도 결과창에 같이 보여주려면 아래 주석 해제
                # else:
                #     self.result_section.result_text.append(line)
        except queue.Empty:
            pass

    def check_progress_output(self):
        if self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if line:
                if "progress :" in line:
                    try:
                        percent = int(line.strip().split("progress :")[1])
                        self.train_section.update_progress(percent)
                    except ValueError:
                        pass
            if self.process.poll() is not None:
                self.timer.stop()

    # ---------- 이동(창 끌기) ----------
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.offset is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    subwin = SubWindow()
    subwin.show()
    sys.exit(app.exec_())
