# existing_mode_window.py
import sys
import os
import traceback
from utils.path_utils import get_dirs

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGraphicsDropShadowEffect, QGroupBox, QProgressBar, QLineEdit, QFileDialog,
    QTextEdit, QSizePolicy, QStyle
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QTimer, Signal, QThread
#=======================================================================================================#

# exe/개발 공통 리소스 경로 헬퍼
def resource_path(name: str) -> str:
    """
    PyInstaller(onefile)의 _MEIPASS와 개발 환경(__file__)을 모두 커버.
    빌드 시 ;main 으로 넣은 데이터도 자동으로 탐색한다.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, name),             # ;.
        os.path.join(base, "main", name),     # ;main (우리가 쓰는 구조)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),  # dev
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # 마지막 안전장치

LOGO_PATH = resource_path("intellino_TM_transparent.png")
HOME_ICON_PATH = resource_path("home.png")
CUSTOM_IMAGE_ROOT, NUMBER_IMAGE_DIR, _ = get_dirs(__file__)


#=======================================================================================================#
#                                               UI 구성                                                  #
#=======================================================================================================#
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

# 2. Train 섹션 
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

# 3. Inference 섹션 
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

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedSize(60, 35)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff; border: 1px solid #ccc; border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #e9ecef; }
            QPushButton:pressed { background-color: #adb5bd; color: white; }
        """)
        self.start_btn.clicked.connect(self.startFunction)

        inference_layout.addWidget(self.file_input)
        inference_layout.addWidget(browse_btn)
        inference_layout.addWidget(self.start_btn)
        inference_group.setLayout(inference_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(inference_group)

    def browse_file(self):
    # 기본 시작 폴더 결정
        base_dir = NUMBER_IMAGE_DIR

        # NUMBER_IMAGE_DIR가 None 이거나 폴더가 아니면,
        # 현재 파일 기준으로 main/custom_image 같은 느낌으로 fallback
        if not base_dir:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_image")

        if not os.path.isdir(base_dir):
            # 이것도 없으면 최종 fallback: 사용자 홈 디렉토리
            base_dir = os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            base_dir,
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if file_path:
            self.file_input.setText(file_path)

    def startFunction(self):
        file_path = self.file_input.text()
        # print(f"[DEBUG] Start button clicked, file = {file_path}")
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


class MnistTrainWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal()        # 성공적으로 끝났을 때
    failed = Signal(str)   # 에러 발생 시

    def run(self):
        try:
            import mnist

            def progress_cb(p):
                self.progress.emit(int(p))

            def log_cb(msg):
                self.log.emit(str(msg))

            mnist.train(progress_cb=progress_cb, log_cb=log_cb)
            self.done.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

class MnistInferWorker(QThread):
    finished = Signal(int)    # 예측 라벨
    error = Signal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path

    def run(self):
        try:
            import mnist
            label = mnist.infer_image(self.image_path)
            self.finished.emit(label)
        except Exception:
            self.error.emit(traceback.format_exc())


#=======================================================================================================#
#                                                 main                                                  #
#=======================================================================================================#
class SubWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._mnist_train_worker = None
        self._mnist_infer_worker = None
        self._mnist_trained_once = False
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)  # 테두리제거
        self.setAttribute(Qt.WA_TranslucentBackground)           # 창 배경에 투명 적용
        self.setFixedSize(800, 800)                              # 창 크기 고정

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

        # ── 로고
        logo_label = QLabel()
        pm = QPixmap(LOGO_PATH)
        if not pm.isNull():
            logo_label.setPixmap(pm.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # 파일이 없을 때 대비
            logo_label.setText("intellino")
            logo_label.setStyleSheet("font-weight:600;")

        # ── 홈 아이콘 (리소스 없으면 표준 아이콘으로 폴백)
        close_btn = QPushButton()
        home_icon = QIcon(HOME_ICON_PATH)
        if home_icon.isNull():
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
        self.inference_section.start_btn.setEnabled(False)

        # 4. Result (가변 스크롤)
        self.result_section = ResultSection()
        layout.addWidget(self.result_section)

        # Dataset 신호 연결 (지금은 MNIST만 동작 연결)
        self.dataset_section.mnist_clicked.connect(self.mnistFunction)
        # self.dataset_section.cifar_clicked.connect(self.cifarFunction)
        # self.dataset_section.speech_clicked.connect(self.speechFunction)

        # 드래그 이동
        self.offset = None
        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

#=======================================================================================================#
#                                              function                                                 #
#=======================================================================================================#
    # run inference
    def run_inference(self, file_path):
        file_path = file_path.strip()
        if not file_path:
            self.result_section.result_text.append("Select a file before starting inference.")
            return

        if self._mnist_infer_worker is not None and self._mnist_infer_worker.isRunning():
            # 이미 추론 중이면 무시
            return

        # self.result_section.result_text.append(f"[DEBUG] Inference start: {file_path}")
        self.result_section.result_text.append(f"Inference start")

        worker = MnistInferWorker(file_path, parent=self)
        worker.finished.connect(self._on_mnist_infer_finished)
        worker.error.connect(self._on_mnist_infer_error)

        self._mnist_infer_worker = worker
        worker.start()

    def _on_mnist_infer_finished(self, label: int):
        self.result_section.result_text.append(f"Inference result: predict_label = {label}")

    def _on_mnist_infer_error(self, msg: str):
        self.result_section.result_text.append("[ERROR] Inference failed:")
        self.result_section.result_text.append(msg)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.offset is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)

    # Dataset 버튼 핸들러
    def mnistFunction(self):
        # 이미 학습이 진행 중이면 중복 실행 방지
        if self._mnist_train_worker is not None and self._mnist_train_worker.isRunning():
            return

        # 로그 영역 초기화 혹은 메시지 추가
        self.result_section.result_text.append("Starting MNIST training...")
        self.train_section.update_progress(0)

        worker = MnistTrainWorker()
        worker.progress.connect(self.train_section.update_progress)
        # worker.log.connect(lambda msg: self.result_section.result_text.append(msg))
        worker.finished.connect(self._on_mnist_train_finished)
        worker.failed.connect(self._on_mnist_train_error)

        self._mnist_train_worker = worker
        worker.start()

    def _on_mnist_train_finished(self):
        self.result_section.result_text.append("MNIST training has been completed.")

        self._mnist_trained_once = True
        self.inference_section.start_btn.setEnabled(True)

    def _on_mnist_train_error(self, msg: str):
        self.result_section.result_text.append("[ERROR] MNIST 학습 중 오류 발생:")
        self.result_section.result_text.append(msg)

    def check_progress_output(self):
        if self.process.stdout:
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    subwin = SubWindow()
    subwin.show()
    sys.exit(app.exec_())
