# custom_3.py
import sys
import os
import subprocess
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog,
    QScrollArea, QTextEdit, QSizePolicy
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QPoint, QTimer, Signal


# -----------------------------
# 공통 버튼 스타일
BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 5px;
        font-weight: bold;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #e9ecef;
    }
    QPushButton:pressed {
        background-color: #adb5bd;
        color: white;
    }
"""


# -----------------------------
# 0. TitleBar (절대좌표 + 드래그 이동)
class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(50)
        self.setStyleSheet(
            "background-color: #f1f3f5; "
            "border-top-left-radius: 15px; border-top-right-radius: 15px;"
        )
        # 스타일 배경 강제 적용 (일부 플랫폼에서 필요)
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pixmap = QPixmap("main/intellino_TM_transparent.png").scaled(
            65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon("main/home.png"))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34, 34)
        close_btn.setStyleSheet("""
            QPushButton { border: none; background-color: transparent; }
            QPushButton:hover { background-color: #dee2e6; border-radius: 17px; }
        """)
        close_btn.clicked.connect(lambda: self._parent and self._parent.close())

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(close_btn)

        self._offset = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._offset is not None and event.buttons() == Qt.LeftButton and self._parent:
            self._parent.move(self._parent.pos() + event.pos() - self._offset)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._offset = None


# -----------------------------
# 7. Train 섹션 (진행률)
class TrainSection(QWidget):
    def __init__(self, title="7. Train", initial=100):
        super().__init__()

        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)
        group.setFixedHeight(70)

        bar_row = QHBoxLayout()
        from PySide2.QtWidgets import QProgressBar  # 지역 import로 충돌 방지
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(initial)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #bbb; border-radius: 3px; background-color: #f1f1f1; }
            QProgressBar::chunk { background-color: #3b82f6; border-radius: 3px; }
        """)
        self.percent_label = QLabel(f"{initial}%")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.percent_label.setFixedWidth(50)

        bar_row.addWidget(self.progress_bar)
        bar_row.addWidget(self.percent_label)
        group.setLayout(bar_row)

        main = QVBoxLayout(self)
        main.addWidget(group)

    def update_progress(self, value: int):
        v = max(0, min(100, int(value)))
        self.progress_bar.setValue(v)
        self.percent_label.setText(f"{v}%")


# -----------------------------
# 8. Inference 섹션
class InferenceSection(QWidget):
    inference_requested = Signal(str)  # 파일 경로 전달

    def __init__(self):
        super().__init__()

        group = QGroupBox("8. Inference")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)
        group.setFixedHeight(80)

        row = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Put in the file to infer")
        self.file_input.setFixedHeight(35)
        self.file_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ccc; border-radius: 8px;
                padding-left: 10px; font-size: 13px;
            }
        """)

        browse_btn = QPushButton("...")
        browse_btn.setFixedSize(35, 35)
        browse_btn.setStyleSheet("""
            QPushButton { border: 1px solid #ccc; border-radius: 8px; background-color: #ffffff; font-weight: bold; }
            QPushButton:hover { background-color: #e9ecef; }
        """)
        browse_btn.clicked.connect(self.browse_file)

        start_btn = QPushButton("Start")
        start_btn.setFixedSize(70, 35)
        start_btn.setStyleSheet(BUTTON_STYLE)
        start_btn.clicked.connect(self._on_start_clicked)

        row.addWidget(self.file_input)
        row.addWidget(browse_btn)
        row.addWidget(start_btn)
        group.setLayout(row)

        main = QVBoxLayout(self)
        main.addWidget(group)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            self.file_input.setText(file_path)

    def _on_start_clicked(self):
        # 파일을 비워두면 mnist.py가 기본 테스트셋으로 추론
        self.inference_requested.emit(self.file_input.text().strip())


# -----------------------------
# 9. Result 섹션
class ResultSection(QWidget):
    def __init__(self):
        super().__init__()

        group = QGroupBox("9. Result")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #b0b0b0; border-radius: 10px;
                margin-top: 10px; padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        """)

        v = QVBoxLayout()
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("""
            QTextEdit {
                font-size: 14px; border: 1px solid #ccc; border-radius: 8px;
                padding: 10px; background-color: #f8f9fa;
            }
        """)
        self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        v.addWidget(self.text)
        group.setLayout(v)

        main = QVBoxLayout(self)
        main.addWidget(group)


# -----------------------------
# 메인 창 (SubWindow 이름 유지: window_2에서 import 하기 좋게)
class SubWindow(QWidget):
    def __init__(self, num_categories: int = 0):
        super().__init__()
        self.num_categories = num_categories  # 필요 시 이후 화면에서 활용
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 800)

        # 컨테이너 (둥근 흰 배경)
        container = QWidget(self)
        container.setStyleSheet("background-color: white; border-radius: 15px;")
        container.setGeometry(0, 0, 800, 800)

        # 그림자
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        # 타이틀바 (절대좌표)
        self.title_bar = TitleBar(self)
        self.title_bar.setParent(container)
        self.title_bar.setGeometry(0, 0, 800, 50)

        # 본문 레이아웃
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)  # 타이틀바 높이 고려
        layout.setSpacing(40)

        # 7, 8, 9 섹션
        self.train_section = TrainSection(initial=100)  # 스샷처럼 100%로 표시
        layout.addWidget(self.train_section)

        self.inference_section = InferenceSection()
        layout.addWidget(self.inference_section)

        self.result_section = ResultSection()
        layout.addWidget(self.result_section)

        # 시그널 연결
        self.inference_section.inference_requested.connect(self.run_inference)

        # 하단 Next 버튼
        layout.addLayout(self._create_next_button())

    def _create_next_button(self):
        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(100, 40)
        self.next_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                border: 1px solid #888; border-radius: 8px;
                background-color: #fefefe;
            }
            QPushButton:hover { background-color: #dee2e6; }
        """)
        # 필요 시 다음 화면으로 연결
        # self.next_btn.clicked.connect(self._go_next)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.next_btn)
        return row

    # -------------------------
    # Inference 수행 (mnist.py 호출)
    def run_inference(self, file_path: str):
        # mnist.py 경로: 현재 파일과 같은 폴더에 있다고 가정 (main/ 안)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mnist_path = os.path.join(script_dir, "mnist.py")

        args = [sys.executable, mnist_path, "infer"]
        if file_path:
            args.append(file_path)

        # 기존 출력 초기화 및 시작 로그
        self.result_section.text.clear()
        #self.result_section.text.append("Inference start")

        # 서브프로세스 실행
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 타이머로 stdout 주기적으로 읽기
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_proc_output)
        self.timer.start(10)

    def _read_proc_output(self):
        if self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if line:
                self.result_section.text.append(line.strip())
            if self.proc.poll() is not None:  # 종료
                # 남은 버퍼 비우기
                rest = self.proc.stdout.read()
                if rest:
                    self.result_section.text.append(rest.strip())
                self.timer.stop()

    # (선택) 다음 화면 연결 자리
    # def _go_next(self):
    #     pass


# 단독 실행 테스트용
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SubWindow(num_categories=0)
    w.show()
    sys.exit(app.exec_())
