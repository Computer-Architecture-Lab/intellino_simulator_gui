import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QGraphicsOpacityEffect
)
from PySide2.QtCore import QPropertyAnimation
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent, QIntValidator
from PySide2.QtCore import Qt, QSize, QPoint
from custom_2 import TrainDatasetGroup, launch_training_window
#from custom_3 import launch_window


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
# 0. TitleBar (절대좌표로 배치, 드래그 이동/닫기 버튼 포함)
class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(50)  # 기존 높이 유지
        self.setStyleSheet(
            "background-color: #f1f3f5; "
            "border-top-left-radius: 15px; border-top-right-radius: 15px;"
        )
        self.setAttribute(Qt.WA_StyledBackground, True)  # ← 이 줄 추가!

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)  # 기존 여백 유지

        logo_label = QLabel()
        # 슬래시(/) 사용해 경로 이스케이프 문제 방지
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
# 공통 입력 박스 베이스
class IntegerInputGroup(QGroupBox):
    def __init__(self, title, example_text, on_apply=None, notice_text=None):
        super().__init__(title)
        self.on_apply_callback = on_apply
        self.setStyleSheet(self._groupbox_style())
        self.setMaximumHeight(130)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        h_layout = QHBoxLayout()
        h_layout.setSpacing(10)

        self.input = QLineEdit()
        self.input.setPlaceholderText(example_text)
        self.input.setValidator(QIntValidator(0, 9999))
        self.input.setFixedSize(600, 35)
        self.input.setStyleSheet(self._input_style())

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedSize(80, 35)
        self.apply_btn.setStyleSheet(BUTTON_STYLE)
        self.apply_btn.clicked.connect(self._on_apply_clicked)

        h_layout.addWidget(self.input)
        h_layout.addWidget(self.apply_btn)
        main_layout.addLayout(h_layout)

        if notice_text:
            notice = QLabel(notice_text)
            notice.setWordWrap(True)
            notice.setStyleSheet("font-size: 11px; color: #555;")
            main_layout.addWidget(notice)

        self.setLayout(main_layout)

    def _on_apply_clicked(self):
        # 값 저장 + 콜백
        try:
            self._value = int(self.input.text())
        except ValueError:
            self._value = 0
        print(f"applied: {self._value}")
        if self.on_apply_callback:
            self.on_apply_callback()

    def get_value(self):
        try:
            return int(self.input.text())
        except ValueError:
            return 0

    def _groupbox_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
        """

    def _input_style(self):
        return """
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 13px;
            }
        """


# -----------------------------
# 1. Category 섹션
class CategoryInputGroup(IntegerInputGroup):
    def __init__(self, on_apply=None):
        super().__init__(
            "1. Number of category to train",
            "ex) 10",
            on_apply=on_apply,
            notice_text=None
        )


# -----------------------------
# 2. Training dataset 섹션
class TrainingInputGroup(IntegerInputGroup):
    def __init__(self, on_apply=None):
        super().__init__(
            "2. Number of training dataset",
            "ex) 1000       Write only one unsigned integer number",
            on_apply=on_apply,
            notice_text=(
                "※ Number of training dataset should be more or equal than number of category to train.\n"
                "※ We recommend preparing at least 100 samples for each category in the training dataset."
            )
        )


# -----------------------------
# 3. Input vector 섹션
class InputVectorGroup(IntegerInputGroup):
    def __init__(self, on_apply=None):
        super().__init__(
            "3. Input vector length",
            "Write only one unsigned integer number",
            on_apply=on_apply,
            notice_text="※ Number of training dataset should be more or equal than number of category to train."
        )


# -----------------------------
# 4. 메모리 계산 결과 섹션
class MemorySizeSection(QGroupBox):
    def __init__(self):
        super().__init__("4. Required memory size")
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
        """)

        layout = QVBoxLayout()
        self.output_box = QTextBrowser()
        self.output_box.setStyleSheet("""
            QTextBrowser {
                border: none;
                padding: 5px;
                background-color: #f9f9f9;
            }
        """)
        layout.addWidget(self.output_box)
        self.setLayout(layout)

    def update_display(self, input_vector_length: int, training_dataset: int):
        if input_vector_length <= 0 or training_dataset <= 0:
            self.output_box.setText("Please enter valid input vector length and number of training dataset.")
            return
        memory_size = input_vector_length * training_dataset
        self.output_box.setText(f"""
    input vector length × number of training dataset ≤ memory size\n
    ⇔ {input_vector_length}Kbyte × {training_dataset} ≤ memory size\n
    ∴ available memory size ≥ {memory_size}Kbyte"""
        )


# -----------------------------
# 메인 윈도우
class Custom_1_Window(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 800)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        container = QWidget(self)
        container.setStyleSheet("background-color: white; border-radius: 15px;")
        container.setGeometry(0, 0, 800, 800)

        # 0) 타이틀바 (절대좌표로 이전과 동일하게)
        self.title_bar = TitleBar(self)
        self.title_bar.setParent(container)
        self.title_bar.setGeometry(0, 0, 800, 50)

        # 본문 레이아웃(기존과 동일 마진/스페이싱)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)  # 타이틀바(50) + 여유(10)
        layout.setSpacing(40)

        # 1~4 블록
        self.category_input = CategoryInputGroup(on_apply=self.update_memory_display)
        layout.addWidget(self.category_input)

        self.train_data_input = TrainingInputGroup(on_apply=self.update_memory_display)
        layout.addWidget(self.train_data_input)

        self.input_vector_input = InputVectorGroup(on_apply=self.update_memory_display)
        layout.addWidget(self.input_vector_input)

        self.memory_display = MemorySizeSection()
        layout.addWidget(self.memory_display)

        layout.addStretch()
        layout.addLayout(self._create_next_button())

    def update_memory_display(self):
        vec_len = self.input_vector_input.get_value()
        train_num = self.train_data_input.get_value()
        category_num = self.category_input.get_value()

        self.memory_display.update_display(vec_len, train_num)

        self.next_btn.setEnabled(vec_len > 0 and train_num > 0 and category_num > 0)

    def _create_next_button(self):
        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(100, 40)
        self.next_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 8px;
                background-color: #fefefe;
            }
            QPushButton:hover {
                background-color: #dee2e6;
            }
        """)
        self.next_btn.clicked.connect(self.nextFunction)
        self.next_btn.setEnabled(False)  # 초기엔 비활성화

        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.next_btn)
        return layout

    # (참고) 타이틀바 드래그는 TitleBar가 처리
    def nextFunction(self):
        num = self.category_input.get_value()
        train_dataset = self.train_data_input.get_value()
        input_vector_len = self.input_vector_input.get_value()
        launch_training_window(num_categories=num)
        # custom_3에서 정보 받아갈 함수 입력

        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(1)
        self.anim.setEndValue(0)
        self.anim.finished.connect(self.close)
        self.anim.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Custom_1_Window()
    window.show()
    sys.exit(app.exec_())
