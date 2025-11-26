import sys
import os

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QButtonGroup,
    QStyle,
)
from PySide2.QtCore import QPropertyAnimation, Qt, QSize
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent, QIntValidator

from custom_2 import launch_training_window
from utils.resource_utils import resource_path
from utils.ui_common import TitleBar, BUTTON_STYLE, TOGGLE_BUTTON_STYLE

# 두 번째 이미지와 같은 큰 테두리 높이
SECOND_STYLE_FIXED_HEIGHT = 260

# 메모리 검토 시 사용할 "최소" 기준값
MIN_INPUT_VECTOR_LENGTH = 64       # input vector length 최소
MIN_SAMPLES_PER_CLASS = 2          # sample dataset per class 최소

# Intellino 메모리 크기(KB → Byte)
INTELLINO_MEMORY_BYTES = {
    2: 2048,
    8: 8192,
    16: 16384,
}

# ── GroupBox 스타일 ──
GROUPBOX_WITH_FLOATING_TITLE = """
    QGroupBox {
        border: 1px solid #b0b0b0;
        border-radius: 10px;
        margin-top: 10px;
        padding: 10px;
        background: transparent;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        background-color: white;
        color: #000000;
    }
"""

# -----------------------------
# 공통 입력 박스 베이스
class IntegerInputGroup(QGroupBox):
    def __init__(self, title, example_text, on_apply=None, notice_text=None):
        super().__init__(title)
        self.on_apply_callback = on_apply
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE)
        self.setMaximumHeight(130)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(6)

        h_layout = QHBoxLayout()
        h_layout.setSpacing(10)

        self.input = QLineEdit()
        self.input.setPlaceholderText(example_text)
        self.input.setValidator(QIntValidator(0, 999999))
        self.input.setFixedHeight(35)
        self.input.setStyleSheet("""
            QLineEdit {
                padding: 5px 10px;
                border: 1px solid #ccc;
                border-radius: 8px;
            }
        """)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedSize(80, 35)
        self.apply_btn.setStyleSheet(BUTTON_STYLE)
        self.apply_btn.clicked.connect(self._on_apply_clicked)

        h_layout.addWidget(self.input)
        h_layout.addWidget(self.apply_btn)
        self.main_layout.addLayout(h_layout)

        if notice_text:
            notice = QLabel(notice_text)
            notice.setWordWrap(True)
            notice.setStyleSheet("font-size: 11px; color: #555;")
            self.main_layout.addWidget(notice)

        self.setLayout(self.main_layout)

    def _on_apply_clicked(self):
        if self.on_apply_callback:
            self.on_apply_callback()

    def get_value(self):
        try:
            return int(self.input.text())
        except ValueError:
            return 0

    def set_value(self, value: int, trigger_update: bool = False):
        self.input.setText(str(value))
        if trigger_update and self.on_apply_callback:
            self.on_apply_callback()


class CategoryInputGroup(IntegerInputGroup):
    def __init__(self, on_apply=None):
        super().__init__("2. Number of class to train", "ex) 10", on_apply=on_apply)
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE + "QGroupBox { font-weight: bold; }")

        # max: N 표시용 라벨
        self.max_label = QLabel("max: -")
        self.max_label.setStyleSheet("font-size: 11px; color: #888;")
        self.main_layout.addWidget(self.max_label)

    def set_max(self, max_value: int):
        if max_value is None or max_value <= 0:
            self.max_label.setText("max: -")
        else:
            self.max_label.setText(f"max: {max_value}")


# -----------------------------
# 1. Intellino memory size
class IntellinoMemorySizeGroup(QGroupBox):
    def __init__(self, on_select=None):
        super().__init__("1. Intellino memory size")
        self.on_select = on_select
        self._selected_kbyte = None
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE + "QGroupBox { font-weight: bold; }")

        wrap = QHBoxLayout()
        wrap.setContentsMargins(14, 10, 14, 10)
        wrap.setSpacing(14)

        self.btn_2k = QPushButton("2KByte")
        self.btn_8k = QPushButton("8KByte")
        self.btn_16k = QPushButton("16KByte")
        for b in (self.btn_2k, self.btn_8k, self.btn_16k):
            b.setCheckable(True)
            b.setStyleSheet(TOGGLE_BUTTON_STYLE)
            b.setFixedHeight(34)
            b.setCursor(Qt.PointingHandCursor)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.addButton(self.btn_2k, 2)
        self.group.addButton(self.btn_8k, 8)
        self.group.addButton(self.btn_16k, 16)
        self.group.buttonClicked[int].connect(self._on_clicked)

        wrap.addStretch()
        wrap.addWidget(self.btn_2k)
        wrap.addWidget(self.btn_8k)
        wrap.addWidget(self.btn_16k)
        wrap.addStretch()
        self.setLayout(wrap)

    def _on_clicked(self, value_kbyte: int):
        self._selected_kbyte = value_kbyte
        if self.on_select:
            self.on_select(value_kbyte)

    def get_selected_size_kbyte(self):
        return self._selected_kbyte


# -----------------------------
# 3. Required memory size
class MemorySizeSection(QGroupBox):
    def __init__(self):
        super().__init__("3. Required memory size")
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE + "QGroupBox { font-weight: bold; }")

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        self.output_box = QTextBrowser()
        self.output_box.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 8px;
                background-color: #f8f9fa;
            }
        """)
        self.output_box.setFixedHeight(SECOND_STYLE_FIXED_HEIGHT)

        layout.addWidget(self.output_box)
        self.setLayout(layout)

    def update_display(self, input_vector_length: int, training_dataset: int, num_classes: int,
                       selected_mem_kb=None, exceed: bool = False):
        if num_classes <= 0:
            self.output_box.setText(
                "Please select memory size and enter a valid number of class to train."
            )
            return

        comparator = ">" if exceed else "≤"

        text = (
            "Assuming minimum settings:\n"
            f"  input vector length ≥ {input_vector_length}\n"
            f"  number of sample dataset per class ≥ {training_dataset}\n\n"
            "input vector length × number of sample dataset per class × "
            "number of class to train ≤ memory size\n\n"
            f"⇒ {input_vector_length} × {training_dataset} × {num_classes} "
            f"{comparator} memory size\n"
        )
        if selected_mem_kb is not None:
            text += f"\nSelected(Intellino): {selected_mem_kb}KByte"
        if exceed:
            text += "\n\nEven with minimum settings, this number of classes exceeds the memory size."

        self.output_box.setText(text)


# -----------------------------
# 메인 윈도우
class Custom_1_Window(QWidget):
    def __init__(self, prev_window=None):
        super().__init__()
        self._prev_window = prev_window
        self._applied = {'cat': False}
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

        self.title_bar = TitleBar(self)
        self.title_bar.setParent(container)
        self.title_bar.setGeometry(0, 0, 800, self.title_bar.height())

        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(20, self.title_bar.height() + 8, 20, 12)
        self.main_layout.setSpacing(16)

        # 1. 메모리 선택
        self.intellino_mem_group = IntellinoMemorySizeGroup(on_select=self.on_memory_size_selected)
        self.main_layout.addWidget(self.intellino_mem_group)

        # 2. class 입력
        self.category_input = CategoryInputGroup(on_apply=lambda: self.on_input_applied('cat'))
        self.main_layout.addWidget(self.category_input)

        # 3. Required memory size 표시
        self.memory_display = MemorySizeSection()
        self.main_layout.addWidget(self.memory_display)

        # Next 버튼 바
        self.next_bar = self._create_next_bar()
        self.main_layout.addWidget(self.next_bar)

    def _create_next_bar(self):
        bar = QWidget()
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch()

        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(100, 40)
        self.next_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #888;
                border-radius: 8px;
                background-color: #fefefe;
            }
            QPushButton:hover { background-color: #dee2e6; }
        """)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.nextFunction)

        h.addWidget(self.next_btn)
        bar.setLayout(h)
        return bar

    def _compute_max_classes_for_memory(self, mem_kb: int):
        if mem_kb is None:
            return None

        mem_bytes = INTELLINO_MEMORY_BYTES.get(mem_kb)
        if mem_bytes is None:
            return None

        per_class_bytes_min = MIN_INPUT_VECTOR_LENGTH * MIN_SAMPLES_PER_CLASS
        if per_class_bytes_min <= 0:
            return None

        return mem_bytes // per_class_bytes_min

    def on_memory_size_selected(self, value_kb: int):
        max_classes = self._compute_max_classes_for_memory(value_kb)
        self.category_input.set_max(max_classes)

        self._applied = {'cat': False}
        self.update_memory_display()

    def on_input_applied(self, key: str):
        self._applied[key] = True
        self.update_memory_display()

    def update_memory_display(self):
        selected_mem = self.intellino_mem_group.get_selected_size_kbyte()
        category_num = self.category_input.get_value()

        product_min = MIN_INPUT_VECTOR_LENGTH * MIN_SAMPLES_PER_CLASS * category_num
        exceed = (
            selected_mem in INTELLINO_MEMORY_BYTES
            and product_min > INTELLINO_MEMORY_BYTES[selected_mem]
        )

        # ★ 여기 오타 수정: self.memory_display
        self.memory_display.update_display(
            MIN_INPUT_VECTOR_LENGTH,
            MIN_SAMPLES_PER_CLASS,
            category_num,
            selected_mem,
            exceed
        )

        all_applied = self._applied['cat']
        self.next_btn.setEnabled(
            all_applied
            and selected_mem is not None
            and category_num > 0
            and not exceed
        )

    def nextFunction(self):
        launch_training_window(
            num_categories=self.category_input.get_value(),
            samples_per_class=MIN_SAMPLES_PER_CLASS,
            input_vector_length=MIN_INPUT_VECTOR_LENGTH,
            selected_mem_kb=self.intellino_mem_group.get_selected_size_kbyte(),
            prev_window=self
        )

        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self.anim = QPropertyAnimation(effect, b"opacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)

        def _after_fade():
            self.setGraphicsEffect(None)
            self.hide()

        self.anim.finished.connect(_after_fade)
        self.anim.start()

    def closeEvent(self, event):
        try:
            if self._prev_window is not None:
                self._prev_window.show()
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Custom_1_Window()
    window.show()
    sys.exit(app.exec_())
