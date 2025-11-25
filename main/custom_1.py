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

# 두 번째 이미지와 같은 큰 테두리 높이(필요시 140~180 사이에서 조정)
SECOND_STYLE_FIXED_HEIGHT = 160

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
# 공통 입력 박스 베이스(2~4)
class IntegerInputGroup(QGroupBox):
    def __init__(self, title, example_text, on_apply=None, notice_text=None):
        super().__init__(title)
        self.on_apply_callback = on_apply
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE)
        self.setMaximumHeight(130)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

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
        main_layout.addLayout(h_layout)

        if notice_text:
            notice = QLabel(notice_text)
            notice.setWordWrap(True)
            notice.setStyleSheet("font-size: 11px; color: #555;")
            main_layout.addWidget(notice)

        self.setLayout(main_layout)

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


class TrainingInputGroup(IntegerInputGroup):
    def __init__(self, on_apply=None):
        super().__init__(
            "3. Number of training dataset per class",
            "ex) 10       Write only one unsigned integer number",
            on_apply=on_apply,
            notice_text=(
                "※ Number of training dataset should be more or equal than number of category to train.\n"
                "※ We recommend preparing at least 100 samples for each category in the training dataset."
            )
        )
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE + "QGroupBox { font-weight: bold; }")


class InputVectorGroup(IntegerInputGroup):
    def __init__(self, on_apply=None):
        super().__init__(
            "4. Input vector length",
            "Write only one unsigned integer number",
            on_apply=on_apply,
            notice_text="※ Number of training dataset should be more or equal than number of category to train."
        )
        self.setStyleSheet(GROUPBOX_WITH_FLOATING_TITLE + "QGroupBox { font-weight: bold; }")


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
# 5. Required memory size
class MemorySizeSection(QGroupBox):
    def __init__(self):
        super().__init__("5. Required memory size")
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
        if input_vector_length <= 0 or training_dataset <= 0 or num_classes <= 0:
            self.output_box.setText(
                "Please enter valid input vector length, number of training dataset per class, "
                "and number of class to train."
            )
            return

        comparator = ">" if exceed else "≤"

        text = (
            "input vector length × number of training dataset per class × number of class to train ≤ memory size\n\n"
            f"⇔ {input_vector_length} × {training_dataset} × {num_classes} {comparator} memory size\n"
        )
        if selected_mem_kb is not None:
            text += f"\nSelected(Intellino): {selected_mem_kb}KByte"
        if exceed:
            text += "\n\nThe configuration value exceeds the memory size."

        self.output_box.setText(text)


# -----------------------------
# 메인 윈도우 — 스크롤 없이 ‘한 화면’에 맞춤
class Custom_1_Window(QWidget):
    def __init__(self, prev_window=None):
        super().__init__()
        self._prev_window = prev_window
        self._applied = {'cat': False, 'train': False, 'vec': False}
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

        self.intellino_mem_group = IntellinoMemorySizeGroup(on_select=self.on_memory_size_selected)
        self.main_layout.addWidget(self.intellino_mem_group)

        self.category_input = CategoryInputGroup(on_apply=lambda: self.on_input_applied('cat'))
        self.main_layout.addWidget(self.category_input)

        self.train_data_input = TrainingInputGroup(on_apply=lambda: self.on_input_applied('train'))
        self.main_layout.addWidget(self.train_data_input)

        self.input_vector_input = InputVectorGroup(on_apply=lambda: self.on_input_applied('vec'))
        self.main_layout.addWidget(self.input_vector_input)

        self.memory_display = MemorySizeSection()
        self.main_layout.addWidget(self.memory_display)
        self.main_layout.addWidget(self._create_next_bar())

    def _create_next_bar(self):
        bar = QWidget()
        h = QHBoxLayout(bar)
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
        self.next_bar = bar
        return bar

    def on_memory_size_selected(self, value_kb: int):
        self.category_input.set_value(10)
        self.input_vector_input.set_value(196)
        train_map = {2: 1, 8: 4, 16: 8}
        self.train_data_input.set_value(train_map.get(value_kb, 0))

        self._applied = {'cat': False, 'train': False, 'vec': False}
        self.update_memory_display()

    def on_input_applied(self, key: str):
        self._applied[key] = True
        self.update_memory_display()

    def update_memory_display(self):
        vec_len = self.input_vector_input.get_value()
        train_num = self.train_data_input.get_value()
        category_num = self.category_input.get_value()
        selected_mem = self.intellino_mem_group.get_selected_size_kbyte()
        thresholds = {2: 2048, 8: 8192, 16: 16384}
        product = vec_len * train_num * category_num
        exceed = (selected_mem in thresholds) and (product > thresholds[selected_mem])

        self.memory_display.update_display(vec_len, train_num, category_num, selected_mem, exceed)

        all_applied = self._applied['cat'] and self._applied['train'] and self._applied['vec']
        self.next_btn.setEnabled(
            all_applied and vec_len > 0 and train_num > 0 and category_num > 0 and not exceed
        )

    def nextFunction(self):
        launch_training_window(
            num_categories=self.category_input.get_value(),
            samples_per_class=self.train_data_input.get_value(),
            input_vector_length=self.input_vector_input.get_value(),
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
