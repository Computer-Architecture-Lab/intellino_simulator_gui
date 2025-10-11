# custom_1.py
import sys, os
ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = os.path.join(ASSETS_DIR, "intellino_TM_transparent.png")
HOME_ICON_PATH = os.path.join(ASSETS_DIR, "home.png")

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QButtonGroup
)
from PySide2.QtCore import QPropertyAnimation, Qt, QSize
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent, QIntValidator
from custom_2 import launch_training_window

# 두 번째 이미지와 같은 큰 테두리 높이(필요시 140~180 사이에서 조정)
SECOND_STYLE_FIXED_HEIGHT = 160

# -----------------------------
# 전역 글꼴(사진 느낌: 산세리프, Medium)
GLOBAL_FONT_QSS = """
* {
    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-weight: 500;
    font-size: 13px;
}
"""

# 공통 버튼 스타일
BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 5px;
        font-weight: 600;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #e9ecef; }
    QPushButton:pressed { background-color: #adb5bd; color: white; }
"""

# 칩/토글 버튼(라운드 사각형)
TOGGLE_BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #C9C9C9;
        border-radius: 10px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #F2F4F6; }
    QPushButton:checked {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #212529;
    }
    QPushButton:pressed { background-color: #E9ECEF; }
"""

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
        font-weight: 600;
        font-size: 13px;
        color: #000000;
    }
"""

# -----------------------------
# 0) TitleBar
class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(56)
        self.setStyleSheet(
            "background-color: #f1f3f5;"
            "border-top-left-radius: 15px; border-top-right-radius: 15px;"
        )
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pixmap = QPixmap(LOGO_PATH).scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon(HOME_ICON_PATH))
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
        # 0 이상 정수만 허용
        self.input.setValidator(QIntValidator(0, 999999))
        self.input.setFixedHeight(35)
        self.input.setStyleSheet("""
            QLineEdit {
                padding: 5px 10px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 13px;
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
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                padding: 8px;
                background-color: #f8f9fa;
                font-size: 13px;
            }
        """)
        # ★ 처음부터 두 번째 이미지 크기(큰 테두리)로 고정
        self.output_box.setFixedHeight(SECOND_STYLE_FIXED_HEIGHT)

        layout.addWidget(self.output_box)
        self.setLayout(layout)

    # exceed 플래그: 초과 시 경고 문구 추가 + 비교기호(≤/>) 전환
    def update_display(self, input_vector_length: int, training_dataset: int, num_classes: int,
                       selected_mem_kb=None, exceed: bool = False):
        if input_vector_length <= 0 or training_dataset <= 0 or num_classes <= 0:
            self.output_box.setText(
                "Please enter valid input vector length, number of training dataset per class, "
                "and number of class to train."
            )
            return

        # 초과 여부에 따라 비교 기호 결정
        comparator = ">" if exceed else "≤"

        # 상단 안내 문구는 기본 부등호(≤)로 유지하고,
        # 사용자가 보게 되는 대입식에는 comparator(≤ 또는 >)를 사용
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
    def __init__(self, prev_window=None):  # ◀ prev_window 추가
        super().__init__()
        self._prev_window = prev_window    # 닫힐 때 복귀할 이전 창
        # Next 활성화: 2~4번 각 입력에 대해 Apply 여부
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

        # 0) 타이틀바
        self.title_bar = TitleBar(self)
        self.title_bar.setParent(container)
        self.title_bar.setGeometry(0, 0, 800, self.title_bar.height())

        # 컨텐츠 레이아웃(스크롤 없이)
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(20, self.title_bar.height() + 8, 20, 12)
        self.main_layout.setSpacing(16)

        # 1~5 블록
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

        # Next 바(항상 화면에 보이도록 고정)
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
                font-weight: 600;
                font-size: 14px;
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

    # 메모리 버튼 클릭: 자동 값 채움 + Apply 상태 초기화 + 텍스트/경고 갱신
    def on_memory_size_selected(self, value_kb: int):
        # 자동 값 채움(Apply는 아직 안 누른 상태로 간주)
        self.category_input.set_value(10)
        self.input_vector_input.set_value(196)
        # ★ 변경: 2K→1, 8K→4, 16K→8
        train_map = {2: 1, 8: 4, 16: 8}
        self.train_data_input.set_value(train_map.get(value_kb, 0))

        # Apply 상태 초기화
        self._applied = {'cat': False, 'train': False, 'vec': False}

        # 텍스트 갱신(Selected(Intellino) 포함) + 초과 검사 반영
        self.update_memory_display()

        # 높이는 이미 고정되어 별도 재계산 없음

    # 각 입력 박스의 Apply 버튼을 눌렀을 때 호출
    def on_input_applied(self, key: str):
        self._applied[key] = True
        self.update_memory_display()

    def update_memory_display(self):
        vec_len = self.input_vector_input.get_value()
        train_num = self.train_data_input.get_value()
        category_num = self.category_input.get_value()
        selected_mem = self.intellino_mem_group.get_selected_size_kbyte()

        # 메모리 한도(바이트)
        thresholds = {2: 2048, 8: 8192, 16: 16384}

        # ★ 변경: 현재 설정값의 곱 = input × train × classes
        product = vec_len * train_num * category_num

        # 초과 여부(메모리 버튼이 선택된 상태에서만 판단)
        exceed = (selected_mem in thresholds) and (product > thresholds[selected_mem])

        # ★ 변경: 5번 회색창 갱신(클래스 개수 포함)
        self.memory_display.update_display(vec_len, train_num, category_num, selected_mem, exceed)

        # Next 활성화: 값 유효 & 2~4번 Apply 완료 & 한도 초과 아님
        all_applied = self._applied['cat'] and self._applied['train'] and self._applied['vec']
        self.next_btn.setEnabled(
            all_applied and vec_len > 0 and train_num > 0 and category_num > 0 and not exceed
        )

    # --- custom_1.py : nextFunction 교체 ---
    def nextFunction(self):
        from custom_2 import launch_training_window  # (상단 import 유지해도 무방)
        launch_training_window(
            num_categories=self.category_input.get_value(),
            samples_per_class=self.train_data_input.get_value(),
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

    # ◀ 창을 '닫을' 때(TitleBar의 home 버튼) 이전 창이 다시 보이도록 처리
    def closeEvent(self, event):
        try:
            if self._prev_window is not None:
                self._prev_window.show()
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_FONT_QSS)

    window = Custom_1_Window()
    window.show()
    sys.exit(app.exec_())
