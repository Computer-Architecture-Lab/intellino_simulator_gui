import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent, QIntValidator
from PySide2.QtCore import Qt, QSize, QPoint

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

# 1. 메모리 사이즈 선택 박스
class MemorySizeGroup(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setMaximumHeight(90)
        self.setStyleSheet(self._groupbox_style())

        layout = QHBoxLayout(spacing=15)
        layout.setContentsMargins(10, 10, 10, 10)

        for size in ["2KByte", "8KByte", "16KByte"]:
            btn = QPushButton(size)
            btn.setFixedSize(100, 40)
            btn.setStyleSheet(BUTTON_STYLE)
            layout.addWidget(btn)

        self.setLayout(layout)

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


# 2. 숫자 입력 박스 (공통 구조)
class IntegerInputGroup(QGroupBox):
    def __init__(self, title, example_text):
        super().__init__(title)
        self.setMaximumHeight(90)
        self.setStyleSheet(self._groupbox_style())

        layout = QHBoxLayout(spacing=10)
        layout.setContentsMargins(10, 10, 10, 10)

        self.input = QLineEdit()
        self.input.setPlaceholderText(example_text)
        self.input.setValidator(QIntValidator(0, 9999))
        self.input.setFixedSize(600, 35)
        self.input.setStyleSheet(self._input_style())

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedSize(80, 35)
        self.apply_btn.setStyleSheet(BUTTON_STYLE)
        self.apply_btn.clicked.connect(lambda: print(f"{title} applied:", self.input.text()))

        layout.addWidget(self.input)
        layout.addWidget(self.apply_btn)
        self.setLayout(layout)

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


# 3. Training dataset + 안내 텍스트
class TrainingInputGroup(IntegerInputGroup):
    def __init__(self):
        super().__init__("3. Number of training dataset", "ex) 1000       Write only one unsigned integer number")
        self.setMaximumHeight(130)

        notice = QLabel(
            "\n".join([
                "※ Number of training dataset should be less or equal than number of category.",
                "※ We recommend preparing at least 100 samples for each category in the training dataset."
            ])
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("font-size: 11px; color: #555;")

        layout = QVBoxLayout()
        layout.addLayout(self.layout())
        layout.addWidget(notice)
        self.setLayout(layout)


# 4. 메인 SubWindow
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

        self._add_title_bar(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)
        layout.setSpacing(50)

        layout.addWidget(MemorySizeGroup("1. Intellino memory size"))
        layout.addWidget(IntegerInputGroup("2. Number of category to train", "ex) 10      Write only one unsigned integer number"))
        layout.addWidget(TrainingInputGroup())

        layout.addStretch()
        layout.addLayout(self._create_next_button())

    def _add_title_bar(self, parent):
        title_bar = QWidget(parent)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet("background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")

        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pixmap = QPixmap("main/intellino_TM_transparent.png").scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon("main/home.png"))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34, 34)
        close_btn.setStyleSheet(("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #dee2e6;
                border-radius: 17px;
            }
            """))
        close_btn.clicked.connect(self.close)

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(close_btn)

        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

    def _create_next_button(self):
        next_btn = QPushButton("Next")
        next_btn.setFixedSize(100, 40)
        next_btn.setStyleSheet("""
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
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(next_btn)
        return layout

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if hasattr(self, 'offset') and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Custom_1_Window()
    window.show()
    sys.exit(app.exec_())
