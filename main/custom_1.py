import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFrame
)
from PySide2.QtGui import (
    QPixmap, QIcon, QColor, QMouseEvent, QIntValidator
)
from PySide2.QtCore import Qt, QSize, QPoint

# 1. 메모리 사이즈 선택 박스
class MemorySizeGroup(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setMaximumHeight(90)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
        """)

        layout = QHBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        for size in ["2KByte", "8KByte", "16KByte"]:
            btn = QPushButton(size)
            btn.setFixedSize(100, 40)
            btn.setStyleSheet("""
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
                QPushButton:pressed {
                    background-color: #adb5bd;
                    color: white;
                }
            """)
            layout.addWidget(btn)

        self.setLayout(layout)


# 2. 숫자 입력 박스 (공통 구조)
class IntegerInputGroup(QGroupBox):
    def __init__(self, title, example_text):
        super().__init__(title)
        self.setMaximumHeight(90)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
        """)

        # 입력 필드와 버튼을 포함한 HBoxLayout
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        input_layout.setContentsMargins(10, 10, 10, 10)

        self.input = QLineEdit()
        self.input.setPlaceholderText(example_text)
        self.input.setValidator(QIntValidator(0, 9999))
        self.input.setFixedHeight(35)
        self.input.setFixedWidth(600)
        self.input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 13px;
            }
        """)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedSize(80, 35)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                border: 1px solid #888;
                border-radius: 8px;
                font-size: 13px;
                background-color: #fefefe;
            }
            QPushButton:hover {
                background-color: #dee2e6;
            }
        """)
        self.apply_btn.clicked.connect(lambda: print(f"{title} applied:", self.input.text()))

        input_layout.addWidget(self.input)
        input_layout.addWidget(self.apply_btn)

        # ✅ 별도의 QWidget으로 묶어서 외부에서도 레이아웃으로 넣을 수 있게 함
        self.input_row_widget = QWidget()
        self.input_row_widget.setLayout(input_layout)

        # 기본 레이아웃
        base_layout = QVBoxLayout()
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.addWidget(self.input_row_widget)
        self.setLayout(base_layout)

    def __init__(self, title, example_text):
        super().__init__(title)
        self.setMaximumHeight(90)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
        """)

        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self.input = QLineEdit()
        self.input.setPlaceholderText(example_text)
        self.input.setValidator(QIntValidator(0, 9999))
        self.input.setFixedHeight(35)
        self.input.setFixedWidth(600)
        self.input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 13px;
            }
        """)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedSize(80, 35)
        apply_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                border: 1px solid #888;
                border-radius: 8px;
                font-size: 13px;
                background-color: #fefefe;
            }
            QPushButton:hover {
                background-color: #dee2e6;
            }
        """)
        apply_btn.clicked.connect(lambda: print(f"{title} applied:", self.input.text()))

        layout.addWidget(self.input)
        layout.addWidget(apply_btn)
        self.setLayout(layout)


# 3. Training dataset + 안내 텍스트
class TrainingInputGroup(IntegerInputGroup):
    def __init__(self):
        super().__init__("3. Number of training dataset", "ex) 1000       Write only one unsigned integer number")
        self.setMaximumHeight(130)

        notice = QLabel("※ Number of training dataset should be less or equal than number of category.\n"
                        "※ We recommend preparing at least 100 samples for each category in the training dataset.")
        notice.setWordWrap(True)
        notice.setStyleSheet("font-size: 11px; color: #555;")

        # 새 레이아웃 구성
        new_layout = QVBoxLayout()
        new_layout.setContentsMargins(10, 10, 10, 10)
        new_layout.setSpacing(5)

        new_layout.addWidget(self.input_row_widget)  # 기존 입력 줄
        new_layout.addWidget(notice)                 # 안내 문구 추가

        self.setLayout(new_layout)

    def __init__(self):
        super().__init__("3. Number of training dataset", "ex) 1000       Write only one unsigned integer number")
        self.setMaximumHeight(110)

        # 안내 문구
        notice = QLabel("※ Number of training dataset should be less or equal than number of category.\n"
                        "※ We recommend preparing at least 100 samples for each category in the training dataset.")
        notice.setWordWrap(True)
        notice.setStyleSheet("font-size: 11px; color: #555;")

        # 원래 레이아웃에 수직으로 추가
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.layout())
        main_layout.addWidget(notice)
        self.setLayout(main_layout)


# 4. 메인 SubWindow
class SubWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 600)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        container = QWidget(self)
        container.setStyleSheet("background-color: white; border-radius: 15px;")
        container.setGeometry(0, 0, 800, 800)

        title_bar = QWidget(container)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet("background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pixmap = QPixmap("main\intellino_TM_transparent.png").scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon("main\home.png"))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34, 34)
        close_btn.setStyleSheet("border: none; background-color: transparent;")
        close_btn.clicked.connect(self.close)

        title_layout.addWidget(logo_label)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)

        # ✅ 메인 레이아웃
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)
        layout.setSpacing(50)            # 그룹 박스 간격

        # ✅ 그룹박스 위젯들 추가
        layout.addWidget(MemorySizeGroup("1. Intellino memory size"))
        layout.addWidget(IntegerInputGroup("2. Number of category to train", "ex) 10      Write only one unsigned integer number"))
        layout.addWidget(TrainingInputGroup())

        # ⛳ 반드시 추가!
        layout.addStretch()  # 👈 이거 없으면 Next가 가려짐!

        # ✅ Next 버튼 우하단에 배치
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
        next_layout = QHBoxLayout()
        next_layout.addStretch()
        next_layout.addWidget(next_btn)
        layout.addLayout(next_layout)

        # 드래그 이동
        self.offset = None
        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.offset and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubWindow()
    window.show()
    sys.exit(app.exec_())
