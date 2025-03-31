# from custom_3 import launch_training_window

# # 예시: 사용자로부터 숫자를 입력받음 (GUI든 콘솔이든)
# num = int(input("카테고리 개수 입력: "))

# # custom_3.py의 창을 실행하면서 개수 전달
# launch_training_window(num)

from PySide2.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit, QTextBrowser,
    QVBoxLayout, QHBoxLayout, QGroupBox, QApplication, QGraphicsDropShadowEffect
)
from PySide2.QtGui import QPixmap, QIcon, QFont, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QPoint
import os

class InputVectorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 800)

        # 그림자 효과 추가
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        # 배경 컨테이너 위젯
        container = QWidget(self)
        container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
            }
        """)
        container.setGeometry(0, 0, 800, 800)

        # 타이틀바
        title_bar = QWidget(container)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet("background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        base_dir = os.path.dirname(__file__)
        logo_label = QLabel()
        logo_path = os.path.join(base_dir, "intellino_TM_transparent.png")
        if os.path.exists(logo_path):
            logo_label.setPixmap(QPixmap(logo_path).scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        close_btn = QPushButton()
        close_btn.setIcon(QIcon(os.path.join(base_dir, "home.png")))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34, 34)
        close_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #dee2e6;
                border-radius: 17px;
            }
        """)
        close_btn.clicked.connect(self.close)

        title_layout.addWidget(logo_label)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)

        # 본문 레이아웃
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 60, 20, 20)
        main_layout.setSpacing(20)

        # Section 4: Available input vector length
        available_group = QGroupBox("4. Available input vector length")
        available_group.setStyleSheet("""
            QGroupBox {
                 font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        available_layout = QVBoxLayout()
        available_box = QTextBrowser()
        available_box.setText(
            "계산 결과 표시"
            
        )
        available_box.setStyleSheet("""
            QTextBrowser {
                border: none;
                padding: 5px;
                background-color: #f9f9f9;
            }
        """)
        available_layout.addWidget(available_box)
        available_group.setLayout(available_layout)
        main_layout.addWidget(available_group)

        main_layout.addSpacing(30)                   #4번 groupbox와 5번 groupbox사이 간격 조절절

        # Section 5: Input vector length
        input_group = QGroupBox("5. Input vector length")
        input_group.setStyleSheet("""
            QGroupBox {
                 font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 10px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        input_layout = QVBoxLayout()
        input_line = QLineEdit()
        input_line.setFixedHeight(35)
        input_line.setPlaceholderText("Write only one unsigned integer number")
        input_line.setStyleSheet("""
            QLineEdit {
                font-size: 13px;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding-left: 10px;
                background-color: #ffffff;
            }
        """)
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedSize(80, 35)
        apply_btn.setStyleSheet("""
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
        """)
        row_layout = QHBoxLayout()
        row_layout.addWidget(input_line)
        row_layout.addSpacing(15)         # 텍스트 입력 창과 버튼 사이 간격 조절
        row_layout.addWidget(apply_btn)
        input_layout.addLayout(row_layout)
        # input_layout.addSpacing(15)         # 텍스트 입력 창과 버튼 사이 간격 조절
        # input_layout.addWidget(apply_btn)
        warning_label = QLabel("\u203b Number of training dataset should be less or equal than number of category.")
        warning_label.setStyleSheet("font-size: 11px; color: gray;")

        input_layout.addSpacing(10)
        input_layout.addWidget(warning_label)
        
        # input_layout.addWidget(warning_label)
        main_layout.addWidget(input_group)
        input_group.setLayout(input_layout)
        # main_layout.addWidget(warning_label)

        

        # Next button
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
        main_layout.addStretch()
        main_layout.addWidget(next_btn, alignment=Qt.AlignRight)

        # 드래그 이동 이벤트
        self.offset = None
        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.offset is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = InputVectorWindow()
    window.show()
    sys.exit(app.exec_())
