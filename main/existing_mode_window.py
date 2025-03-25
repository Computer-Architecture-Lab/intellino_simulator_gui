import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, \
    QGraphicsDropShadowEffect, QGroupBox
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QPoint


class SubWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 600)

        # 바깥쪽 그림자 효과
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        # 메인 컨테이너
        container = QWidget(self)
        container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
            }
        """)
        container.setGeometry(0, 0, 800, 600)

        # ✅ 타이틀바를 layout에 넣지 않고 직접 geometry로 배치
        title_bar = QWidget(container)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet(
            "background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pixmap = QPixmap("intellino_TM_transparent.png").scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon("home.png"))
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

        # 이제 레이아웃은 타이틀바 아래부터 시작하도록 마진 조절
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 60, 10, 10)  # top margin 60으로 조절

        # Dataset 그룹박스
        dataset_group = QGroupBox("1. Dataset")
        dataset_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #b0b0b0;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)

        dataset_layout = QHBoxLayout()

        button_style = """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton:pressed {
                background-color: #adb5bd;
                color: white;
            }
        """

        mnist_btn = QPushButton("MNIST")
        mnist_btn.setFixedSize(QSize(120, 50))
        mnist_btn.setStyleSheet(button_style)

        cifar_btn = QPushButton("CIFAR-10")
        cifar_btn.setFixedSize(QSize(120, 50))
        cifar_btn.setStyleSheet(button_style)

        speech_btn = QPushButton("Speech Commands")
        speech_btn.setFixedSize(QSize(190, 50))
        speech_btn.setStyleSheet(button_style)

        dataset_layout.addWidget(mnist_btn)
        dataset_layout.addWidget(cifar_btn)
        dataset_layout.addWidget(speech_btn)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # 드래그 기능
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
    app = QApplication(sys.argv)
    subwin = SubWindow()
    subwin.show()
    sys.exit(app.exec_())
