import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
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

class TrainDatasetGroup(QGroupBox):
    def __init__(self, num_categories=3):  # 기본적으로 3개 생성
        super().__init__("6. Training datasets of each category")
        self.setStyleSheet("""
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

        self.category_inputs = []  # 각 입력 필드 저장용
        self.setLayout(self._build_ui(num_categories))

    def _build_ui(self, num_categories):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        for i in range(1, num_categories + 1):
            h_layout = QHBoxLayout()

            label = QLabel(f"Category {i}")
            label.setFixedWidth(80)
            label.setStyleSheet("font-size: 13px;")

            file_input = QLineEdit()
            file_input.setPlaceholderText("Put the train datasets of this category.")
            file_input.setFixedHeight(35)
            file_input.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    padding-left: 10px;
                    font-size: 13px;
                }
            """)

            browse_btn = QPushButton("...")
            browse_btn.setFixedSize(35, 35)
            browse_btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    background-color: #ffffff;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
            """)
            browse_btn.clicked.connect(self.browse_file)

            h_layout.addWidget(label)
            h_layout.addWidget(file_input)
            h_layout.addWidget(browse_btn)

            self.category_inputs.append(file_input)
            layout.addLayout(h_layout)

        return layout

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            self.file_input.setText(file_path)


class SubWindow(QWidget):
    def __init__(self, num_categories):
        super().__init__()
        self.num_categories=num_categories
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

        layout.addWidget(TrainDatasetGroup(num_categories=self.num_categories))
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
        close_btn.setStyleSheet("border: none; background-color: transparent;")
        close_btn.clicked.connect(self.close)

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(close_btn)

        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

    def _create_next_button(self):
        next_btn = QPushButton("Train Start")
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

def launch_training_window(num_categories):
    app = QApplication(sys.argv)
    window = SubWindow(num_categories)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SubWindow()
    window.show()
    sys.exit(app.exec_())
