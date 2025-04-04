import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog, QScrollArea
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QPoint
from functools import partial  # TypeError 방지. 안정적인 파일 경로 전달


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
    def __init__(self, num_categories=3):
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
        self.category_inputs = []
        self._build_ui(num_categories)

    def _build_ui(self, num_categories):
        outer_layout = QVBoxLayout(self)

        # 스크롤바 추가
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        if num_categories >= 10:
            scroll_area.setFixedHeight(400)  # 필요 시 이 높이는 조절 가능

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)

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
            browse_btn.clicked.connect(partial(self.browse_file, file_input))

            label_input = QLineEdit()
            label_input.setPlaceholderText("Enter label")
            label_input.setFixedHeight(35)
            label_input.setFixedWidth(100)
            label_input.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    padding-left: 10px;
                    font-size: 13px;
                }
            """)

            h_layout.addWidget(label)
            h_layout.addWidget(file_input)
            h_layout.addWidget(browse_btn)
            h_layout.addWidget(label_input)

            self.category_inputs.append((file_input, label_input))
            scroll_layout.addLayout(h_layout)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        # 💄 스크롤바 스타일 적용
        scroll_area.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: #f1f3f5;
                width: 10px;
                margin: 5px 0 5px 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #adb5bd;
                min-height: 25px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #868e96;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        outer_layout.addWidget(scroll_area)

    def browse_file(self, file_input):
        print("📂 browse_file 호출됨")  # 디버깅용
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            file_input.setText(file_path)

class Custom_2_Window(QWidget):
    def __init__(self, num_categories=3):
        super().__init__()
        self.num_categories=num_categories
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.setFixedSize(800, 800)

        container = QWidget(self)
        container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
            }
        """)
        container.setGeometry(0, 0, 800, 800)

        # ✅ 그림자 효과는 container에만 적용
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 0, 0, 120))
        container.setGraphicsEffect(shadow)


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
    window = Custom_2_Window(num_categories)
    window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Custom_2_Window()
    window.show()
    sys.exit(app.exec_())
