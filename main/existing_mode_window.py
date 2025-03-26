import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, \
    QGraphicsDropShadowEffect, QGroupBox, QProgressBar, QLineEdit, QFileDialog, QTextEdit, QSizePolicy
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QPoint


# 2. Train 섹션 (최소 높이)
class TrainSection(QWidget):
    def __init__(self):
        super().__init__()

        train_group = QGroupBox("2. Train")
        train_group.setStyleSheet("""
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
        train_group.setFixedHeight(70)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 3px;
                background-color: #f1f1f1;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        self.percent_label = QLabel("0%")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.percent_label.setFixedWidth(40)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.percent_label)
        train_group.setLayout(progress_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(train_group)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.percent_label.setText(f"{value}%")


# 3. Inference 섹션 (최소 높이)
class InferenceSection(QWidget):
    def __init__(self):
        super().__init__()

        inference_group = QGroupBox("3. Inference")
        inference_group.setStyleSheet("""
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
        inference_group.setFixedHeight(80)

        inference_layout = QHBoxLayout()

        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Put in the file to infer")
        self.file_input.setFixedHeight(35)
        self.file_input.setStyleSheet("""
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

        start_btn = QPushButton("Start")
        start_btn.setFixedSize(60, 35)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton:pressed {
                background-color: #adb5bd;
                color: white;
            }
        """)

        inference_layout.addWidget(self.file_input)
        inference_layout.addWidget(browse_btn)
        inference_layout.addWidget(start_btn)
        inference_group.setLayout(inference_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(inference_group)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            self.file_input.setText(file_path)


# 4. Result 섹션 (가변 크기 + 스크롤 가능)
class ResultSection(QWidget):
    def __init__(self):
        super().__init__()

        result_group = QGroupBox("4. Result")
        result_group.setStyleSheet("""
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

        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setText("Inference result : 7")
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                background-color: #f8f9fa;
            }
        """)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(result_group)


# 메인 창
class SubWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 800)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        container = QWidget(self)
        container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
            }
        """)
        container.setGeometry(0, 0, 800, 800)

        title_bar = QWidget(container)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet(
            "background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pixmap = QPixmap("main\intellino_TM_transparent.png").scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon("main\home.png"))
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

        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 60, 10, 10)

        # 1. Dataset
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
        dataset_layout.setContentsMargins(10, 5, 10, 5)
        dataset_layout.setSpacing(10)

        dataset_layout.addStretch()

        button_style = """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 5px 5px;
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
        mnist_btn = QPushButton("MNIST")
        mnist_btn.setMinimumSize(QSize(90, 35))
        mnist_btn.setStyleSheet(button_style)

        cifar_btn = QPushButton("CIFAR-10")
        cifar_btn.setMinimumSize(QSize(110, 35))
        cifar_btn.setStyleSheet(button_style)

        speech_btn = QPushButton("Speech Commands")
        speech_btn.setMinimumSize(QSize(180, 35))
        speech_btn.setStyleSheet(button_style)

        dataset_layout.addWidget(mnist_btn)
        dataset_layout.addWidget(cifar_btn)
        dataset_layout.addWidget(speech_btn)

        dataset_layout.addStretch()

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # 2. Train (최소)
        self.train_section = TrainSection()
        layout.addWidget(self.train_section)

        # 3. Inference (최소)
        self.inference_section = InferenceSection()
        layout.addWidget(self.inference_section)

        # 4. Result (가변 스크롤)
        self.result_section = ResultSection()
        layout.addWidget(self.result_section)

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
