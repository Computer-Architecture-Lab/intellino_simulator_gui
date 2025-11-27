# utils/ui_common.py
from PySide2.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QStyle
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent
from PySide2.QtCore import Qt, QSize
from utils.resource_utils import resource_path

BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 5px;
        font-weight: bold;
        font-size: 14px;
    }
    QPushButton:hover { background-color: #e9ecef; }
    QPushButton:pressed { background-color: #adb5bd; color: white; }
"""

TOGGLE_BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #C9C9C9;
        border-radius: 10px;
        padding: 6px 14px;
    }
    QPushButton:hover { background-color: #F2F4F6; }
    QPushButton:checked {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #212529;
    }
    QPushButton:pressed { background-color: #E9ECEF; }
"""

class TitleBar(QWidget):
    """
    custom_1, custom_2, custom_3, custom_4 에서 전부 복붙해 쓰던 TitleBar를 하나로 통일.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(50)
        self.setStyleSheet("background-color: #f1f3f5; border-top-left-radius:15px; border-top-right-radius:15px;")
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)

        # Logo
        logo_label = QLabel()
        pm = QPixmap(resource_path("image/intellino_TM_transparent.png"))
        if pm.isNull():
            logo_label.setText("intellino")
            logo_label.setStyleSheet("font-weight:600;")
        else:
            logo_label.setPixmap(pm.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Home btn
        home_btn = QPushButton()
        icon = QIcon(resource_path("image/home.png"))
        if icon.isNull():
            icon = self.style().standardIcon(QStyle.SP_DirHomeIcon)
        home_btn.setIcon(icon)
        home_btn.setIconSize(QSize(24,24))
        home_btn.setFixedSize(34,34)
        home_btn.setStyleSheet("QPushButton { border:none; background:transparent; } QPushButton:hover { background:#dee2e6; border-radius:17px; }")
        home_btn.clicked.connect(self._on_home)

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(home_btn)

        self._offset = None

    def _on_home(self):
        if self._parent:
            self._parent.close()

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._offset = e.pos()

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._offset is not None and e.buttons() == Qt.LeftButton and self._parent:
            self._parent.move(self._parent.pos() + e.pos() - self._offset)

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._offset = None
