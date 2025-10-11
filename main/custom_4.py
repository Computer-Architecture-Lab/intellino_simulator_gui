# custom_4.py
import sys
import os
from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QFrame, QSizePolicy
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = os.path.join(ASSETS_DIR, "intellino_TM_transparent.png")
HOME_ICON_PATH = os.path.join(ASSETS_DIR, "home.png")

# 공통 버튼 스타일
BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 6px 12px;
        font-weight: bold;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #e9ecef; }
    QPushButton:pressed { background-color: #adb5bd; color: white; }
"""

# 타이틀 바
class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(50)
        self.setStyleSheet(
            "background-color: #f1f3f5; "
            "border-top-left-radius: 15px; border-top-right-radius: 15px;"
        )
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 15, 0)

        logo_label = QLabel()
        pix = QPixmap(LOGO_PATH).scaled(
            65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        logo_label.setPixmap(pix)

        home_btn = QPushButton()
        home_btn.setIcon(QIcon(HOME_ICON_PATH))
        home_btn.setIconSize(QSize(24, 24))
        home_btn.setFixedSize(34, 34)
        home_btn.setStyleSheet("""
            QPushButton { border: none; background-color: transparent; }
            QPushButton:hover { background-color: #dee2e6; border-radius: 17px; }
        """)
        # ✔ 올바른 메서드로 연결(중첩 정의/람다 중복 연결 제거)
        home_btn.clicked.connect(self._on_home_clicked)

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(home_btn)

        self._offset = None

    def _on_home_clicked(self):
        app = QApplication.instance()
        if app:
            app.setStyleSheet("")
        if self._parent:
            self._parent.close()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._offset is not None and event.buttons() == Qt.LeftButton and self._parent:
            self._parent.move(self._parent.pos() + event.pos() - self._offset)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._offset = None

# 10. Experiment graph 섹션
class ExperimentGraphSection(QWidget):
    def __init__(self):
        super().__init__()

        group = QGroupBox("9. Experiment graph")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 14px;
                border: 1px solid #a9a9a9;
                border-radius: 12px;
                margin-top: 10px; padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
        """)

        v = QVBoxLayout()
        v.setContentsMargins(10, 10, 10, 10)

        # 내부 캔버스(빈 그래프 영역)
        self.canvas = QFrame()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(450)
        self.canvas.setStyleSheet("""
            QFrame {
                background-color: #f5f6f7;
                border: 1px solid #d1d1d1;
                border-radius: 18px;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self.canvas)
        shadow.setBlurRadius(18)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 0)
        self.canvas.setGraphicsEffect(shadow)

        v.addWidget(self.canvas)
        group.setLayout(v)

        main = QVBoxLayout(self)
        main.addWidget(group)

# 메인 창
class ExperimentWindow(QWidget):
    def __init__(self, num_categories: int = 0):
        super().__init__()
        self.num_categories = num_categories
        self._custom1_window = None  # Reconfigure로 열리는 창 참조 저장
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 800)

        container = QWidget(self)
        container.setStyleSheet("background-color: white; border-radius: 15px;")
        container.setGeometry(0, 0, 800, 800)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        self.title_bar = TitleBar(self)
        self.title_bar.setParent(container)
        self.title_bar.setGeometry(0, 0, 800, 50)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)
        layout.setSpacing(20)

        # 10. Experiment graph
        self.graph_section = ExperimentGraphSection()
        # 그래프 영역이 남는 공간을 모두 차지하도록 확장
        self.graph_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.graph_section)

        # ▼ 남는 공간을 위의 그래프가 흡수하도록, 버튼 영역을 맨 아래로
        layout.addStretch(1)

        # 하단 버튼 영역 (오른쪽 정렬, 세로 배치)
        btn_col = QVBoxLayout()
        btn_col.setSpacing(12)

        self.reconf_btn = QPushButton("Reconfigure")
        self.reconf_btn.setFixedSize(120, 40)
        self.reconf_btn.setStyleSheet(BUTTON_STYLE)
        # ▼ Reconfigure 클릭 시 custom_1.py로 이동
        self.reconf_btn.clicked.connect(self._open_reconfigure)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.setFixedSize(120, 40)
        self.finish_btn.setStyleSheet(BUTTON_STYLE)
        self.finish_btn.clicked.connect(self.close)

        btn_col.addWidget(self.reconf_btn)
        btn_col.addWidget(self.finish_btn)

        # 버튼 레이아웃을 고정 크기 컨테이너에 담아 우하단 정렬
        btn_container = QWidget()
        btn_container.setLayout(btn_col)
        btn_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(btn_container, 0, Qt.AlignRight | Qt.AlignBottom)

    def _open_reconfigure(self):
        from custom_1 import Custom_1_Window, GLOBAL_FONT_QSS

        # Reconfigure 창만 스타일 적용 (앱 전역 X)
        self._custom1_window = Custom_1_Window(prev_window=self)
        try:
            self._custom1_window.setStyleSheet(GLOBAL_FONT_QSS)
        except Exception:
            pass

        self._custom1_window.show()
        self.hide()

# 실행 테스트용
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ExperimentWindow()
    w.show()
    sys.exit(app.exec_())
