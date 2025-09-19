import sys, os
from functools import partial

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog, QScrollArea
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent
from PySide2.QtCore import Qt, QSize, QPoint, Signal  # ★ Signal 사용

# ✅ window_3(custom_3.py)에서 SubWindow를 import
from custom_3 import SubWindow as Window3


# 공통 버튼 스타일(필요 시 재사용)
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
    # ★ 모든 카테고리의 (파일, 라벨) 입력이 유효한지 여부를 내보내는 신호
    completeness_changed = Signal(bool)

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
        self.category_inputs = []  # [(file_input, label_input), ...]
        self._last_complete = None
        self._build_ui(num_categories)
        # 초기 상태 한 번 평가
        self._emit_completeness()

    def _build_ui(self, num_categories):
        outer_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        if num_categories >= 10:
            scroll_area.setFixedHeight(400)

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
            # ★ 변경 감지
            file_input.textChanged.connect(self._on_fields_changed)

            browse_btn = QPushButton("...")
            browse_btn.setFixedSize(35, 35)
            browse_btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    background-color: #ffffff;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #e9ecef; }
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
            # ★ 변경 감지
            label_input.textChanged.connect(self._on_fields_changed)

            h_layout.addWidget(label)
            h_layout.addWidget(file_input)
            h_layout.addWidget(browse_btn)
            h_layout.addWidget(label_input)

            self.category_inputs.append((file_input, label_input))
            scroll_layout.addLayout(h_layout)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        # 스크롤바 스타일
        scroll_area.setStyleSheet("""
            QScrollBar:vertical { border: none; background: #f1f3f5; width: 10px; margin: 5px 0; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #adb5bd; min-height: 25px; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #868e96; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

        outer_layout.addWidget(scroll_area)

    def browse_file(self, file_input):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if file_path:
            file_input.setText(file_path)  # textChanged 시그널이 자동으로 발동 → 검증 갱신

    # --------- 검증 로직 ---------
    def is_complete(self) -> bool:
        """모든 카테고리에서 파일 경로와 라벨이 채워졌는지 + 파일 존재 여부를 확인."""
        for file_input, label_input in self.category_inputs:
            f = file_input.text().strip()
            l = label_input.text().strip()
            if not f or not l:
                return False
            if not os.path.exists(f):
                return False
        return True

    def _on_fields_changed(self, *_):
        self._emit_completeness()

    def _emit_completeness(self):
        complete = self.is_complete()
        # 이전 값과 다를 때만 신호 발행(쓸데없는 중복 방지)
        if complete != self._last_complete:
            self._last_complete = complete
            self.completeness_changed.emit(complete)


class Custom_2_Window(QWidget):
    def __init__(self, num_categories=3, prev_window=None):
        super().__init__()
        self.num_categories = num_categories
        self.win3 = None
        self.prev_window = prev_window  # ← Back 동작에 사용할 이전 창 인스턴스
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 800)

        container = QWidget(self)
        container.setStyleSheet("background-color: white; border-radius: 15px;")
        container.setGeometry(0, 0, 800, 800)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(0, 0, 0, 120))
        container.setGraphicsEffect(shadow)

        self._add_title_bar(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)
        layout.setSpacing(50)

        # ★ 데이터셋 그룹 보관(검증 신호를 받기 위해)
        self.dataset_group = TrainDatasetGroup(num_categories=self.num_categories)
        layout.addWidget(self.dataset_group)

        layout.addStretch()
        layout.addLayout(self._create_nav_buttons())  # ← Back | Train Start

        # ★ 초기 비활성화 + 상태 동기화 연결 (Train Start)
        self.next_btn.setEnabled(False)
        self.dataset_group.completeness_changed.connect(self.next_btn.setEnabled)
        # 스타일에서 비활성화 시각화
        self.next_btn.setStyleSheet(self.next_btn.styleSheet() + """
            QPushButton:disabled {
                background-color: #f1f3f5;
                color: #adb5bd;
                border: 1px solid #ddd;
            }
        """)
        # 한 번 초기 평가
        self.next_btn.setEnabled(self.dataset_group.is_complete())

    def _add_title_bar(self, parent):
        title_bar = QWidget(parent)
        title_bar.setGeometry(0, 0, 800, 50)
        title_bar.setStyleSheet("background-color: #f1f3f5; border-top-left-radius: 15px; border-top-right-radius: 15px;")

        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(15, 0, 15, 0)

        # 경로 안전화(원하면 절대경로로 대체 가능)
        logo_label = QLabel()
        pixmap = QPixmap("main/intellino_TM_transparent.png").scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)

        close_btn = QPushButton()
        close_btn.setIcon(QIcon("main/home.png"))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34, 34)
        close_btn.setStyleSheet("""
            QPushButton { border: none; background-color: transparent; }
            QPushButton:hover { background-color: #dee2e6; border-radius: 17px; }
        """)
        close_btn.clicked.connect(self.close)

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(close_btn)

        # 드래그 이동
        title_bar.mousePressEvent = self.mousePressEvent
        title_bar.mouseMoveEvent = self.mouseMoveEvent

    def _create_nav_buttons(self):
        # 공통 스타일
        btn_style = """
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #888;
                border-radius: 8px;
                background-color: #fefefe;
            }
            QPushButton:hover { background-color: #dee2e6; }
        """
        # Back 버튼 (왼쪽 아래)
        self.back_btn = QPushButton("Back")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(self.go_back)

        # Train Start 버튼 (오른쪽 아래)
        self.next_btn = QPushButton("Train Start")
        self.next_btn.setFixedSize(100, 40)
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.clicked.connect(self.open_window3)

        # 하단 배치: 왼쪽 Back, 오른쪽 Start
        layout = QHBoxLayout()
        layout.addWidget(self.back_btn)
        layout.addStretch()
        layout.addWidget(self.next_btn)
        return layout

    def open_window3(self):
        # ★ 안전 가드(버튼이 활성화되어 있어도 최종 확인)
        if not self.dataset_group.is_complete():
            return
        if self.win3 is None:
            self.win3 = Window3(num_categories=self.num_categories)
        self.win3.show()
        self.close()

    def go_back(self):
        """
        이전 단계로 돌아가기.
        - prev_window가 주어지면 그 창을 다시 보여주고 포커스/최상위로 올림.
        - 없으면 단순히 현재 창만 닫음.
        """
        if self.prev_window is not None:
            try:
                # (custom_1에서 페이드 효과를 제거/숨김 처리했으므로 보통 불필요하지만)
                # 안전하게 복귀 시 포커스를 보장
                self.prev_window.show()
                self.prev_window.raise_()
                self.prev_window.activateWindow()
            except Exception:
                pass
        self.close()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if hasattr(self, 'offset') and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)


def launch_training_window(num_categories, prev_window=None):
    window = Custom_2_Window(num_categories=num_categories, prev_window=prev_window)
    window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 단독 실행 시 이전 창이 없으므로 Back은 현재 창만 닫습니다.
    window = Custom_2_Window()
    window.show()
    sys.exit(app.exec_())