# custom_2.py
import sys, os, subprocess
from functools import partial

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = os.path.join(ASSETS_DIR, "intellino_TM_transparent.png")
HOME_ICON_PATH = os.path.join(ASSETS_DIR, "home.png")
MESSAGE_BOX_QSS = """
* {
    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-weight: 500;
    font-size: 13px;
}
"""

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog, QScrollArea,
    QMessageBox, QSizePolicy, QGraphicsOpacityEffect, QLabel
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor
from PySide2.QtCore import Qt, QSize, Signal, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from custom_3 import SubWindow as Window3
from path_utils import get_dirs

# ---------------------------------------------------------------------
# custom_1의 QSS를 그대로 쓰기 위한 Fallback 상수(순환참조 시 사용)
GLOBAL_FONT_QSS_FALLBACK = """
* {
    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-weight: 500;
    font-size: 13px;
}
"""
GROUPBOX_WITH_FLOATING_TITLE_FALLBACK = """
    QGroupBox {
        border: 1px solid #b0b0b0;
        border-radius: 10px;
        margin-top: 10px;
        padding: 10px;
        background: transparent;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        background-color: white;
        font-weight: 600;
        font-size: 13px;
        color: #000000;
    }
"""
# ---------------------------------------------------------------------

# 프로젝트 상대 경로 확보
CUSTOM_IMAGE_ROOT, BASE_NUMBER_DIR, KMEANS_SELECTED_DIR = get_dirs(__file__)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

BUTTON_STYLE = """
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 5px;
        font-weight: bold;
        font-size: 13px;
    }
    QPushButton:hover { background-color: #e9ecef; }
    QPushButton:pressed { background-color: #adb5bd; color: white; }
"""

class TrainDatasetGroup(QGroupBox):
    completeness_changed = Signal(bool)

    def __init__(self, num_categories=3, base_dir=BASE_NUMBER_DIR, required_per_class=1):
        super().__init__("5. Training datasets of each category")

        # ▶ custom_1의 GROUPBOX_WITH_FLOATING_TITLE을 그대로 적용
        try:
            from custom_1 import GROUPBOX_WITH_FLOATING_TITLE as GB_STYLE
        except Exception:
            GB_STYLE = GROUPBOX_WITH_FLOATING_TITLE_FALLBACK
        
        self.setObjectName("TrainDSGroup")
        self.setStyleSheet(GB_STYLE + """
            QGroupBox#TrainDSGroup::title {
                font-weight: 700;   /* 굵게 */
                font-weight: bold;  /* QSS 호환용 */
            }
        """)

        self.base_dir = base_dir
        self.required_per_class = max(1, int(required_per_class))

        self.category_inputs = []   # (dir_input, label_input)
        self.row_widgets = []       # {"dir_input":..,"label_input":..,"count_label":..}
        self._last_complete = None

        self._build_ui(num_categories)
        self._on_fields_changed()   # 초기 상태 반영

    def _build_ui(self, num_categories):
        outer_layout = QVBoxLayout(self)

        # ── 한 줄 높이/간격(요청 반영: 살짝 키움) ──
        ROW_EDIT_H = 38

        if num_categories <= 5:
            ROW_SPACING = 24
        elif num_categories <= 7:
            ROW_SPACING = 22
        elif num_categories <= 10:
            ROW_SPACING = 14
        else:
            ROW_SPACING = 10

        body = QWidget()
        v = QVBoxLayout(body)
        v.setSpacing(ROW_SPACING)
        v.setContentsMargins(8, 8, 8, 8)

        add_top_bottom_stretch = num_categories <= 5
        if add_top_bottom_stretch:
            v.addStretch(1)   # top stretch

        for i in range(1, num_categories + 1):
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"Category {i}")
            label.setFixedWidth(80)
            label.setStyleSheet("font-size: 13px;")

            dir_input = QLineEdit()
            dir_input.setPlaceholderText("Select folder")
            dir_input.setFixedHeight(ROW_EDIT_H)
            dir_input.setStyleSheet("""
                QLineEdit { border: 1px solid #ccc; border-radius: 8px; padding-left: 10px; font-size: 13px; }
            """)
            dir_input.textChanged.connect(self._on_fields_changed)

            browse_btn = QPushButton("...")
            browse_btn.setFixedSize(ROW_EDIT_H, ROW_EDIT_H)
            browse_btn.setStyleSheet("""
                QPushButton { border: 1px solid #ccc; border-radius: 8px; background-color: #ffffff; font-weight: bold; }
                QPushButton:hover { background-color: #e9ecef; }
            """)
            browse_btn.clicked.connect(partial(self.browse_number_folder, dir_input))

            count_badge = QLabel(f"0/{self.required_per_class}")
            count_badge.setFixedWidth(70)
            count_badge.setFixedHeight(ROW_EDIT_H - 4)  # 34px 정도
            count_badge.setAlignment(Qt.AlignCenter)
            count_badge.setStyleSheet(self._badge_style(0))

            label_input = QLineEdit()
            label_input.setPlaceholderText("Enter label")
            label_input.setFixedHeight(ROW_EDIT_H)
            label_input.setFixedWidth(110)
            label_input.setStyleSheet("""
                QLineEdit { border: 1px solid #ccc; border-radius: 8px; padding-left: 10px; font-size: 13px; }
            """)
            label_input.textChanged.connect(self._on_fields_changed)

            h.addWidget(label)
            h.addWidget(dir_input)
            h.addWidget(browse_btn)
            h.addWidget(count_badge)
            h.addWidget(label_input)

            self.category_inputs.append((dir_input, label_input))
            self.row_widgets.append({"dir_input": dir_input, "label_input": label_input, "count_label": count_badge})
            v.addLayout(h)

            if num_categories <= 4 and i < num_categories:
                v.addStretch(1)

        if add_top_bottom_stretch:
            v.addStretch(1)   # bottom stretch

        body.setLayout(v)

        # ≤10 : 스크롤 없이 전부 보이기
        if num_categories <= 10:
            outer_layout.addWidget(body)
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(body)

            row_h = max(ROW_EDIT_H + ROW_SPACING, body.sizeHint().height() // max(1, num_categories))
            view_h = row_h * 10 + 24
            scroll.setFixedHeight(view_h)

            scroll.setStyleSheet("""
                QScrollBar:vertical { border:none; background:#f1f3f5; width:10px; margin:5px 0; border-radius:5px; }
                QScrollBar::handle:vertical { background:#adb5bd; min-height:25px; border-radius:5px; }
                QScrollBar::handle:vertical:hover { background:#868e96; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
            """)
            outer_layout.addWidget(scroll)

    def _badge_style(self, n):
        if n >= self.required_per_class:
            return "QLabel { background:#e6fcf5; border:1px solid #37b24d; color:#2b8a3e; border-radius:6px; padding:4px; }"
        elif n > 0:
            return "QLabel { background:#fff4e6; border:1px solid #f08c00; color:#d9480f; border-radius:6px; padding:4px; }"
        else:
            return "QLabel { background:#ffe3e3; border:1px solid #fa5252; color:#c92a2a; border-radius:6px; padding:4px; }"

    def _update_row_status(self, di, cl):
        path = di.text().strip()
        n = 0
        if os.path.isdir(path):
            try:
                n = sum(1 for f in os.listdir(path) if str(f).lower().endswith(IMG_EXTS))
            except Exception:
                n = 0
        cl.setText(f"{n}/{self.required_per_class}")
        cl.setStyleSheet(self._badge_style(n))

    def browse_number_folder(self, dir_input):
        path = QFileDialog.getExistingDirectory(
            self, "Select number folder (0~9)", self.base_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not path:
            return
        base = os.path.basename(os.path.normpath(path))
        parent = os.path.dirname(os.path.normpath(path))

        if base.isdigit() and len(base) == 1 and os.path.normpath(parent).lower() == os.path.normpath(self.base_dir).lower():
            dir_input.setText(path)
            try:
                n = sum(1 for f in os.listdir(path) if str(f).lower().endswith(IMG_EXTS))
            except Exception:
                n = 0
            if n < self.required_per_class:
                box = QMessageBox(self)
                box.setWindowTitle("Not enough images")
                box.setIcon(QMessageBox.Warning)
                box.setText(
                    f"This folder has {n} images (required {self.required_per_class}).\n"
                    "Please add more images or select another folder."
                )
                box.setStyleSheet(MESSAGE_BOX_QSS)
                box.exec_()
        else:
            QMessageBox.warning(
                self,
                "Invalid folder",
                f"You can only select folders named 0–9 under 'main/custom_image/number_image'.\n\n"
                f"Current selection: {path}\n"
                f"Expected base path: {self.base_dir}"
            )

    def is_complete(self) -> bool:
        for row in self.row_widgets:
            d = row["dir_input"].text().strip()
            l = row["label_input"].text().strip()
            if not d or not l:
                return False
            if not os.path.isdir(d):
                return False
            base = os.path.basename(os.path.normpath(d))
            if not (base.isdigit() and len(base) == 1):
                return False
            if os.path.normpath(os.path.dirname(d)).lower() != os.path.normpath(self.base_dir).lower():
                return False
            try:
                n = sum(1 for f in os.listdir(d) if str(f).lower().endswith(IMG_EXTS))
            except Exception:
                return False
            if n < self.required_per_class:
                return False
        return True

    def _on_fields_changed(self, *_):
        for row in self.row_widgets:
            self._update_row_status(row["dir_input"], row["count_label"])
        self._emit_completeness()

    def _emit_completeness(self):
        ok = self.is_complete()
        if ok != self._last_complete:
            self._last_complete = ok
            self.completeness_changed.emit(ok)

    def get_selection(self):
        items = []
        for idx, row in enumerate(self.row_widgets, start=1):
            items.append({
                "category_index": idx,
                "dir": row["dir_input"].text().strip(),
                "label": row["label_input"].text().strip()
            })
        return items


class Custom_2_Window(QWidget):
    def __init__(self, num_categories=3, samples_per_class=1, prev_window=None):
        super().__init__()
        self.num_categories = num_categories
        self.samples_per_class = max(1, int(samples_per_class))
        self.win3 = None
        self.prev_window = prev_window
        self._orig_app_qss = ""   # 전역 스타일 복원용
        self._setup_ui()

    # custom_1과 동일한 전역 폰트 QSS를 QApplication에 적용/복원
    def _apply_global_font(self):
        app = QApplication.instance()
        if not app:
            return
        self._orig_app_qss = app.styleSheet() or ""
        try:
            from custom_1 import GLOBAL_FONT_QSS   # ← 순환참조 없이 런타임 import
        except Exception:
            GLOBAL_FONT_QSS = GLOBAL_FONT_QSS_FALLBACK
        app.setStyleSheet(self._orig_app_qss + ("\n" + GLOBAL_FONT_QSS if GLOBAL_FONT_QSS else ""))

    def _restore_global_font(self):
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self._orig_app_qss)

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # ★ 전역 폰트 QSS를 먼저 적용 — QGroupBox::title까지 동일한 글씨체로
        self._apply_global_font()

        self.setFixedSize(800, 800)

        container = QWidget(self)
        container.setStyleSheet("background-color: white; border-radius: 15px;")
        container.setGeometry(0, 0, 800, 800)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30); shadow.setOffset(0, 0); shadow.setColor(QColor(0,0,0,120))
        container.setGraphicsEffect(shadow)

        self._add_title_bar(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)
        layout.setSpacing(24)

        self.dataset_group = TrainDatasetGroup(
            num_categories=self.num_categories,
            base_dir=BASE_NUMBER_DIR,
            required_per_class=self.samples_per_class
        )
        layout.addWidget(self.dataset_group)
        self.dataset_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addStretch()
        layout.addLayout(self._create_nav_buttons())

        self.next_btn.setEnabled(False)
        self.dataset_group.completeness_changed.connect(self.next_btn.setEnabled)
        self.next_btn.setStyleSheet(self.next_btn.styleSheet() + """
            QPushButton:disabled { background-color:#f1f3f5; color:#adb5bd; border:1px solid #ddd; }
        """)
        self.next_btn.setEnabled(self.dataset_group.is_complete())

    def _add_title_bar(self, parent):
        bar = QWidget(parent); bar.setGeometry(0,0,800,50)
        bar.setStyleSheet("background-color:#f1f3f5; border-top-left-radius:15px; border-top-right-radius:15px;")
        h = QHBoxLayout(bar); h.setContentsMargins(15,0,15,0)

        logo = QLabel()
        logo.setPixmap(QPixmap(LOGO_PATH).scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        close_btn = QPushButton()
        close_btn.setIcon(QIcon(HOME_ICON_PATH))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34,34)
        close_btn.setStyleSheet("QPushButton{border:none;background:transparent;} QPushButton:hover{background:#dee2e6;border-radius:17px;}")
        close_btn.clicked.connect(self.close)
        h.addWidget(logo); h.addStretch(); h.addWidget(close_btn)

        bar.mousePressEvent = self.mousePressEvent
        bar.mouseMoveEvent  = self.mouseMoveEvent

        def _go_home():
            self._restore_global_font()   # 전역 스타일 복원
            self.close()

        close_btn.clicked.connect(_go_home)

    def _create_nav_buttons(self):
        btn_style = """
            QPushButton { font-weight:bold; font-size:14px; border:1px solid #888; border-radius:8px; background-color:#fefefe; }
            QPushButton:hover { background-color:#dee2e6; }
        """
        self.back_btn = QPushButton("Back"); self.back_btn.setFixedSize(100,40); self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(self.go_back)

        self.next_btn = QPushButton("Train Start"); self.next_btn.setFixedSize(100,40); self.next_btn.setStyleSheet(btn_style)
        self.next_btn.clicked.connect(self.start_kmeans)

        row = QHBoxLayout(); row.addWidget(self.back_btn); row.addStretch(); row.addWidget(self.next_btn)
        return row

    def start_kmeans(self):
        if not self.dataset_group.is_complete():
            return

        sel = self.dataset_group.get_selection()

        if getattr(self, "win3", None) is None:
            self.win3 = Window3(
                selection=sel,
                samples_per_class=self.samples_per_class,
                prev_window=self.prev_window
            )

        # 다음 창 위치/크기 맞추기
        try:
            self.win3.setGeometry(self.geometry())
        except Exception:
            self.win3.move(self.pos())

        # 1) 다음 창 먼저 정상 표시
        self.win3.show()
        self.win3.raise_()
        self.win3.activateWindow()

        # 2) 현재 창 화면을 스냅샷(픽스맵)으로 캡처
        snap = self.grab()  # QWidget.grab()은 알파(모서리 라운드 포함)를 보존

        # 3) 다음 창 위에 오버레이 라벨을 하나 얹고, 캡처 이미지를 붙임
        overlay = QLabel(self.win3)
        overlay.setPixmap(snap)
        overlay.setGeometry(0, 0, self.width(), self.height())
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.raise_()
        overlay.show()

        # 4) 오버레이만 투명하게 사라지는 효과
        effect = QGraphicsOpacityEffect(overlay)
        overlay.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(180)  # 160~220ms 권장
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad)

        # 5) 끝나면 오버레이 제거 + 이전 창 숨김(종료 아님)
        def _done():
            overlay.deleteLater()
            # 이전 창은 보이지 않게만(앱 종료/비는 구간 방지)
            self.hide()

        # 애니메이션 가비지컬렉션 방지
        self._overlay_anim = anim
        anim.finished.connect(_done)
        anim.start()


        def _end():
            # hide로 전환(앱 종료/깜빡임 방지), 다음에 다시 열릴 수 있도록 원복
            self.hide()
            try:
                self.setWindowOpacity(1.0)
            except Exception:
                pass

        self._anim_group.finished.connect(_end)
        self._anim_group.start()

    def go_back(self):
        if self.prev_window is not None:
            try:
                self.prev_window.show(); self.prev_window.raise_(); self.prev_window.activateWindow()
            except Exception:
                pass
        self.close()

    def closeEvent(self, e):
        self._restore_global_font()  # 다른 방식으로 닫혀도 복원
        super().closeEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self.offset = e.pos()
    def mouseMoveEvent(self, e):
        if hasattr(self, 'offset') and e.buttons() == Qt.LeftButton:
            self.move(self.pos() + e.pos() - self.offset)

def launch_training_window(num_categories, samples_per_class, prev_window=None):
    window = Custom_2_Window(num_categories=num_categories, samples_per_class=samples_per_class, prev_window=prev_window)
    window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Custom_2_Window()
    window.show()
    sys.exit(app.exec_())