import sys, os, subprocess
from functools import partial

def resource_path(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # 1) PyInstaller onefile 실행
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
        candidates += [
            os.path.join(base, name),          # ;.
            os.path.join(base, "main", name),  # ;main
        ]

    # 2) 개발 환경
    candidates += [
        os.path.join(here, name),
        os.path.join(here, "main", name),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]  # fallback

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = resource_path("intellino_TM_transparent.png")
HOME_ICON_PATH = resource_path("home.png")

MESSAGE_BOX_QSS = """
QMessageBox {
    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-weight: 500;
    font-size: 14px;
}
"""

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog, QScrollArea,
    QMessageBox, QSizePolicy, QGraphicsOpacityEffect
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor
from PySide2.QtCore import Qt, QSize, Signal, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from custom_3 import SubWindow as Window3
from path_utils import get_dirs

# ---------------------------------------------------------------------
#GLOBAL_FONT_QSS_FALLBACK = """
#* {
#    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
#                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
#    font-weight: 500;
#    font-size: 14px;
#}
#"""
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
        font-weight: 500;
        font-size: 14px;
        color: #000000;
    }
"""
# ---------------------------------------------------------------------

CUSTOM_IMAGE_ROOT, BASE_NUMBER_DIR, KMEANS_SELECTED_DIR = get_dirs(__file__)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

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


# 경로 정규화/비교 유틸 ----------------------------------------------------
def _canon_path(p: str) -> str:
    """realpath + abspath + 대소문자까지 통일해서 비교용 문자열 생성"""
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(p)))
    except Exception:
        return os.path.normcase(os.path.abspath(p))


def _same_dir(a: str, b: str) -> bool:
    """가능하면 samefile, 아니면 정규화 문자열로 비교"""
    try:
        return os.path.samefile(a, b)
    except Exception:
        return _canon_path(a) == _canon_path(b)
# ---------------------------------------------------------------------
class TrainDatasetGroup(QGroupBox):
    completeness_changed = Signal(bool)

    def __init__(self, num_categories=3, base_dir=BASE_NUMBER_DIR, required_per_class=1):
        super().__init__("6. Training datasets of each category")

        try:
            from custom_1 import GROUPBOX_WITH_FLOATING_TITLE as GB_STYLE
        except Exception:
            GB_STYLE = GROUPBOX_WITH_FLOATING_TITLE_FALLBACK

        self.setObjectName("TrainDSGroup")
        self.setStyleSheet(GB_STYLE + """
            QGroupBox#TrainDSGroup::title {
                font-weight: 700;
                font-weight: bold;
            }
        """)

        # 기준 경로(정규화)
        self.base_dir = os.path.abspath(base_dir)
        self._base_dir_canon = _canon_path(self.base_dir)

        self.required_per_class = max(1, int(required_per_class))

        self.category_inputs = []   # (dir_input, label_input)
        self.row_widgets = []       # {"dir_input":..,"label_input":..,"count_label":..}
        self._last_complete = None

        self._build_ui(num_categories)
        self._on_fields_changed()

    def _build_ui(self, num_categories):
        outer_layout = QVBoxLayout(self)

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
            v.addStretch(1)

        for i in range(1, num_categories + 1):
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"Category {i}")
            label.setFixedWidth(80)
            #label.setStyleSheet("font-size: 13px;")

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
            count_badge.setFixedHeight(ROW_EDIT_H - 4)
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
            v.addStretch(1)

        body.setLayout(v)

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
        # DontResolveSymlinks 제거: 경로 표현을 최대한 단일화
        path = QFileDialog.getExistingDirectory(
            self,
            "Select number folder (0~9)",
            self.base_dir,
            QFileDialog.ShowDirsOnly
        )
        if not path:
            return

        base = os.path.basename(os.path.normpath(path))
        parent = os.path.dirname(os.path.normpath(path))

        base_ok = base.isdigit() and len(base) == 1
        parent_ok = _same_dir(parent, self.base_dir)

        if base_ok and parent_ok:
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
            parent = os.path.dirname(os.path.normpath(d))

            if not (base.isdigit() and len(base) == 1):
                return False
            if not _same_dir(parent, self.base_dir):
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
    def __init__(self, num_categories=3, samples_per_class=1, input_vector_length=0, selected_mem_kb=None, prev_window=None):
        super().__init__()
        self.num_categories = num_categories
        self.samples_per_class = max(1, int(samples_per_class))
        self.input_vector_length = int(input_vector_length or 0)
        self.selected_mem_kb = selected_mem_kb
        self.win3 = None
        self.prev_window = prev_window
        self._orig_app_qss = ""
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)

        #self._apply_global_font()
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
        pm = QPixmap(LOGO_PATH)
        if pm.isNull():
            logo.setText("intellino")
            logo.setStyleSheet("font-weight:600; font-size:18px; color:#333;")
        else:
            logo.setPixmap(pm.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        close_btn = QPushButton()
        close_btn.setIcon(QIcon(HOME_ICON_PATH))
        close_btn.setIconSize(QSize(24, 24))
        close_btn.setFixedSize(34,34)
        close_btn.setStyleSheet("QPushButton{border:none;background:transparent;} QPushButton:hover{background:#dee2e6;border-radius:17px;}")
        #close_btn.clicked.connect(self.close)
        h.addWidget(logo); h.addStretch(); h.addWidget(close_btn)

        bar.mousePressEvent = self.mousePressEvent
        bar.mouseMoveEvent  = self.mouseMoveEvent

        def _go_home():
            if self.prev_window:
                self.prev_window.show()
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
                prev_window=self.prev_window,
                exp_params={
                    "num_classes": self.num_categories,
                    "samples_per_class": self.samples_per_class,
                    "input_vec_len": self.input_vector_length,
                    "memory_kb": self.selected_mem_kb
                }
            )

        try:
            self.win3.setGeometry(self.geometry())
        except Exception:
            self.win3.move(self.pos())

        self.win3.show()
        self.win3.raise_()
        self.win3.activateWindow()

        snap = self.grab()
        overlay = QLabel(self.win3)
        overlay.setPixmap(snap)
        overlay.setGeometry(0, 0, self.width(), self.height())
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.raise_()
        overlay.show()

        effect = QGraphicsOpacityEffect(overlay)
        overlay.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(180)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InOutQuad)

        def _done():
            overlay.deleteLater()
            self.hide()

        self._overlay_anim = anim
        anim.finished.connect(_done)
        anim.start()

    def go_back(self):
        if self.prev_window is not None:
            try:
                self.prev_window.show(); self.prev_window.raise_(); self.prev_window.activateWindow()
            except Exception:
                pass
        self.close()

    def closeEvent(self, e):
        #self._restore_global_font()
        super().closeEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self.offset = e.pos()
    def mouseMoveEvent(self, e):
        if hasattr(self, 'offset') and e.buttons() == Qt.LeftButton:
            self.move(self.pos() + e.pos() - self.offset)


def launch_training_window(num_categories, samples_per_class, input_vector_length, selected_mem_kb, prev_window=None):
    window = Custom_2_Window(
        num_categories=num_categories,
        samples_per_class=samples_per_class,
        input_vector_length=input_vector_length,
        selected_mem_kb=selected_mem_kb,
        prev_window=prev_window
    )
    window.show()