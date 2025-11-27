# custom_2.py
import sys, os, subprocess
from functools import partial

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QGraphicsDropShadowEffect, QFileDialog, QScrollArea,
    QMessageBox, QSizePolicy
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor
from PySide2.QtCore import Qt, QSize, Signal

from custom_3 import SubWindow as Window3
from utils.path_utils import get_dirs

from utils.resource_utils import resource_path
from utils.ui_common import BUTTON_STYLE
#=======================================================================================================#

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = resource_path("image/intellino_TM_transparent.png")
HOME_ICON_PATH = resource_path("image/home.png")

MESSAGE_BOX_QSS = """
QMessageBox {
    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-weight: 500;
    font-size: 14px;
}
"""

# ---------------------------------------------------------------------
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
        font-weight: 500;
        font-size: 14px;
        color: #000000;
    }
"""
# ---------------------------------------------------------------------

# get_dirs 결과 중 BASE_NUMBER_DIR만 실제로 사용
_, BASE_NUMBER_DIR, _ = get_dirs(__file__)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

BUTTON_STYLE_LOCAL = BUTTON_STYLE  # alias (필요 시 커스터마이징 가능)
#=======================================================================================================#
#                                              function                                                 #
#=======================================================================================================#
# custom_1에서 import 해서 사용하는 함수
def launch_training_window(num_categories, samples_per_class, input_vector_length, selected_mem_kb, prev_window=None):
    """
    custom_1에서 호출하는 진입 함수.
    전달받은 파라미터로 Custom_2_Window를 띄워준다.
    """
    window = Custom_2_Window(
        num_categories=num_categories,
        samples_per_class=samples_per_class,
        input_vector_length=input_vector_length,
        selected_mem_kb=selected_mem_kb,
        prev_window=prev_window,
    )
    window.show()
    return window
#=======================================================================================================#
#                                               UI 구성                                                  #
#=======================================================================================================#
class TrainDatasetGroup(QGroupBox):
    """
    4. Datasets of each category

    - C(=num_categories) 개수만큼 행을 만든다.
    - 각 행에서 폴더를 하나 선택하면
      → 그 폴더 안의 이미지(.png, .jpg, .jpeg, .bmp) 전체를
         해당 category의 'raw sample data'로 보관만 한다.
    - 실제로 sample / test 로 나누는 작업은
      custom_3(Window3)에서 selection['files']를 받아서 별도로 수행한다.
    """
    completeness_changed = Signal(bool)

    MIN_IMAGES_PER_CATEGORY = 12

#카테고리 수를 받아서 UI(각 카테고리별 폴더 선택 행)를 초기화하고 표시할 준비를 한다.
    def __init__(self, num_categories=3, base_dir=BASE_NUMBER_DIR):
        super().__init__("4. Datasets of each category")

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

        # 폴더 선택 시 기본 시작 위치로 사용할 경로
        self.base_dir = os.path.abspath(base_dir)

        self.category_inputs = []   # (dir_input, label_input)
        self.row_widgets = []       # {"dir_input":..,"label_input":..,"count_label":..}
        self._last_complete = None

        self._build_ui(num_categories)
        self._on_fields_changed()

#C개의 카테고리 입력 행 생성(UI 구성), 폴더 입력창·라벨·이미지 개수 표시 뱃지를 만든다.
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

            dir_input = QLineEdit()
            dir_input.setPlaceholderText("Select folder (any local folder)")
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
            # 어디든 선택 가능
            browse_btn.clicked.connect(partial(self.browse_any_folder, dir_input))

            # 폴더 안 이미지 개수 표시 (예: "12 images")
            count_badge = QLabel("0 images")
            count_badge.setFixedWidth(90)
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
            self.row_widgets.append(
                {"dir_input": dir_input, "label_input": label_input, "count_label": count_badge}
            )
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

#이미지 개수 n에 따라 초록(이미지 있음) / 빨강(없음) 스타일시트를 반환한다.
    def _badge_style(self, n):
        """
        단순히 '이미지가 있냐/없냐'만 시각적으로 표시.
        n > 0  → 초록 (OK)
        n == 0 → 빨강 (데이터 없음)
        """
        if n > self.MIN_IMAGES_PER_CATEGORY:
            return (
                "QLabel { background:#e6fcf5; border:1px solid #37b24d; "
                "color:#2b8a3e; border-radius:6px; padding:4px; }"
            )
        else:
            return (
                "QLabel { background:#ffe3e3; border:1px solid #fa5252; "
                "color:#c92a2a; border-radius:6px; padding:4px; }"
            )

#주어진 폴더 안에서 .png/.jpg/.jpeg/.bmp 이미지 파일 개수를 센다
    def _count_images_in_dir(self, path: str) -> int:
        if not os.path.isdir(path):
            return 0
        try:
            return sum(1 for f in os.listdir(path) if str(f).lower().endswith(IMG_EXTS))
        except Exception:
            return 0

    def _update_row_status(self, di, cl):
        path = di.text().strip()
        n = self._count_images_in_dir(path)
        cl.setText(f"{n} images")
        cl.setStyleSheet(self._badge_style(n))

    def browse_any_folder(self, dir_input):
        """로컬 디스크 어디든 폴더 선택 허용. 폴더 안의 이미지는 '전부' 보관."""
        start_dir = self.base_dir if os.path.isdir(self.base_dir) else os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self,
            "Select folder (contains images)",
            start_dir,
            QFileDialog.ShowDirsOnly
        )
        if not path:
            return

        dir_input.setText(path)
        # 실제 개수는 _on_fields_changed → _update_row_status 에서 갱신.

    def is_complete(self) -> bool:
        """
        각 행이 '완성'되었다고 판단하는 조건:
        - 디렉터리 경로 입력됨
        - 디렉터리 실제 존재
        - 이미지가 1장 이상 존재
        - 라벨 텍스트가 비어 있지 않음
        """
        for row in self.row_widgets:
            d = row["dir_input"].text().strip()
            l = row["label_input"].text().strip()
            if not d or not l:
                return False
            if not os.path.isdir(d):
                return False
            n = self._count_images_in_dir(d)
            if n < self.MIN_IMAGES_PER_CATEGORY:
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
        """
        각 category에 대해:
        - dir   : 선택된 폴더 경로
        - label : 사용자가 입력한 라벨
        - files : 폴더 안의 이미지 파일 전체 경로 리스트
        를 반환한다.
        (여기서는 '보관만' 하고, 실제 sample / test split은
         custom_3에서 selection['files'] 기반으로 수행)
        """
        items = []
        for idx, row in enumerate(self.row_widgets, start=1):
            d = row["dir_input"].text().strip()
            l = row["label_input"].text().strip()

            files = []
            if os.path.isdir(d):
                try:
                    files = [
                        os.path.join(d, f)
                        for f in os.listdir(d)
                        if str(f).lower().endswith(IMG_EXTS)
                    ]
                except Exception:
                    files = []

            items.append({
                "category_index": idx,
                "dir": d,
                "label": l,
                "files": files,   # 폴더 안 raw sample data 전부
            })
        return items

#=======================================================================================================#
#                                                 main                                                  #
#=======================================================================================================#
class Custom_2_Window(QWidget):
    def __init__(self, num_categories=3, samples_per_class=1, input_vector_length=0, selected_mem_kb=None, prev_window=None):
        super().__init__()
        self.num_categories = num_categories
        self.samples_per_class = max(1, int(samples_per_class))
        self.input_vector_length = int(input_vector_length or 0)
        self.selected_mem_kb = selected_mem_kb
        self.win3 = None
        self.prev_window = prev_window
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
        shadow.setColor(QColor(0,0,0,120))
        container.setGraphicsEffect(shadow)

        self._add_title_bar(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 60, 20, 20)
        layout.setSpacing(24)

        # 여기서는 class별 raw sample data 폴더만 선택
        self.dataset_group = TrainDatasetGroup(
            num_categories=self.num_categories,
            base_dir=BASE_NUMBER_DIR,
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
        bar = QWidget(parent)
        bar.setGeometry(0,0,800,50)
        bar.setStyleSheet("background-color:#f1f3f5; border-top-left-radius:15px; border-top-right-radius:15px;")
        h = QHBoxLayout(bar)
        h.setContentsMargins(15,0,15,0)

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
        close_btn.setStyleSheet(
            "QPushButton{border:none;background:transparent;}"
            "QPushButton:hover{background:#dee2e6;border-radius:17px;}"
        )
        h.addWidget(logo)
        h.addStretch()
        h.addWidget(close_btn)

        # 드래그로 창 이동
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
        self.back_btn = QPushButton("Back")
        self.back_btn.setFixedSize(100,40)
        self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(self.go_back)

        # 버튼 라벨을 'Start'로 변경
        self.next_btn = QPushButton("Start")
        self.next_btn.setFixedSize(100,40)
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.clicked.connect(self.start_kmeans)

        row = QHBoxLayout()
        row.addWidget(self.back_btn)
        row.addStretch()
        row.addWidget(self.next_btn)
        return row

    def start_kmeans(self):
        """
        Start 버튼 핸들러.
        선택된 폴더 내 raw sample 이미지 전체를 다음 창(custom_3.SubWindow)에 넘긴다.
        (실제 sample / test split은 custom_3에서 selection['files'] 기반으로 수행)
        """
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

        # 스냅샷/페이드 애니 없이, 현재 창만 숨김
        self.hide()

    def go_back(self):
        if self.prev_window is not None:
            try:
                self.prev_window.show()
                self.prev_window.raise_()
                self.prev_window.activateWindow()
            except Exception:
                pass
        self.close()

    def closeEvent(self, e):
        super().closeEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.offset = e.pos()

    def mouseMoveEvent(self, e):
        if hasattr(self, 'offset') and e.buttons() == Qt.LeftButton:
            self.move(self.pos() + e.pos() - self.offset)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Custom_2_Window()
    w.show()
    sys.exit(app.exec_())
