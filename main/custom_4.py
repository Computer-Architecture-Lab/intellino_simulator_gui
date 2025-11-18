import sys
import os
from pathlib import Path

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QSizePolicy
)
from PySide2.QtGui import QPixmap, QIcon, QColor, QMouseEvent, QPainter, QPalette
from PySide2.QtCore import Qt, QSize

# matplotlib 임베딩
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl

# ──────────────────────────────────────────────
# exe/개발 환경 공통 리소스 경로 헬퍼
def resource_path(relative_path: str) -> str:
    """
    PyInstaller(onefile) 실행 시 임시 폴더(sys._MEIPASS)와
    개발 환경(__file__ 기준)을 모두 커버.
    빌드 때 dest가 '.' 또는 'main'이어도 자동 탐색.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()
    candidates = [
        base / relative_path,            # --add-data "...;."
        base / "main" / relative_path,  # --add-data "...;main"
        base.parent / relative_path,    # 혹시 상위 폴더에 있을 때
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])  # 못 찾으면 1순위 경로 반환(디버깅용)
# ──────────────────────────────────────────────

# 리소스 경로
LOGO_PATH = resource_path("intellino_TM_transparent.png")
HOME_ICON_PATH = resource_path("home.png")

# ── 실험 상태 전역 ─────────────────────────────────────────
class ExperimentState:
    MAX_RUNS = 5
    def __init__(self):
        self.runs = []  # [(label:str, acc:float), ...]

    def add_run(self, label: str, acc: float):
        if label is None or acc is None:
            return
        if len(self.runs) < self.MAX_RUNS:
            self.runs.append((str(label), float(acc)))

    def is_full(self) -> bool:
        return len(self.runs) >= self.MAX_RUNS

    def clear(self):
        self.runs.clear()

    def get_labels_accs(self):
        labels = [lb for lb, _ in self.runs]
        accs   = [float(ac) for _, ac in self.runs]
        return labels, accs

EXPERIMENT_STATE = ExperimentState()
# ───────────────────────────────────────────────────────────

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

# ── 타이틀 바 ──
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
        pm = QPixmap(LOGO_PATH)
        if pm.isNull():
            logo_label.setText("intellino")
            logo_label.setStyleSheet("font-weight:600;")
        else:
            pix = pm.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pix)

        home_btn = QPushButton()
        home_btn.setIcon(QIcon(HOME_ICON_PATH))
        home_btn.setIconSize(QSize(24, 24))
        home_btn.setFixedSize(34, 34)
        home_btn.setStyleSheet("""
            QPushButton { border: none; background-color: transparent; }
            QPushButton:hover { background-color: #dee2e6; border-radius: 17px; }
        """)
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


# ── Matplotlib 캔버스 ──
class AccuracyCanvas(FigureCanvas):
    """
    - X축 슬롯: 항상 5칸 고정
    - 막대 색상/두께: 기존 유지
    - ✅ 배경을 항상 완전 불투명(white)으로 강제
    """
    DEFAULT_BAR_WIDTH = 0.8
    BAR_WIDTH_SCALE   = 1.0 / 2.0
    BAR_COLOR         = "cornflowerblue"

    def __init__(self, parent=None):
        # Matplotlib 전역 배경/투명 설정 방지
        mpl.rcParams['figure.facecolor'] = 'white'
        mpl.rcParams['axes.facecolor']   = 'white'
        mpl.rcParams['savefig.facecolor'] = 'white'
        mpl.rcParams['savefig.transparent'] = False

        fig = Figure(figsize=(5.0, 4.0), tight_layout=True, facecolor='white', edgecolor='white')
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ✅ Qt 위젯 배경을 완전 불투명 흰색으로 강제
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)     # Qt가 투명 합성하지 않도록 힌트
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)

        pal = self.palette()
        pal.setColor(QPalette.Window, Qt.white)
        pal.setColor(QPalette.Base,   Qt.white)
        self.setPalette(pal)
        self.setStyleSheet("background-color: white;")

        self.ax = self.figure.add_subplot(111)
        self._init_axes()

    def _init_axes(self):
        self.ax.clear()

        # ✅ Figure/Axes 모두 흰색·불투명 강제
        self.figure.patch.set_facecolor('white')
        self.figure.patch.set_alpha(1.0)
        self.ax.set_facecolor('white')
        self.ax.patch.set_alpha(1.0)

        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Accuracy (%)")
        self.ax.set_xlabel("Parameters")
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # ✅ 페인트 전에 전체 영역을 흰색으로 칠해 투명 채널/합성 제거
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.white)
        p.end()
        super().paintEvent(event)

    def update_plot(self, labels, accuracies):
        self._init_axes()

        max_runs = ExperimentState.MAX_RUNS
        labels     = list(labels or [])
        accuracies = [float(a) for a in (accuracies or [])]

        # 5칸 고정 패딩
        if len(labels) < max_runs:
            pad = max_runs - len(labels)
            labels     += [""] * pad
            accuracies += [0.0] * pad

        xs = list(range(1, max_runs + 1))
        bar_w = self.DEFAULT_BAR_WIDTH * self.BAR_WIDTH_SCALE
        self.ax.bar(xs, accuracies, color=self.BAR_COLOR, width=bar_w)

        self.ax.set_xlim(0.5, max_runs + 0.5)
        self.ax.set_xticks(xs)
        self.ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

        # 값 라벨: 0%는 생략
        for xi, acc in zip(xs, accuracies):
            if acc > 0:
                self.ax.text(xi, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=9)

        self.draw_idle()


# 9. Experiment graph 섹션
class ExperimentGraphSection(QWidget):
    def __init__(self):
        super().__init__()

        group = QGroupBox("9. Experiment graph")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 14px;
                border: 1px solid #a9a9a9;
                border-radius: 12px;
                margin-top: 10px;
                /* ▶ 전체 패딩을 살짝 줄여 테두리 안으로 자연스럽게 */
                padding: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
        """)

        v = QVBoxLayout()
        v.setContentsMargins(10, 10, 10, 16)
        v.setSpacing(8)

        # ✅ 백플레이트(완전 흰색) 위에 캔버스를 얹고, 섀도우는 백플레이트에만 적용
        self.backplate = QWidget()
        self.backplate.setStyleSheet("background-color: white; border-radius: 8px;")
        bp_layout = QVBoxLayout(self.backplate)
        bp_layout.setContentsMargins(12, 12, 12, 12)  # 캔버스와 가장자리 간격
        bp_layout.setSpacing(0)

        self.canvas = AccuracyCanvas()
        self.canvas.setMinimumHeight(520)
        bp_layout.addWidget(self.canvas)

        shadow = QGraphicsDropShadowEffect(self.backplate)
        shadow.setBlurRadius(18)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 0)
        self.backplate.setGraphicsEffect(shadow)

        v.addWidget(self.backplate)
        v.addSpacing(8)

        group.setLayout(v)

        main = QVBoxLayout(self)
        main.addWidget(group)

    def refresh(self):
        labels, accs = EXPERIMENT_STATE.get_labels_accs()
        self.canvas.update_plot(labels, accs)


# 메인 창
class ExperimentWindow(QWidget):
    def __init__(self, num_categories: int = 0):
        super().__init__()
        self.num_categories = num_categories
        self._custom1_window = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 프레임리스 둥근 모서리 유지
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

        # 9. Experiment graph
        self.graph_section = ExperimentGraphSection()
        self.graph_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.graph_section)

        layout.addStretch(1)

        # 하단 버튼들
        btn_col = QVBoxLayout()
        btn_col.setSpacing(12)

        self.reconf_btn = QPushButton("Reconfigure")
        self.reconf_btn.setFixedSize(120, 40)
        self.reconf_btn.setStyleSheet(BUTTON_STYLE)
        self.reconf_btn.clicked.connect(self._open_reconfigure)

        self.finish_btn = QPushButton("Finish")
        self.finish_btn.setFixedSize(120, 40)
        self.finish_btn.setStyleSheet(BUTTON_STYLE)
        self.finish_btn.clicked.connect(self._on_finish_clicked)  # Finish 시 초기화

        btn_col.addWidget(self.reconf_btn)
        btn_col.addWidget(self.finish_btn)

        btn_container = QWidget()
        btn_container.setLayout(btn_col)
        btn_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(btn_container, 0, Qt.AlignRight | Qt.AlignBottom)

        # 초기 상태 반영
        self._update_controls()
        self.refresh_graph()

    def _update_controls(self):
        full = EXPERIMENT_STATE.is_full()
        self.reconf_btn.setEnabled(not full)
        if full:
            self.reconf_btn.setToolTip("Maximum 5 runs reached. Please finish (Home).")
        else:
            self.reconf_btn.setToolTip("Configure parameters and run more experiments.")

    def refresh_graph(self):
        self.graph_section.refresh()

    def showEvent(self, e):
        self._update_controls()
        self.refresh_graph()
        super().showEvent(e)

    def _open_reconfigure(self):
        # 5회 도달 시 재설정 금지, 홈으로만
        if EXPERIMENT_STATE.is_full():
            self.close()
            return

        from custom_1 import Custom_1_Window, GLOBAL_FONT_QSS
        self._custom1_window = Custom_1_Window(prev_window=self)
        try:
            self._custom1_window.setStyleSheet(GLOBAL_FONT_QSS)
        except Exception:
            pass

        self._custom1_window.show()
        self.hide()

    def _on_finish_clicked(self):
        """
        Finish 누르면 그래프 초기화 → 전역 상태 비우고 창 닫기
        """
        try:
            EXPERIMENT_STATE.clear()
        finally:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ExperimentWindow()
    w.show()
    sys.exit(app.exec_())
