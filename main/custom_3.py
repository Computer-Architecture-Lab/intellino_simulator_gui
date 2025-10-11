# custom_3.py
import sys, os, shutil, datetime, numpy as np
import traceback
from PIL import Image

ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LOGO_PATH = os.path.join(ASSETS_DIR, "intellino_TM_transparent.png")
HOME_ICON_PATH = os.path.join(ASSETS_DIR, "home.png")

from PySide2.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGraphicsDropShadowEffect, QTextEdit, QSizePolicy, QGraphicsOpacityEffect, QLabel
)
from PySide2.QtGui import QPixmap, QIcon, QMouseEvent, QColor
from PySide2.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from path_utils import get_dirs

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# 프로젝트 상대 경로 확보
CUSTOM_IMAGE_ROOT, NUMBER_IMAGE_DIR, DEFAULT_OUTPUT_ROOT = get_dirs(__file__)

BUTTON_STYLE = """
    QPushButton { background-color:#ffffff; border:1px solid #ccc; border-radius:10px; padding:5px; font-weight:bold; font-size:13px; }
    QPushButton:hover { background-color:#e9ecef; }
    QPushButton:pressed { background-color:#adb5bd; color:white; }
"""

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

        # 로고
        logo_label = QLabel()
        pm = QPixmap(LOGO_PATH)
        if pm.isNull():
            # 파일이 없을 때 대비
            logo_label.setText("intellino")
            logo_label.setStyleSheet("font-weight:600;")
        else:
            logo_label.setPixmap(pm.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 홈 버튼
        self.home_btn = QPushButton()
        icon = QIcon(HOME_ICON_PATH)
        self.home_btn.setIcon(icon)
        self.home_btn.setIconSize(QSize(24, 24))
        self.home_btn.setFixedSize(34, 34)
        self.home_btn.setStyleSheet("""
            QPushButton { border: none; background-color: transparent; }
            QPushButton:hover { background-color: #dee2e6; border-radius: 17px; }
        """)
        # 연결
        self.home_btn.clicked.connect(self._on_home_clicked)

        layout.addWidget(logo_label)
        layout.addStretch()
        layout.addWidget(self.home_btn)

        self._offset = None

    def _on_home_clicked(self):
        """홈 아이콘 클릭: 전역 스타일 초기화 후 창 닫기(메인 복귀 시 글씨 크기 이슈 방지)."""
        app = QApplication.instance()
        if app:
            app.setStyleSheet("")  # 전역 스타일 초기화 (메인으로 돌아갈 때 폰트 축소 방지)
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

class ProgressSection(QWidget):
    def __init__(self, title="6. Train"):
        super().__init__()
        g = QGroupBox(title); g.setStyleSheet("QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}")
        g.setFixedHeight(70)
        row = QHBoxLayout()
        from PySide2.QtWidgets import QProgressBar
        self.bar = QProgressBar(); self.bar.setValue(0); self.bar.setTextVisible(False); self.bar.setFixedHeight(8)
        self.bar.setStyleSheet("QProgressBar{border:1px solid #bbb;border-radius:3px;background:#f1f1f1;} QProgressBar::chunk{background:#3b82f6;border-radius:3px;}")
        self.perc = QLabel("0%"); self.perc.setAlignment(Qt.AlignRight|Qt.AlignVCenter); self.perc.setFixedWidth(50)
        row.addWidget(self.bar); row.addWidget(self.perc); g.setLayout(row)
        v = QVBoxLayout(self); v.addWidget(g)
    def update(self, v:int):
        v = max(0, min(100, int(v))); self.bar.setValue(v); self.perc.setText(f"{v}%")

class LogSection(QWidget):
    def __init__(self):
        super().__init__()
        g = QGroupBox("8. Result"); g.setStyleSheet("QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}")
        lay = QVBoxLayout()
        self.text = QTextEdit(); self.text.setReadOnly(True); self.text.setStyleSheet("QTextEdit{font-size:14px;border:1px solid #ccc;border-radius:8px;padding:10px;background:#f8f9fa;}")
        self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.text); g.setLayout(lay)
        v = QVBoxLayout(self); v.addWidget(g)
    def append(self, s:str): self.text.append(s)

def simple_kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 0):
    n = len(X); k = max(1, min(k, n))
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, size=k, replace=False)].astype(np.float64)

    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        new_centers = []

        for j in range(k):
            pts = X[labels == j]
            new_centers.append(pts.mean(axis=0) if len(pts)>0 else X[rng.randint(n)])
        new_centers = np.stack(new_centers, axis=0)

        if np.allclose(new_centers, centers): break
        centers = new_centers

    chosen_idx = []
    for j in range(k):
        d2 = ((X - centers[j]) ** 2).sum(axis=1)
        chosen_idx.append(int(np.argmin(d2)))
    return list(dict.fromkeys(chosen_idx))[:k]

def load_images_from_dir(dir_path: str):
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if str(f).lower().endswith(IMG_EXTS)]
    X=[]; keep=[]
    for p in files:
        try:
            img = Image.open(p).convert("L").resize((28,28))
            arr = np.asarray(img, dtype=np.float32)/255.0
            X.append(arr.flatten()); keep.append(p)
        except Exception:
            pass
    if not X: return np.empty((0,784), dtype=np.float32), []
    return np.stack(X, axis=0), keep

class SubWindow(QWidget):
    def __init__(self, selection, samples_per_class: int = 1, prev_window=None, output_root=DEFAULT_OUTPUT_ROOT):
        super().__init__()
        self.selection = selection
        self.samples_per_class = max(1, int(samples_per_class))
        self.output_root = output_root
        self.prev_window = prev_window
        self.win4 = None

        # ✅ 다음 창 전달에 필요한 카테고리 수 보장
        self.num_categories = len(self.selection)

        self._setup_ui()
        QTimer.singleShot(150, self._run_kmeans)

    def _setup_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setFixedSize(800,800)
        container = QWidget(self); container.setStyleSheet("background-color:white; border-radius:15px;"); container.setGeometry(0,0,800,800)
        shadow = QGraphicsDropShadowEffect(self); shadow.setBlurRadius(30); shadow.setColor(QColor(0,0,0,100)); self.setGraphicsEffect(shadow)
        self.title_bar = TitleBar(self); self.title_bar.setParent(container); self.title_bar.setGeometry(0,0,800,50)
        layout = QVBoxLayout(container); layout.setContentsMargins(20,60,20,20); layout.setSpacing(30)

        self.progress = ProgressSection(); layout.addWidget(self.progress)

        out_g = QGroupBox("7. Output folder")
        out_g.setStyleSheet("QGroupBox{font-weight:bold;border:1px solid #b0b0b0;border-radius:10px;margin-top:10px;padding:10px;} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 5px;}")
        row = QHBoxLayout()
        self.out_label = QLabel(self.output_root)
        self.out_label.setStyleSheet("font-size:13px;")
        self.open_btn = QPushButton("Open folder")
        self.open_btn.clicked.connect(self._open_output_folder)
        
        self.open_btn.setFixedSize(110,32) 
        self.open_btn.setStyleSheet(BUTTON_STYLE) 
        row.addWidget(self.out_label); row.addStretch(); row.addWidget(self.open_btn); out_g.setLayout(row)
        layout.addWidget(out_g)

        self.log = LogSection(); layout.addWidget(self.log)

        # 하단 버튼
        btn_row = QHBoxLayout()
        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(110, 38)
        self.next_btn.setStyleSheet(BUTTON_STYLE)
        self.next_btn.clicked.connect(self._go_next)
        self.next_btn.setEnabled(False)

        btn_row.addStretch()
        btn_row.addWidget(self.next_btn)
        layout.addLayout(btn_row)

    def _run_kmeans(self):
        import datetime, os, shutil
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(self.output_root, ts)
        os.makedirs(save_root, exist_ok=True)
        self.out_label.setText(save_root)

        total = len(self.selection)
        for i, item in enumerate(self.selection, start=1):
            dir_path = item["dir"]
            label = str(item["label"])
            X, srcs = load_images_from_dir(dir_path)
            if len(srcs) == 0:
                self.progress.update(int(i / total * 100))
                continue
            k = min(self.samples_per_class, len(srcs))
            chosen = simple_kmeans(X, k, max_iter=50, seed=0)
            dst_dir = os.path.join(save_root, label); os.makedirs(dst_dir, exist_ok=True)
            for r, idx in enumerate(chosen, start=1):
                src = srcs[idx]; base = os.path.basename(src)
                dst = os.path.join(dst_dir, f"{label}_sel{r:02d}_{base}")
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
            self.progress.update(int(i / total * 100))

        self.progress.update(100)
        self.log.text.clear()
        self.log.append("[Done] K-means selection completed and files saved.")
        self.next_btn.setEnabled(True)

    def _open_output_folder(self):
        p = self.out_label.text().strip()
        if not p: return
        try:
            if sys.platform.startswith("win"): os.startfile(p)           # type: ignore[attr-defined]
            elif sys.platform == "darwin": __import__("subprocess").Popen(["open", p])
            else: __import__("subprocess").Popen(["xdg-open", p])
        except Exception: pass

    def _go_next(self):
        # 1) 지연 임포트
        try:
            from custom_4 import ExperimentWindow as Window4
        except Exception as e:
            self.log.append(f"[Error] Failed to import next window: {e}\n{traceback.format_exc()}")
            return

        # 2) 다음 창 생성/유지
        try:
            if getattr(self, "win4", None) is None:
                # ✅ custom_2 → custom_3에서 전달된 카테고리 수 사용 보장
                self.win4 = Window4(num_categories=getattr(self, "num_categories", 0))

            # 위치/크기 맞추기
            try:
                self.win4.setGeometry(self.geometry())
            except Exception:
                self.win4.move(self.pos())

            # 3) 다음 창 먼저 정상 표시
            self.win4.show()
            self.win4.raise_()
            self.win4.activateWindow()

            # 4) 현재 창 스냅샷 → 다음 창 위 오버레이 라벨로 얹기
            snap = self.grab()

            overlay = QLabel(self.win4)
            overlay.setPixmap(snap)
            overlay.setGeometry(0, 0, self.width(), self.height())
            overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            overlay.raise_()
            overlay.show()

            eff = QGraphicsOpacityEffect(overlay)
            overlay.setGraphicsEffect(eff)

            anim = QPropertyAnimation(eff, b"opacity", self)
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

        except Exception as e:
            self.log.append(f"[Error] Failed to open next window: {e}\n{traceback.format_exc()}")


    # (미사용이지만 필요시 쓸 수 있도록 유지)
    def _create_next_button(self):
        self.next_btn = QPushButton("Next")
        self.next_btn.setFixedSize(100, 40)
        self.next_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                border: 1px solid #888; border-radius: 8px;
                background-color: #fefefe;
            }
            QPushButton:hover { background-color: #dee2e6; }
            QPushButton:disabled { background-color:#f1f3f5; color:#adb5bd; border:1px solid #ddd; }
        """)
        try:
            self.next_btn.clicked.disconnect()
        except Exception:
            pass
        self.next_btn.clicked.connect(self._go_next)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.next_btn)
        return row

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SubWindow(selection=[], samples_per_class=1)
    w.show()
    sys.exit(app.exec_())
