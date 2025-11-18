import sys
import os
from pathlib import Path

from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from ui_mainwindow import Ui_MainWindow  # GUI main window 
from existing_mode_window import SubWindow   # easymode window
from custom_1 import Custom_1_Window         # custom_1 window


def resource_path(relative_path: str) -> str:
    """
    PyInstaller로 빌드했을 때(_MEIPASS)와 개발 중(__file__ 기준) 모두에서
    리소스 파일을 안정적으로 찾기 위한 경로 헬퍼.
    - 기본: _MEIPASS(실행 중 임시 폴더) 또는 현재 파일 위치
    - 보너스: 빌드 시 dest를 'main'으로 준 경우도 자동으로 커버
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()
    candidates = [
        base / relative_path,            # dest='.' 로 넣은 경우
        base / "main" / relative_path,  # dest='main' 으로 넣은 경우
        base.parent / relative_path,    # 혹시 모를 상위 폴더 위치
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # 못 찾으면 1순위 경로를 그대로 반환(디버깅 메시지용)
    return str(candidates[0])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("iCore")

        self.ui.easyModeBtn.clicked.connect(self.subFunction)
        self.ui.hardModeBtn.clicked.connect(self.customFunction)

        # 이미지 경로를 resource_path로 해결
        img_path  = resource_path("intellino_TM.png")
        img_path2 = resource_path("intellino_TM_transparent.png")
        img_path3 = resource_path("home.png")

        if os.path.exists(img_path):
            self.ui.imageLabel.setPixmap(QPixmap(img_path))
            self.ui.imageLabel.setScaledContents(True)
        else:
            print("no image file:", img_path)
            # 보조 이미지들로 대체 시도 (선택 사항)
            for alt in (img_path2, img_path3):
                if os.path.exists(alt):
                    self.ui.imageLabel.setPixmap(QPixmap(alt))
                    self.ui.imageLabel.setScaledContents(True)
                    break

    def subFunction(self):
        self.sub_window = SubWindow()
        self.sub_window.show()

    def customFunction(self):
        self.custom_window = Custom_1_Window()
        self.custom_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
