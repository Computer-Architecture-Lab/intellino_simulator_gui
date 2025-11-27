import sys
import os
from pathlib import Path

from PySide2.QtGui import QPixmap, QIcon
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from ui_mainwindow import Ui_MainWindow                 # GUI main window 
from existing_mode_window import SubWindow              # easymode window
from custom_1 import Custom_1_Window                    # custom_1 window

from utils.resource_utils import resource_path


GLOBAL_FONT_QSS = """
* {
    font-family: 'Inter', 'Pretendard', 'Noto Sans', 'Segoe UI',
                 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-size: 14px;
    font-weight: 500;
}
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("iCore")

        self.ui.easyModeBtn.clicked.connect(self.subFunction)
        self.ui.hardModeBtn.clicked.connect(self.customFunction)

        img_path  = resource_path("image/intellino_TM.png")
        img_path2 = resource_path("image/intellino_TM_transparent.png")
        img_path3 = resource_path("image/home.png")

        print("img_path :", img_path,  os.path.exists(img_path))
        print("img_path2:", img_path2, os.path.exists(img_path2))
        print("img_path3:", img_path3, os.path.exists(img_path3))

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

    icon_path = resource_path("icore_image.ico")
    app.setWindowIcon(QIcon(icon_path))

    # 모든 창에 동일 폰트 적용: 여기서 단 한 번만!
    app.setStyleSheet(GLOBAL_FONT_QSS)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())