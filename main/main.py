import sys
import os
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from ui_mainwindow import Ui_MainWindow  # GUI main window 
from existing_mode_window import SubWindow   # easymode window
from custom_1 import Custom_1_Window         # custom_1 window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # BTN click시 정의
        self.ui.easyModeBtn.clicked.connect(self.subFunction)
        self.ui.hardModeBtn.clicked.connect(self.customFunction)

        # 이미지 로딩
        img_path = os.path.join(os.path.dirname(__file__), "intellino_TM.png")
        img_path2 = os.path.join(os.path.dirname(__file__), "intellino_TM_transparent.png")
        img_path3 = os.path.join(os.path.dirname(__file__), "home.png")

        if os.path.exists(img_path):
            self.ui.imageLabel.setPixmap(QPixmap(img_path))
            self.ui.imageLabel.setScaledContents(True)
        else:
            print("이미지 파일 없음:", img_path)

    # 버튼 눌렀을 때 실행할 함수
    def subFunction(self):
        self.sub_window = SubWindow()
        self.sub_window.show()

    # 버튼 눌렀을 때 실행할 함수
    def customFunction(self):
        self.custom_window = Custom_1_Window()
        self.custom_window.show()




#main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



