import sys
import os
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from ui_mainwindow import Ui_MainWindow  # GUI main window 
from hardmode_window import Ui_HardmodeWindow   # easymode window
from existing_mode_window import SubWindow   # easymode window
from custom_1 import Custom_1_Window         # custom_1 window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        # 이미지 로딩
        img_path = os.path.join(os.path.dirname(__file__), "intellino_TM.png")
        img_path2 = os.path.join(os.path.dirname(__file__), "intellino_TM_transparent.png")
        img_path3 = os.path.join(os.path.dirname(__file__), "home.png")

        if os.path.exists(img_path):
            self.ui.imageLabel.setPixmap(QPixmap(img_path))
            self.ui.imageLabel.setScaledContents(True)
        else:
            print("이미지 파일 없음:", img_path)

        # intellino_TM_transparent.png → 다른 QLabel에 표시한다고 가정 (예: logoLabel)
        if hasattr(self.ui, 'logo_Label'):
            if os.path.exists(img_path2):
                self.ui.logoLabel.setPixmap(QPixmap(img_path2).scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                print("이미지 파일 없음:", img_path2)

        # home.png → 버튼 아이콘으로 설정 (예: closeBtn)
        if hasattr(self.ui, 'close_btn'):
            if os.path.exists(img_path3):
                self.ui.closeBtn.setIcon(QIcon(img_path3))
                self.ui.closeBtn.setIconSize(QSize(24, 24))
            else:
                print("이미지 파일 없음:", img_path3)


        self.ui.easyModeBtn.clicked.connect(self.subFunction)
        self.ui.hardModeBtn.clicked.connect(self.customFunction)


    def subFunction(self):

        self.sub_window = SubWindow()
        self.sub_window.show()

    def customFunction(self):

        self.custom_window = Custom_1_Window()
        self.custom_window.show()




#main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



