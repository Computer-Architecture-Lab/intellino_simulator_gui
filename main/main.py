import sys
import os
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow
from ui_mainwindow import Ui_MainWindow  # GUI main window 
from easymode_window import Ui_EasymodeWindow   # easymode window


# # qt 플랫폼 플러그인 경로 설정정
# env_root = os.path.join(os.path.dirname(sys.executable), "Library", "plugins")
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(env_root, "platforms")
# os.environ["QT_PLUGIN_PATH"] = env_root  # 추가!

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        # 이미지 로딩
        img_path = os.path.join(os.path.dirname(__file__), "intellino_TM.png")
        if os.path.exists(img_path):
            self.ui.imageLabel.setPixmap(QPixmap(img_path))
            self.ui.imageLabel.setScaledContents(True)
        else:
            print("이미지 파일을 찾을 수 없습니다:", img_path)


        self.ui.easyModeBtn.clicked.connect(self.easyBtnFunction)
        self.ui.hardModeBtn.clicked.connect(self.hardBtnFunction)

    def easyBtnFunction(self):
        self.easyMode_window = EasyModeWindow()
        self.easyMode_window.show()

    def hardBtnFunction(self):
        print("open")


class EasyModeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_EasymodeWindow()
        self.ui.setupUi(self)

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())