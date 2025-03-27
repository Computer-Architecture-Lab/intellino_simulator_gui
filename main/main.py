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


class Sub_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(800, 600)

        # 리소스 경로 기준: 현재 파일 위치
        base_dir = os.path.dirname(__file__)
        img_path2 = os.path.join(base_dir, "intellino_TM_transparent.png")
        img_path3 = os.path.join(base_dir, "home.png")

        # ... (기존 UI 코드 생략)

        # logo_label에 이미지 설정
        logo_label = QLabel()
        if os.path.exists(img_path2):
            pixmap = QPixmap(img_path2).scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            print("❌ 로고 이미지 없음:", img_path2)

        # close_btn에 아이콘 설정
        close_btn = QPushButton()
        if os.path.exists(img_path3):
            close_btn.setIcon(QIcon(img_path3))
            close_btn.setIconSize(QSize(24, 24))
        else:
            print("❌ 버튼 아이콘 이미지 없음:", img_path3)




#main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



