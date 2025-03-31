# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(631, 472)
        MainWindow.setMinimumSize(QSize(631, 472))
        MainWindow.setMaximumSize(QSize(631, 472))
        font = QFont()
        font.setFamily(u"Agency FB")
        MainWindow.setFont(font)
        MainWindow.setStyleSheet(u"background-color: white;\n"
"")
        MainWindow.setAnimated(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.easyModeBtn = QPushButton(self.centralwidget)
        self.easyModeBtn.setObjectName(u"easyModeBtn")
        self.easyModeBtn.setGeometry(QRect(30, 300, 261, 131))
        font1 = QFont()
        font1.setFamily(u"Bahnschrift")
        font1.setPointSize(16)
        font1.setBold(True)
        font1.setWeight(75)
        self.easyModeBtn.setFont(font1)
        self.easyModeBtn.setStyleSheet(u"""
QPushButton {
    background-color: qlineargradient(
        x1: 0, y1: 0,
        x2: 0, y2: 1,
        stop: 0 #00BFFF,
        stop: 1 #48D1CC
    );
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 16px;
}
QPushButton:hover {
    background-color: #1aa3ff;
}
""")  
        self.hardModeBtn = QPushButton(self.centralwidget)
        self.hardModeBtn.setObjectName(u"hardModeBtn")
        self.hardModeBtn.setGeometry(QRect(340, 300, 261, 131))
        self.hardModeBtn.setFont(font1)
        self.hardModeBtn.setStyleSheet(u"""
QPushButton {
    background-color: qlineargradient(
        x1: 0, y1: 0,
        x2: 0, y2: 1,
        stop: 0 #0000CD,
        stop: 1 #1E90FF
    );
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 16px;
}
QPushButton:hover {
    background-color: #1aa3ff;
}
""")
        self.imageLabel = QLabel(self.centralwidget)
        self.imageLabel.setObjectName(u"imageLabel")
        self.imageLabel.setGeometry(QRect(120, 30, 391, 251))
        self.imageLabel.setPixmap(QPixmap(u"intellino_TM.png"))
        self.imageLabel.setScaledContents(False)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.easyModeBtn.setText(QCoreApplication.translate("MainWindow", u"Existing datasets mode", None))
        self.hardModeBtn.setText(QCoreApplication.translate("MainWindow", u"Custom datasets mode", None))
        self.imageLabel.setText("")
    # retranslateUi

