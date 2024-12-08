# coding: utf-8
import sys

from PySide6.QtWidgets import QHeaderView
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication, QStyleOptionViewItem, QTableWidget, QTableWidgetItem,
    QWidget, QHBoxLayout, QLabel, QVBoxLayout,QDialog
)

from qfluentwidgets import (
    TableWidget, isDarkTheme, setTheme, Theme, TableView,
    TableItemDelegate, setCustomStyleSheet
)

class CurationQWidget(QDialog):
    def __init__(self):
        super().__init__()

        #  Style adjustments
        self.setWindowTitle("Curation")
        self.setStyleSheet("CurationQWidget{background: rgb(255, 255, 255)} ")
        self.hBoxLayout = QHBoxLayout(self)
        self.resize(550, 650)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CurationQWidget()
    w.show()
    app.exec()
