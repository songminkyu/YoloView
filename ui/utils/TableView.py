# coding: utf-8
import sys

from PySide6.QtWidgets import QHeaderView
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem, QTableWidget, QTableWidgetItem, QWidget, QHBoxLayout

from qfluentwidgets import TableWidget, isDarkTheme, setTheme, Theme, TableView, TableItemDelegate, setCustomStyleSheet


class TableViewDelegate(TableItemDelegate):
    """ Custom table item delegate """

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex):
        super().initStyleOption(option, index)
        if index.column() != 1:
            return

        if isDarkTheme():
            option.palette.setColor(QPalette.Text, Qt.white)
            option.palette.setColor(QPalette.HighlightedText, Qt.white)
        else:
            option.palette.setColor(QPalette.Text, Qt.black)
            option.palette.setColor(QPalette.HighlightedText, Qt.black)


class TableViewDelegate(TableItemDelegate):
    """Custom table item delegate"""

    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex):
        super().initStyleOption(option, index)
        if index.column() != 1:
            return

        if isDarkTheme():
            option.palette.setColor(QPalette.Text, Qt.white)
            option.palette.setColor(QPalette.HighlightedText, Qt.white)
        else:
            option.palette.setColor(QPalette.Text, Qt.black)
            option.palette.setColor(QPalette.HighlightedText, Qt.black)


class TableViewQWidget(QWidget):
    def __init__(self, infoList=None, errorList=None):
        super().__init__()
        self.setWindowTitle("Result Statistics")
        self.hBoxLayout = QHBoxLayout(self)

        # Original table for successful detections
        self.tableView = TableWidget(self)
        self.tableView.setItemDelegate(TableViewDelegate(self.tableView))
        self.tableView.setSelectRightClickedRow(True)
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)
        self.tableView.setWordWrap(False)
        self.tableView.setRowCount(1000)
        self.tableView.setColumnCount(3)
        self.Infos = infoList if infoList else []
        info_count = 1
        for i, info in enumerate(self.Infos):
            self.tableView.setItem(i, 0, QTableWidgetItem(str(info_count)))
            info_count += 1
            for j in range(1, len(info) + 1):
                self.tableView.setItem(i, j, QTableWidgetItem(info[j - 1]))

        self.tableView.verticalHeader().hide()
        self.tableView.setHorizontalHeaderLabels(["Index", "Class", "Frequency"])
        self.tableView.resizeColumnsToContents()
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.setSortingEnabled(True)

        # New table for failed detections and error messages
        self.errorTableView = TableWidget(self)
        self.errorTableView.setItemDelegate(TableViewDelegate(self.errorTableView))
        self.errorTableView.setSelectRightClickedRow(True)
        self.errorTableView.setBorderVisible(True)
        self.errorTableView.setBorderRadius(8)
        self.errorTableView.setWordWrap(False)
        self.errorTableView.setRowCount(1000)
        self.errorTableView.setColumnCount(3)
        self.Errors = errorList if errorList else []
        error_count = 1
        for i, error in enumerate(self.Errors):
            self.errorTableView.setItem(i, 0, QTableWidgetItem(str(error_count)))
            error_count += 1
            for j in range(1, len(error) + 1):
                self.errorTableView.setItem(i, j, QTableWidgetItem(error[j - 1]))

        self.errorTableView.verticalHeader().hide()
        self.errorTableView.setHorizontalHeaderLabels(["Index", "File Name", "Result"])
        self.errorTableView.resizeColumnsToContents()
        self.errorTableView.setColumnWidth(0, 100)  # Index 컬럼
        self.errorTableView.setColumnWidth(1, 150)  # File Name 컬럼
        self.errorTableView.setColumnWidth(2, 300)  # Result 컬럼
        self.errorTableView.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.errorTableView.setSortingEnabled(True)

        # Adjust layout to place tables side by side
        self.hBoxLayout.setContentsMargins(20, 10, 20, 10)
        self.hBoxLayout.addWidget(self.tableView)
        self.hBoxLayout.addWidget(self.errorTableView)
        self.resize(1100, 550)

        # Style adjustments
        self.setStyleSheet("TableViewQWidget{background: rgb(255, 255, 255)} ")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TableViewQWidget()
    w.show()
    app.exec()
