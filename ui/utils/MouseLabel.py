from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal
from YOLOSHOW import *

# class LabelMouse(QLabel):
#     double_clicked = Signal()
#
#     # 마우스 더블클릭 이벤트
#     def mouseDoubleClickEvent(self, event):
#         self.double_clicked.emit()
#
#     def mouseMoveEvent(self):
#         """
#         label2 위로 마우스를 이동할 때 이벤트가 트리거됩니다.
#         :return:
#         """
#         print('label2 위로 마우스를 이동할 때 이벤트가 트리거됩니다')
#

class Label_click_Mouse(QLabel):
    clicked = Signal()

    # 마우스 클릭 이벤트
    def mousePressEvent(self, event):
        self.clicked.emit()
