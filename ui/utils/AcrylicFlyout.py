# coding:utf-8
from typing import Union
from PySide6.QtCore import QPoint, Qt, QRect, QRectF, Signal, QSize, QMargins, QPropertyAnimation, QObject, \
    QParallelAnimationGroup, QEasingCurve
from PySide6.QtGui import QPixmap, QPainter, QColor, QPainterPath, QIcon, QImage, QCursor, QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QApplication, QGraphicsDropShadowEffect,
                               QFrame,QMainWindow, QScrollArea, QTableWidgetItem, QHeaderView, QSplitter)

from qfluentwidgets import TableWidget, TableItemDelegate
from qfluentwidgets import  FluentIconBase, FlyoutAnimationType, \
    FlyoutAnimationManager, drawIcon, isDarkTheme, ImageLabel, TransparentToolButton, FluentIcon, FluentStyleSheet, \
    TextWrap
from qfluentwidgets.common.screen import getCurrentScreenGeometry
from qfluentwidgets.components.material import AcrylicWidget
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams

class IconWidget(QWidget):

    def __init__(self, icon, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(36, 54)
        self.icon = icon

    def paintEvent(self, e):
        if not self.icon:
            return

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing |
                               QPainter.SmoothPixmapTransform)

        rect = QRectF(8, (self.height() - 20) / 2, 20, 20)
        drawIcon(self.icon, painter, rect)

class FlyoutViewBase(QWidget):
    """ Flyout view base class """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def addWidget(self, widget: QWidget, stretch=0, align=Qt.AlignLeft):
        raise NotImplementedError

    def backgroundColor(self):
        return QColor(40, 40, 40) if isDarkTheme() else QColor(248, 248, 248)

    def borderColor(self):
        return QColor(0, 0, 0, 45) if isDarkTheme() else QColor(0, 0, 0, 17)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        painter.setBrush(self.backgroundColor())
        painter.setPen(self.borderColor())

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)

class FlyoutView(FlyoutViewBase):
    """ Flyout view """

    closed = Signal()

    def __init__(self, title: str, content: str, icon: Union[FluentIconBase, QIcon, str] = None,
                 image: Union[str, QPixmap, QImage] = None, isClosable=False, parent=None):
        super().__init__(parent=parent)
        """
        Parameters
        ----------
        title: str
            the title of teaching tip

        content: str
            the content of teaching tip

        icon: InfoBarIcon | FluentIconBase | QIcon | str
            the icon of teaching tip

        image: str | QPixmap | QImage
            the image of teaching tip

        isClosable: bool
            whether to show the close button

        parent: QWidget
            parent widget
        """
        self.icon = icon
        self.title = title
        self.image = image
        self.content = content
        self.isClosable = isClosable

        self.vBoxLayout = QVBoxLayout(self)
        self.viewLayout = QHBoxLayout()
        self.widgetLayout = QVBoxLayout()

        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content, self)
        self.iconWidget = IconWidget(icon, self)
        self.imageLabel = ImageLabel(self)
        self.closeButton = TransparentToolButton(FluentIcon.CLOSE, self)
        self.__initWidgets()

    def __initWidgets(self):
        self.imageLabel.setImage(self.image)

        self.closeButton.setFixedSize(32, 32)
        self.closeButton.setIconSize(QSize(12, 12))
        self.closeButton.setVisible(self.isClosable)
        self.titleLabel.setVisible(bool(self.title))
        self.contentLabel.setVisible(bool(self.content))
        self.iconWidget.setHidden(self.icon is None)

        self.closeButton.clicked.connect(self.closed)

        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')
        FluentStyleSheet.TEACHING_TIP.apply(self)

        self.__initLayout()

    def __initLayout(self):
        self.vBoxLayout.setContentsMargins(1, 1, 1, 1)
        self.widgetLayout.setContentsMargins(0, 8, 0, 8)
        self.viewLayout.setSpacing(4)
        self.widgetLayout.setSpacing(0)
        self.vBoxLayout.setSpacing(0)

        # add icon widget
        if not self.title or not self.content:
            self.iconWidget.setFixedHeight(36)

        self.vBoxLayout.addLayout(self.viewLayout)
        self.viewLayout.addWidget(self.iconWidget, 0, Qt.AlignTop)

        # add text
        self._adjustText()
        self.widgetLayout.addWidget(self.titleLabel)
        self.widgetLayout.addWidget(self.contentLabel)
        self.viewLayout.addLayout(self.widgetLayout)

        # add close button
        self.closeButton.setVisible(self.isClosable)
        self.viewLayout.addWidget(
            self.closeButton, 0, Qt.AlignRight | Qt.AlignTop)

        # adjust content margins
        margins = QMargins(6, 5, 6, 5)
        margins.setLeft(20 if not self.icon else 5)
        margins.setRight(20 if not self.isClosable else 6)
        self.viewLayout.setContentsMargins(margins)

        # add image
        self._adjustImage()
        self._addImageToLayout()

    def addWidget(self, widget: QWidget, stretch=0, align=Qt.AlignLeft):
        """ add widget to view """
        self.widgetLayout.addSpacing(8)
        self.widgetLayout.addWidget(widget, stretch, align)

    def _addImageToLayout(self):
        self.imageLabel.setBorderRadius(8, 8, 0, 0)
        self.imageLabel.setHidden(self.imageLabel.isNull())
        self.vBoxLayout.insertWidget(0, self.imageLabel)

    def _adjustText(self):
        w = min(900, QApplication.screenAt(
            QCursor.pos()).geometry().width() - 200)


        # adjust title
        chars = max(min(w / 10, 120), 30)
        self.titleLabel.setText(TextWrap.wrap(self.title, chars, False)[0])

        # adjust content
        chars = max(min(w / 9, 120), 30)
        self.contentLabel.setText(TextWrap.wrap(self.content, chars, False)[0])

    def _adjustImage(self):
        w = self.vBoxLayout.sizeHint().width() - 2
        self.imageLabel.scaledToWidth(w)

    def showEvent(self, e):
        super().showEvent(e)
        self._adjustImage()
        self.adjustSize()

class Flyout(QWidget):
    """ Flyout """

    closed = Signal()

    def __init__(self, view: FlyoutViewBase, parent=None, isDeleteOnClose=True):
        super().__init__(parent=parent)
        self.view = view
        self.hBoxLayout = QHBoxLayout(self)
        self.aniManager = None  # type: FlyoutAnimationManager
        self.isDeleteOnClose = isDeleteOnClose

        self.hBoxLayout.setContentsMargins(15, 8, 15, 20)
        self.hBoxLayout.addWidget(self.view)
        self.setShadowEffect()

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint |
                            Qt.NoDropShadowWindowHint)

    def setShadowEffect(self, blurRadius=35, offset=(0, 8)):
        """ add shadow to dialog """
        color = QColor(0, 0, 0, 80 if isDarkTheme() else 30)
        self.shadowEffect = QGraphicsDropShadowEffect(self.view)
        self.shadowEffect.setBlurRadius(blurRadius)
        self.shadowEffect.setOffset(*offset)
        self.shadowEffect.setColor(color)
        self.view.setGraphicsEffect(None)
        self.view.setGraphicsEffect(self.shadowEffect)

    def closeEvent(self, e):
        if self.isDeleteOnClose:
            self.deleteLater()

        super().closeEvent(e)
        self.closed.emit()

    def showEvent(self, e):
        # fixes #780
        self.activateWindow()
        super().showEvent(e)

    def exec(self, pos: QPoint, aniType=FlyoutAnimationType.PULL_UP):
        """ show calendar view """
        self.aniManager = FlyoutAnimationManager.make(aniType, self)
        self.show()
        self.aniManager.exec(pos)

    @classmethod
    def make(cls, view: FlyoutViewBase, target: Union[QWidget, QPoint, QFrame] = None, parent=None,
             aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        """ create and show a flyout

        Parameters
        ----------
        view: FlyoutViewBase
            flyout view

        target: QWidget | QPoint
            the target widget or position to show flyout

        parent: QWidget
            parent window

        aniType: FlyoutAnimationType
            flyout animation type

        isDeleteOnClose: bool
            whether delete flyout automatically when flyout is closed
        """
        w = cls(view, parent, isDeleteOnClose)

        if target is None:
            return w

        # show flyout first so that we can get the correct size
        w.show()

        # move flyout to the top of target
        if isinstance(target, QWidget):
            target = FlyoutAnimationManager.make(aniType, w).position(target)

        w.exec(target, aniType)
        return w

    @classmethod
    def create(cls, title: str, content: str, icon: Union[FluentIconBase, QIcon, str] = None,
               image: Union[str, QPixmap, QImage] = None, isClosable=False, target: Union[QWidget, QPoint] = None,
               parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        """ create and show a flyout using the default view

        Parameters
        ----------
        title: str
            the title of teaching tip

        content: str
            the content of teaching tip

        icon: InfoBarIcon | FluentIconBase | QIcon | str
            the icon of teaching tip

        image: str | QPixmap | QImage
            the image of teaching tip

        isClosable: bool
            whether to show the close button

        target: QWidget | QPoint
            the target widget or position to show flyout

        parent: QWidget
            parent window

        aniType: FlyoutAnimationType
            flyout animation type

        isDeleteOnClose: bool
            whether delete flyout automatically when flyout is closed
        """
        view = FlyoutView(title, content, icon, image, isClosable)
        w = cls.make(view, target, parent, aniType, isDeleteOnClose)
        view.closed.connect(w.close)
        return w

    def fadeOut(self):
        self.fadeOutAni = QPropertyAnimation(self, b'windowOpacity', self)
        self.fadeOutAni.finished.connect(self.close)
        self.fadeOutAni.setStartValue(1)
        self.fadeOutAni.setEndValue(0)
        self.fadeOutAni.setDuration(120)
        self.fadeOutAni.start()


class AcrylicFlyoutViewBase(AcrylicWidget, FlyoutViewBase):
    """ Acrylic flyout view base """

    def acrylicClipPath(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        return path

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        self._drawAcrylic(painter)

        # draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self.borderColor())
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)


class AcrylicFlyoutView(AcrylicWidget, FlyoutView):
    """ Acrylic flyout view """

    def acrylicClipPath(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        return path

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        self._drawAcrylic(painter)

        # draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self.borderColor())
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)


class AcrylicFlyout(Flyout):
    """ Acrylic flyout """

    @classmethod
    def create(cls, title: str, content: str, icon: Union[FluentIconBase, QIcon, str] = None,
               image: Union[str, QPixmap, QImage] = None, isClosable=False, target: Union[QWidget, QPoint] = None,
               parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        """ create and show a flyout using the default view

        Parameters
        ----------
        title: str
            the title of teaching tip

        content: str
            the content of teaching tip

        icon: InfoBarIcon | FluentIconBase | QIcon | str
            the icon of teaching tip

        image: str | QPixmap | QImage
            the image of teaching tip

        isClosable: bool
            whether to show the close button

        target: QWidget | QPoint
            the target widget or position to show flyout

        parent: QWidget
            parent window

        aniType: FlyoutAnimationType
            flyout animation type

        isDeleteOnClose: bool
            whether delete flyout automatically when flyout is closed
        """
        view = AcrylicFlyoutView(title, content, icon, image, isClosable)
        w = cls.make(view, target, parent, aniType, isDeleteOnClose)
        view.closed.connect(w.close)
        return w

    def exec(self, pos: QPoint, aniType=FlyoutAnimationType.PULL_UP):
        """ show calendar view """
        self.aniManager = FlyoutAnimationManager.make(aniType, self)

        if isinstance(self.view, AcrylicWidget):
            pos = self.aniManager._adjustPosition(pos)
            self.view.acrylicBrush.grabImage(QRect(pos, self.layout().sizeHint()))

        self.show()
        self.aniManager.exec(pos)


class ResultChartView(QMainWindow):
    def __init__(self, result_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Results Chart")
        self.setFixedSize(1200, 750)  # Set window size

        # Set background color to white
        self.setStyleSheet("background-color: white;")

        # Create a central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Main layout to hold chart and table side by side

        # Create a QSplitter to divide chart and table horizontally with a 6:4 ratio
        splitter = QSplitter(Qt.Horizontal)

        # Chart area
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a figure and canvas for the chart
        self.figure = Figure(figsize=(8, 12))  # Adjust figure size for horizontal bar chart
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Set the canvas as the scrollable widget
        scroll_area.setWidget(self.canvas)

        # Add toolbar and chart to the layout
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(scroll_area)

        # Add the chart widget to the splitter
        splitter.addWidget(chart_widget)

        # Table area
        table_widget_container = QWidget()
        table_layout = QVBoxLayout(table_widget_container)

        # Add a title label for the table
        table_title = QLabel("[ Result Table ]")
        table_title.setAlignment(Qt.AlignCenter)
        table_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px 0;")

        # Create the styled table widget using qfluentwidgets
        self.table_widget = TableWidget(self)
        self.table_widget.setItemDelegate(TableItemDelegate(self.table_widget))
        self.table_widget.setSelectRightClickedRow(True)
        self.table_widget.setBorderVisible(True)
        self.table_widget.setBorderRadius(8)
        self.table_widget.setRowCount(len(result_data))
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Index", "Class", "Frequency", "Percent"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.verticalHeader().setVisible(False)  # Hide row numbers

        # Populate the table with data
        self.populate_table(result_data)

        # Add the title and table to the layout
        table_layout.addWidget(table_title)
        table_layout.addWidget(self.table_widget)

        # Add the table widget container to the splitter
        splitter.addWidget(table_widget_container)

        # Set initial size ratio between chart and table
        splitter.setStretchFactor(0, 4)  # Chart area takes 4 parts
        splitter.setStretchFactor(1, 6)  # Table area takes 6 parts

        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

        # Center the window on the parent
        self.center_on_parent()
        # Plot the data
        self.plot_result_statics(result_data)

    def populate_table(self, result_data):
        total = sum(result_data.values())

        for i, (cls, freq) in enumerate(result_data.items()):
            percent = (freq / total) * 100
            index_item = QTableWidgetItem(str(i + 1))
            class_item = QTableWidgetItem(cls)
            freq_item = QTableWidgetItem(str(freq))
            percent_item = QTableWidgetItem(f"{percent:.2f}%")

            # Set color based on percentage
            if percent > 20:
                # Set font color to red
                index_item.setForeground(QColor("red"))
                class_item.setForeground(QColor("red"))
                freq_item.setForeground(QColor("red"))
                percent_item.setForeground(QColor("red"))
            else:
                # Set font color to black
                index_item.setForeground(QColor("black"))
                class_item.setForeground(QColor("black"))
                freq_item.setForeground(QColor("black"))
                percent_item.setForeground(QColor("black"))

            # Add items to table
            self.table_widget.setItem(i, 0, index_item)
            self.table_widget.setItem(i, 1, class_item)
            self.table_widget.setItem(i, 2, freq_item)
            self.table_widget.setItem(i, 3, percent_item)

    def plot_result_statics(self, result_data):
        categories = list(result_data.keys())
        values = list(result_data.values())
        total = sum(values)
        percentages = [(value / total) * 100 for value in values]

        # Clear the figure and enable constrained layout
        self.figure.clear()
        self.figure.set_constrained_layout(True)  # Enables automatic layout adjustment without extra padding

        ax = self.figure.add_subplot(111)

        # Plot horizontal bars
        y_positions = np.arange(len(categories))
        bars = ax.barh(y_positions, percentages, color='skyblue')
        ax.set_title("[ Detection Results Target Category Statistical Proportion ]",fontsize=11)
        ax.set_ylabel("Target Category")
        ax.set_xlabel("Percentage (%)")

        # Set y-ticks to custom positions with category labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories)

        # Add percentages next to or inside the bars based on bar length
        for bar, percentage in zip(bars, percentages):
            xval = bar.get_width()
            if xval > 20:  # If bar is long, place text inside the bar
                ax.text(xval - 1, bar.get_y() + bar.get_height() / 2, f"{percentage:.2f}%",
                        ha="right", va="center", color="red", fontsize=8)
            else:  # Place text outside for shorter bars
                ax.text(xval + 0.5, bar.get_y() + bar.get_height() / 2, f"{percentage:.2f}%",
                        ha="left", va="center", color="black", fontsize=8)

        # Remove any default padding by adjusting rcParams if necessary
        rcParams.update({
            'figure.autolayout': False,
            'axes.titlepad': 0,
            'axes.labelpad': 0,
        })

        # Draw the canvas
        self.canvas.draw()

    def center_on_parent(self):
        if self.parent():
            # Get parent window geometry
            parent_geometry = self.parent().frameGeometry()
            parent_center = parent_geometry.center()

            # Move the window to the center of the parent
            self.move(parent_center - self.rect().center())