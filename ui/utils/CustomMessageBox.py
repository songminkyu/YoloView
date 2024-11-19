from qfluentwidgets import MessageBoxBase,SubtitleLabel,BodyLabel, StrongBodyLabel

class MessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, parent=None, topmessage=None, bottommessage=None):
        super().__init__(parent)
        self.topMessage = SubtitleLabel(topmessage, self)
        self.topMessage.pixelFontSize = 20  # 여기서 폰트 크기를 변경

        if bottommessage is not None:
            self.bottomMessage = BodyLabel(bottommessage, self)
            self.bottomMessage.pixelFontSize = 16

        # add widget to view layout
        self.viewLayout.addWidget(self.topMessage)

        if bottommessage is not None:
            self.viewLayout.addWidget(self.bottomMessage)

        self.widget.setMinimumWidth(400)
