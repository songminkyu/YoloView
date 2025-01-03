from PySide6.QtCore import QUrl
from PySide6.QtGui import QFont
from qfluentwidgets import MessageBoxBase, SubtitleLabel, LineEdit, PushButton, setTheme, Theme

class RtspInputMessageBox(MessageBoxBase):
    """ Custom message box """

    def __init__(self, parent=None, mode=None):
        super().__init__(parent)
        self.urlLineEdit = LineEdit(self)
        if mode == "single":
            self.titleLabel = SubtitleLabel('Input Rtsp/Http/Https URL', self)
            self.urlLineEdit.setPlaceholderText('rtsp:// - http:// - https://')
        else:
            self.titleLabel = SubtitleLabel('Input Rtsp URL', self)
            self.urlLineEdit.setPlaceholderText('rtsp://')

        self.urlLineEdit.setFont(QFont("Segoe UI", 14))
        self.urlLineEdit.setClearButtonEnabled(True)

        # add widget to view layout
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.urlLineEdit)

        # Add user authentication checkbox
        self.authCheckBox = CheckBox('Enable User Authentication', self)
        self.authCheckBox.setFont(QFont("Segoe UI", 12))
        self.authCheckBox.stateChanged.connect(self._toggleAuthFields)
        self.viewLayout.addWidget(self.authCheckBox)

        # Add username and password fields (initially hidden)
        self.usernameLineEdit = LineEdit(self)
        self.usernameLineEdit.setPlaceholderText('Username')
        self.usernameLineEdit.setFont(QFont("Segoe UI", 14))
        self.usernameLineEdit.setClearButtonEnabled(True)
        self.usernameLineEdit.setVisible(False)

        self.passwordLineEdit = LineEdit(self)
        self.passwordLineEdit.setPlaceholderText('Password')
        self.passwordLineEdit.setFont(QFont("Segoe UI", 14))
        self.passwordLineEdit.setClearButtonEnabled(True)
        self.passwordLineEdit.setEchoMode(LineEdit.Password)
        self.passwordLineEdit.setVisible(False)

        # Add optional fields to layout
        self.viewLayout.addWidget(self.usernameLineEdit)
        self.viewLayout.addWidget(self.passwordLineEdit)

        # change the text of button
        self.yesButton.setText('Confirm')
        self.cancelButton.setText('Cancel')
        self.yesButton.clicked.connect(self.getValues)

        self.widget.setMinimumWidth(400)
        self.yesButton.setDisabled(True)
        self.urlLineEdit.textChanged.connect(self._validateInputs)

    def _toggleAuthFields(self, state):
        """Toggle visibility and requirement of username and password fields."""
        is_checked = state == Qt.Checked.value
        self.usernameLineEdit.setVisible(is_checked)
        self.passwordLineEdit.setVisible(is_checked)
        if is_checked:
            self.usernameLineEdit.textChanged.connect(self._validateInputs)
            self.passwordLineEdit.textChanged.connect(self._validateInputs)
        else:
            self.usernameLineEdit.textChanged.disconnect()
            self.passwordLineEdit.textChanged.disconnect()
            self.usernameLineEdit.clear()
            self.passwordLineEdit.clear()
        self._validateInputs()

    def _validateInputs(self):
        """Validate URL and optionally username/password."""
        url_valid = QUrl(self.urlLineEdit.text()).isValid()
        auth_valid = True
        if self.authCheckBox.isChecked():
            auth_valid = bool(self.usernameLineEdit.text()) and bool(self.passwordLineEdit.text())
        self.yesButton.setEnabled(url_valid and auth_valid)

    def _validateUrl(self, text):
        self.yesButton.setEnabled(QUrl(text).isValid())

    def getValues(self):
        """Get the entered values."""
        url = self.urlLineEdit.text()
        username = self.usernameLineEdit.text() if self.authCheckBox.isChecked() else None
        password = self.passwordLineEdit.text() if self.authCheckBox.isChecked() else None
        if username and password:
            print(f"URL: {url}, Username: {username}, Password: {password}")
        else:
            print(f"URL: {url}")
