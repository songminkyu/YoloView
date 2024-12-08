# coding: utf-8
import sys
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QDialog, QLineEdit, QFileDialog
)
from PySide6.QtCore import Qt
from qfluentwidgets import (
    PrimaryPushButton, PushButton, CheckBox, Theme, setTheme
)


class CurationQWidget(QDialog):
    def __init__(self):
        super().__init__()

        # Style adjustments
        self.setWindowTitle("Curation")
        self.setStyleSheet("CurationQWidget{background: rgb(255, 255, 255)}")
        self.setFixedSize(600, 450)

        # Main layout
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setAlignment(Qt.AlignTop)  # Align all elements to the top
        self.mainLayout.setSpacing(5)  # Adjust spacing between elements
        self.mainLayout.setContentsMargins(20, 20, 20, 20)  # Adjust margins

        # Add directory path selection section with top margin
        self.directory_layout = QHBoxLayout()
        self.directory_layout.setContentsMargins(0, 0, 0, 20)  # Add top margin to move it down
        self.directory_path_edit = QLineEdit(self)
        self.directory_path_edit.setReadOnly(True)  # Disable editing of the text box
        self.directory_path_edit.setPlaceholderText("The selected directory path will be displayed here.")
        self.directory_path_edit.setFixedHeight(30)

        self.select_directory_button = PrimaryPushButton("...", self)
        self.select_directory_button.setFixedHeight(30)
        self.select_directory_button.clicked.connect(self.open_directory_dialog)

        self.directory_layout.addWidget(self.directory_path_edit)
        self.directory_layout.addWidget(self.select_directory_button)

        # Add directory layout at the bottom
        self.mainLayout.addLayout(self.directory_layout)

        # Add qfluentwidgets styled checkboxes for features
        self.feature_layout = QVBoxLayout()
        self.feature_layout.setSpacing(10)  # Reduce spacing between checkboxes
        features = [
            "데이터 정리 및 중복 제거",  # DatasetCleaner.py
            "Segmentation 제거",  # DatasetCleaner.py
            "Bounding Box 제거",  # DatasetCleaner.py
            "클래스 ID 변경",  # DatasetChangeClassId.py
            "데이터 분할 비율 조정",  # DatasetDistributionbalance.py
            "이미지 품질 평가",  # DatasetImageQualityEvaluator.py
            "이미지 및 라벨 정렬",  # DatasetSorting.py
        ]
        self.checkboxes = []

        for feature in features:
            checkbox = CheckBox(feature, self)
            self.feature_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        self.mainLayout.addLayout(self.feature_layout)

        # Add bottom button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 120, 0, 0)  # Add some top margin to separate from other elements
        self.button_layout.setSpacing(10)
        self.button_layout.setAlignment(Qt.AlignRight)  # Align buttons to the right

        self.proceed_button = PrimaryPushButton("Progress", self)
        self.proceed_button.clicked.connect(self.proceed_action)

        self.cancel_button = PushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_action)

        self.button_layout.addWidget(self.proceed_button)
        self.button_layout.addWidget(self.cancel_button)

        self.mainLayout.addLayout(self.button_layout)

        # Add spacer to push the directory layout to the bottom
        #self.mainLayout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def open_directory_dialog(self):
        """Open a dialog to select a directory and display the selected path."""
        selected_directory = QFileDialog.getExistingDirectory(self, "Select directory", "")
        if selected_directory:
            self.directory_path_edit.setText(selected_directory)

    def proceed_action(self):
        """Handle the proceed button click."""
        print("진행 버튼 클릭됨!")

        # 모든 체크박스의 상태를 확인하고 출력
        for idx, checkbox in enumerate(self.checkboxes):
            state = checkbox.isChecked()  # 체크박스의 상태 확인 (True/False)
            feature_name = checkbox.text()  # 체크박스의 라벨
            print(f"[{idx + 1}] {feature_name} 선택됨: {state}")

    def cancel_action(self):
        """Handle the cancel button click."""
        print("취소 버튼 클릭됨!")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply the qfluentwidgets theme
    setTheme(Theme.LIGHT)

    w = CurationQWidget()
    w.show()
    app.exec()
