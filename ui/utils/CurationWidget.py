# coding: utf-8
import sys
from enum import Enum, auto
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QDialog, QFileDialog, QLineEdit, QLabel
)
from PySide6.QtCore import Qt
from qfluentwidgets import (
    PrimaryPushButton, PushButton, CheckBox, Theme, setTheme
)

class Features(Enum):
    remove_mismatched_label_image_data = auto()
    remove_zero_textsize_images = auto()
    change_label_image_filenames = auto()
    remove_low_quality_images = auto()
    remove_segmentation = auto()
    remove_bounding_box = auto()
    remove_images_by_class_id = auto()
    change_class_id = auto()
    adjust_data_split_ratio = auto()
    sort_images_labels = auto()

    @property
    def description(self):
        descriptions = {
            Features.remove_mismatched_label_image_data: "Remove data with mismatched label and image names",
            Features.remove_zero_textsize_images: "Remove matching images if label text size is 0",
            Features.change_label_image_filenames: "Change label and image file names by matching them (add padding to file name)",
            Features.remove_low_quality_images: "Remove low quality after image quality evaluation",
            Features.remove_segmentation: "Remove segmentation",
            Features.remove_bounding_box: "Remove bounding box",
            Features.remove_images_by_class_id: "Remove images and labels through class id to be removed",
            Features.change_class_id: "Change class ID",
            Features.adjust_data_split_ratio: "Adjust data split ratio",
            Features.sort_images_labels: "Sort images and labels"
        }
        return descriptions[self]

class CurationQWidget(QDialog):
    def __init__(self):
        super().__init__()

        # Style adjustments
        self.setWindowTitle("Curation")
        self.setStyleSheet("CurationQWidget{background: rgb(255, 255, 255)}")
        self.setFixedSize(600, 580)  # 창 크기 고정

        # Main layout
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setAlignment(Qt.AlignTop)  # 모든 컨트롤을 위쪽으로 정렬
        self.mainLayout.setSpacing(5)  # 컨트롤 간의 간격 조정
        self.mainLayout.setContentsMargins(20, 20, 20, 20)  # 여백 조정

        # Add directory path selection section with top margin
        self.directory_layout = QHBoxLayout()
        self.directory_layout.setContentsMargins(0, 0, 0, 15)  # 하단 마진 추가
        self.directory_path_edit = QLineEdit(self)
        self.directory_path_edit.setReadOnly(True)  # 텍스트 박스를 읽기 전용으로 설정
        self.directory_path_edit.setPlaceholderText("The selected directory path will be displayed here.")
        self.directory_path_edit.setFixedHeight(30)
        self.directory_path_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                font: 600 9pt "Segoe UI";
                border: 1px solid #c0c0c0;  /* 기본 테두리 */
                border-radius: 4px;
            }
            QLineEdit:focus {
                border: 1px solid #009faa;  /* 포커스 시 동일한 두께의 색상만 변경 */
            }
        """)
        self.select_directory_button = PrimaryPushButton("...", self)
        self.select_directory_button.setFixedHeight(30)
        self.select_directory_button.clicked.connect(self.open_directory_dialog)

        self.directory_layout.addWidget(self.directory_path_edit)
        self.directory_layout.addWidget(self.select_directory_button)

        # Add directory layout at the top
        self.mainLayout.addLayout(self.directory_layout)

        # feature layout
        self.feature_layout = QVBoxLayout()
        self.feature_layout.setSpacing(15)

        self.checkboxes = {}

        # changed class id
        self.target_classid_edit = self.create_line_edit("Class Id to Change (ex : 0,1,2,3)")
        self.new_classid_edit = self.create_line_edit("New class id (ex : 1)")

        # Delete class id
        self.delete_classid_edit = self.create_line_edit("Remove class id (ex : 0,1,2,3)")

        # Ratio edits
        self.train_ratio_edit = self.create_line_edit("Train ratio (ex : 0.70)")
        self.valid_ratio_edit = self.create_line_edit("Valid ratio (ex : 0.15)")
        self.test_ratio_edit = self.create_line_edit("Test ratio (ex : 0.15)")


        # 체크박스 생성
        for feature in Features:
            checkbox = CheckBox(feature.description, self)
            self.feature_layout.addWidget(checkbox)
            self.checkboxes[feature] = checkbox

            if feature == Features.remove_images_by_class_id:
                self.feature_layout.addSpacing(5)
                self.deleteid_layout = QHBoxLayout()
                self.deleteid_layout.addWidget(self.delete_classid_edit)
                self.feature_layout.addLayout(self.deleteid_layout)
                self.feature_layout.addSpacing(15)

            elif feature == Features.adjust_data_split_ratio:
                self.feature_layout.addSpacing(5)
                self.ratio_layout = QHBoxLayout()
                self.ratio_layout.addWidget(self.train_ratio_edit)
                self.ratio_layout.addWidget(self.valid_ratio_edit)
                self.ratio_layout.addWidget(self.test_ratio_edit)
                self.feature_layout.addLayout(self.ratio_layout)
                self.feature_layout.addSpacing(15)


            elif feature == Features.change_class_id:
                self.feature_layout.addSpacing(5)
                self.changeid_layout = QHBoxLayout()
                self.changeid_layout.addWidget(self.target_classid_edit)
                self.changeid_layout.addWidget(self.new_classid_edit)
                self.feature_layout.addLayout(self.changeid_layout)
                self.feature_layout.addSpacing(15)


        self.mainLayout.addLayout(self.feature_layout)

        # Add bottom button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 50, 0, 0)  # 버튼 위쪽 마진 추가
        self.button_layout.setSpacing(10)
        self.button_layout.setAlignment(Qt.AlignRight)  # 버튼을 오른쪽으로 정렬

        self.proceed_button = PrimaryPushButton("Progress", self)
        self.proceed_button.clicked.connect(self.proceed_action)

        self.cancel_button = PushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_action)

        self.button_layout.addWidget(self.proceed_button)
        self.button_layout.addWidget(self.cancel_button)

        self.mainLayout.addLayout(self.button_layout)

    def create_line_edit(self, placeholder, validator=None):
        line_edit = QLineEdit(self)
        line_edit.setFixedHeight(30)
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet("""
               QLineEdit {
                   padding: 5px;
                   font: 600 9pt "Segoe UI";
                   border: 1px solid #c0c0c0;
                   border-radius: 4px;
               }
               QLineEdit:focus {
                   border: 1px solid #009faa;
               }
           """)
        if validator:
            line_edit.setValidator(validator)
        return line_edit

    def open_directory_dialog(self):
        """Open a dialog to select a directory and display the selected path."""
        selected_directory = QFileDialog.getExistingDirectory(self, "Select directory", "")
        if selected_directory:
            self.directory_path_edit.setText(selected_directory)

    def proceed_action(self):
        """Handle the proceed button click."""
        print("진행 버튼 클릭됨!")
        current_path = self.directory_path_edit.text()
        print(f"현재 경로: {current_path}")

        # Enum을 통한 체크박스 상태 확인
        for feature, checkbox in self.checkboxes.items():
            state = checkbox.isChecked()
            print(f"{feature.name} ( {feature.description} ) 선택됨: {state}")

        # Adjust data split ratio 체크 시
        if self.checkboxes[Features.adjust_data_split_ratio].isChecked():
            train_ratio = self.train_ratio_edit.text()
            valid_ratio = self.valid_ratio_edit.text()
            test_ratio = self.test_ratio_edit.text()
            print(f"Train ratio: {train_ratio}, Valid ratio: {valid_ratio}, Test ratio: {test_ratio}")

    def cancel_action(self):
        """Handle the cancel button click."""
        print("취소 버튼 클릭됨!")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    w = CurationQWidget()
    w.show()
    app.exec()
