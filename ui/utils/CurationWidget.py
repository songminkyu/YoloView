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
from functools import partial
from sympy import false


class Features(Enum):
    remove_mismatched_label_image_data = auto()
    classify_zero_textsize_images = auto()
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
            Features.classify_zero_textsize_images: "Classify or remove matching images if the label text size is 0",
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
        self.setFixedSize(600, 600)  # 창 크기 고정

        # Main layout
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setAlignment(Qt.AlignTop)  # 모든 컨트롤을 위쪽으로 정렬
        self.mainLayout.setSpacing(5)  # 컨트롤 간의 간격 조정
        self.mainLayout.setContentsMargins(20, 20, 20, 20)  # 여백 조정

        # Please specify the directory to be curated
        (self.curation_directory_layout, self.curation_directory_path_edit,
         self.curation_select_directory) = self.create_directory("The selected directory path will be displayed here.","curation")

        self.mainLayout.addLayout(self.curation_directory_layout)

        # feature layout
        self.feature_layout = QVBoxLayout()

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

        self.classify_directory_layout, self.classify_directory_path_edit, self.classify_select_directory = self.create_directory(
                    "If you prefer classification over deletion, please specify the directory to classify.","classify")
        self.classify_directory_path_edit.setEnabled(False)
        self.classify_select_directory.setEnabled(False)
        self.classify_directory_layout.setContentsMargins(0, 1, 0, 20)

        # 체크박스 생성
        for feature in Features:
            checkbox = CheckBox(feature.description, self)
            checkbox.stateChanged.connect(lambda state, f=feature: self.checkbox_state_changed(state, f))
            self.feature_layout.addWidget(checkbox)
            self.checkboxes[feature] = checkbox

            if feature == Features.classify_zero_textsize_images:
                self.feature_layout.addLayout(self.classify_directory_layout)

            elif feature == Features.remove_images_by_class_id:
                self.deleteid_layout = QHBoxLayout()
                self.deleteid_layout.setContentsMargins(0, 1, 0, 20)
                self.deleteid_layout.setSpacing(15)
                self.deleteid_layout.addWidget(self.delete_classid_edit)
                self.feature_layout.addLayout(self.deleteid_layout)

            elif feature == Features.adjust_data_split_ratio:
                self.ratio_layout = QHBoxLayout()
                self.ratio_layout.setContentsMargins(0, 1, 0, 20)
                self.ratio_layout.setSpacing(15)
                self.ratio_layout.addWidget(self.train_ratio_edit)
                self.ratio_layout.addWidget(self.valid_ratio_edit)
                self.ratio_layout.addWidget(self.test_ratio_edit)
                self.feature_layout.addLayout(self.ratio_layout)

            elif feature == Features.change_class_id:
                self.changeid_layout = QHBoxLayout()
                self.changeid_layout.setContentsMargins(0, 1, 0, 20)
                self.changeid_layout.setSpacing(15)
                self.changeid_layout.addWidget(self.target_classid_edit)
                self.changeid_layout.addWidget(self.new_classid_edit)
                self.feature_layout.addLayout(self.changeid_layout)

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

    def checkbox_state_changed(self, state, feature):
        if self.checkboxes[Features.classify_zero_textsize_images].isChecked():
            self.classify_directory_path_edit.setEnabled(True)
            self.classify_select_directory.setEnabled(True)
        else:
            self.classify_directory_path_edit.setEnabled(False)
            self.classify_select_directory.setEnabled(False)

        if self.checkboxes[Features.remove_images_by_class_id].isChecked():
            self.delete_classid_edit.setEnabled(True)
        else:
            self.delete_classid_edit.setEnabled(False)

        if self.checkboxes[Features.adjust_data_split_ratio].isChecked():
            self.train_ratio_edit.setEnabled(True)
            self.valid_ratio_edit.setEnabled(True)
            self.test_ratio_edit.setEnabled(True)

        else:
            self.train_ratio_edit.setEnabled(False)
            self.valid_ratio_edit.setEnabled(False)
            self.test_ratio_edit.setEnabled(False)

        if self.checkboxes[Features.change_class_id].isChecked():
            self.target_classid_edit.setEnabled(True)
            self.new_classid_edit.setEnabled(True)
        else:
            self.target_classid_edit.setEnabled(False)
            self.new_classid_edit.setEnabled(False)


    def create_line_edit(self, placeholder, validator=None):
        line_edit = QLineEdit(self)
        line_edit.setFixedHeight(30)
        line_edit.setEnabled(False)
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

    def create_directory(self, placeholder, type, validator=None):
        directory_layout = QHBoxLayout()
        directory_layout.setContentsMargins(0, 0, 0, 15)

        directory_path_edit = QLineEdit(self)
        directory_path_edit.setReadOnly(True)
        directory_path_edit.setPlaceholderText(placeholder)
        directory_path_edit.setFixedHeight(30)
        directory_path_edit.setStyleSheet("""
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
            directory_path_edit.setValidator(validator)

        select_directory_button = PrimaryPushButton("...", self)
        select_directory_button.setFixedHeight(30)

        # 방법1: functools.partial 활용
        select_directory_button.clicked.connect(lambda :  self.open_directory_dialog(type))

        directory_layout.addWidget(directory_path_edit)
        directory_layout.addWidget(select_directory_button)

        return directory_layout, directory_path_edit, select_directory_button

    def open_directory_dialog(self, type):
        """Open a dialog to select a directory and display the selected path."""
        selected_directory = QFileDialog.getExistingDirectory(self, "Select directory", "")
        if type == "curation" and selected_directory:
            # separator를 활용한 처리
            self.curation_directory_path_edit.setText(selected_directory)

        if type == "classify" and self.checkboxes[Features.classify_zero_textsize_images].isChecked():
            self.classify_directory_path_edit.setText(selected_directory)


    def proceed_action(self):
        """Handle the proceed button click."""
        print("진행 버튼 클릭됨!")
        current_path = self.curation_directory_path_edit.text()
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
