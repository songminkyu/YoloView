# coding: utf-8
import os
import sys
from enum import Enum, auto

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QDialog, QFileDialog, QLineEdit, QProgressBar,
    QScrollArea, QWidget,QLabel,QRadioButton, QButtonGroup
)
from qfluentwidgets import (
    PrimaryPushButton, PushButton, CheckBox, Theme, setTheme
)

from pyiqa.archs.brisque_arch import brisque
from utils.curation.DatasetChangeClassId import DatasetChangeClassId
from utils.curation.DatasetCleaner import DatasetCleaner
from utils.curation.DatasetDistributionbalance import DatasetDistributionbalance
from utils.curation.DatasetSorting import DatasetSorting
from utils.curation.DatasetImageQualityEvaluator import ImageQualityAssessmentReorganizer

class Features(Enum):
    remove_mismatched_label_image_data = auto()
    remove_duplicate_label_image_data = auto()
    classify_zero_textsize_images = auto()
    image_evaluation_classification = auto()
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
            Features.remove_duplicate_label_image_data: "Remove duplicate image data",
            Features.classify_zero_textsize_images: "Classify or remove matching images if the label text size is 0",
            Features.image_evaluation_classification: "Image quality evaluation and classification",
            Features.remove_low_quality_images: "Remove low quality after image quality evaluation",
            Features.remove_segmentation: "Remove segmentation",
            Features.remove_bounding_box: "Remove bounding box",
            Features.remove_images_by_class_id: "Remove images and labels through class id to be removed",
            Features.change_class_id: "Change class ID",
            Features.adjust_data_split_ratio: "Adjust data split ratio",
            Features.sort_images_labels: "Change label and image file names by matching them (add padding to file name)"
        }
        return descriptions[self]

class CurationQWidget(QDialog):
    def __init__(self):
        super().__init__()

        # Style adjustments
        self.setWindowTitle("Curation")
        self.setStyleSheet("CurationQWidget{background: rgb(255, 255, 255)}")
        self.setFixedSize(640, 700)  # 창 크기 고정
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
        # feature layout 스크롤 영역 추가
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)  # 스크롤 영역 크기를 조정할 수 있도록 설정
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;  /* 테두리 제거 */
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px 0 0px 0;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #b0b0b0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                height: 0px;
                width: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        # feature_layout을 담을 위젯 생성
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("""
            QWidget {
                color: black;                
                background-color: white;                     
                border-radius: 8px; /* 내부 위젯의 모서리를 둥글게 설정 */
            }    
           
        """)

        self.feature_layout = QVBoxLayout(self.scroll_widget)
        self.checkboxes = {}

        # changed class id
        self.target_classid_edit = self.create_line_edit("Class Id to Change (ex : 0,1,2,3)")
        self.new_classid_edit = self.create_line_edit("New class id (ex : 1)")

        # image evaluation
        self.brisque_edit = self.create_line_edit("BRISQUE score (lower is better)")
        self.niqe_edit = self.create_line_edit("NIQE score (lower is better)")
        self.piqe_edit = self.create_line_edit("PIQE score (lower is better)")

        # Delete class id
        self.delete_classid_edit = self.create_line_edit("Remove class id (ex : 0,1,2,3)")

        # Ratio edits
        self.train_ratio_edit = self.create_line_edit("Train ratio (ex : 0.70)")
        self.valid_ratio_edit = self.create_line_edit("Valid ratio (ex : 0.15)")
        self.test_ratio_edit = self.create_line_edit("Test ratio (ex : 0.15)")

        self.zero_size_classify_directory_layout, self.zero_size_classify_directory_path_edit, self.zero_size_classify_select_directory = self.create_directory(
                    "If you choose classification instead of deletion, please specify the directory to classify.","zero_size_classify")
        self.zero_size_classify_directory_layout.setEnabled(False)
        self.zero_size_classify_directory_path_edit.setEnabled(False)
        self.zero_size_classify_select_directory.setContentsMargins(0, 10, 0, 10)

        self.image_evaluation_classify_directory_layout, self.image_evaluation_classify_directory_path_edit, self.image_evaluation_classify_select_directory = self.create_directory(
            "Please specify the directory to save the images classified by the image quality assessment results.", "image_evaluation_classify")
        self.image_evaluation_classify_directory_layout.setEnabled(False)
        self.image_evaluation_classify_directory_path_edit.setEnabled(False)
        self.image_evaluation_classify_select_directory.setContentsMargins(0, 15, 0, 0)

        # 체크박스 생성
        for feature in Features:
            checkbox = CheckBox(feature.description, self)
            checkbox.stateChanged.connect(lambda state, f=feature: self.checkbox_state_changed(state, f))
            self.feature_layout.addWidget(checkbox)
            self.checkboxes[feature] = checkbox

            if feature == Features.classify_zero_textsize_images:
                self.feature_layout.addLayout(self.zero_size_classify_directory_layout)

            elif feature == Features.remove_images_by_class_id:
                self.deleteid_layout = QHBoxLayout()
                self.deleteid_layout.setContentsMargins(0, 10, 0, 10)
                self.deleteid_layout.setSpacing(15)
                self.deleteid_layout.addWidget(self.delete_classid_edit)
                self.feature_layout.addLayout(self.deleteid_layout)

            elif feature == Features.adjust_data_split_ratio:
                self.ratio_layout = QHBoxLayout()
                self.ratio_layout.setContentsMargins(0, 10, 0, 10)
                self.ratio_layout.setSpacing(15)
                self.ratio_layout.addWidget(self.train_ratio_edit)
                self.ratio_layout.addWidget(self.valid_ratio_edit)
                self.ratio_layout.addWidget(self.test_ratio_edit)
                self.feature_layout.addLayout(self.ratio_layout)

            elif feature == Features.image_evaluation_classification:
                self.image_evaluation_group = QButtonGroup(self)
                self.feature_layout.addLayout(self.image_evaluation_classify_directory_layout)

                self.image_evaluation_layout = QVBoxLayout()
                self.image_evaluation_layout.setContentsMargins(0, 10, 0, 10)
                self.image_evaluation_layout.setSpacing(15)

                # BRISQUE
                self.brisque_radiobutton = QRadioButton("BRISQUE", self)
                self.brisque_radiobutton.setEnabled(False)
                self.brisque_label = QLabel(
                    "BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) : Assesses image "
                    "quality without a reference image. Lower values indicate better quality.",
                    self)
                self.brisque_label.setWordWrap(True)
                self.brisque_line_layout = QVBoxLayout()
                self.brisque_line_layout.addWidget(self.brisque_radiobutton)
                self.brisque_line_layout.addWidget(self.brisque_label)
                self.brisque_line_layout.addWidget(self.brisque_edit)
                self.image_evaluation_layout.addLayout(self.brisque_line_layout)

                # NIQE
                self.niqe_radiobutton = QRadioButton("NIQE", self)
                self.niqe_radiobutton.setEnabled(False)
                self.niqe_label = QLabel(
                    "NIQE (Natural Image Quality Evaluator) : A no-reference image quality measurement metric based "
                    "on statistical features from natural scene statistics (NSS). Lower values indicate better image quality.",
                    self
                )
                self.niqe_label.setWordWrap(True)
                self.niqe_line_layout = QVBoxLayout()
                self.niqe_line_layout.addWidget(self.niqe_radiobutton)
                self.niqe_line_layout.addWidget(self.niqe_label)
                self.niqe_line_layout.addWidget(self.niqe_edit)
                self.image_evaluation_layout.addLayout(self.niqe_line_layout)

                # PIQE
                self.piqe_radiobutton = QRadioButton("PIQE", self)
                self.piqe_radiobutton.setEnabled(False)
                self.piqe_label = QLabel(
                    "PIQE (Perception-based Image Quality Evaluator) : A no-reference image quality assessment metric that evaluates "
                    "the quality of an image based on perceptual criteria. Lower values indicate better image quality.",
                    self
                )
                self.piqe_label.setWordWrap(True)
                self.piqe_line_layout = QVBoxLayout()
                self.piqe_line_layout.addWidget(self.piqe_radiobutton)
                self.piqe_line_layout.addWidget(self.piqe_label)
                self.piqe_line_layout.addWidget(self.piqe_edit)
                self.image_evaluation_layout.addLayout(self.piqe_line_layout)

                self.image_evaluation_group.addButton(self.brisque_radiobutton)
                self.image_evaluation_group.addButton(self.niqe_radiobutton)
                self.image_evaluation_group.addButton(self.piqe_radiobutton)

                self.feature_layout.addLayout(self.image_evaluation_layout)

            elif feature == Features.change_class_id:
                self.changeid_layout = QHBoxLayout()
                self.changeid_layout.setContentsMargins(0, 10, 0, 10)
                self.changeid_layout.setSpacing(15)
                self.changeid_layout.addWidget(self.target_classid_edit)
                self.changeid_layout.addWidget(self.new_classid_edit)
                self.feature_layout.addLayout(self.changeid_layout)

        # scroll_widget을 scroll_area에 추가
        self.scroll_area.setWidget(self.scroll_widget)

        # 기존 mainLayout에 scroll_area 추가
        self.mainLayout.addWidget(self.scroll_area)

        self.progress_bar_layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName(u"progress_bar")
        self.progress_bar.setMinimumSize(QSize(0, 20))
        self.progress_bar.setMaximumSize(QSize(16777215, 20))
        self.progress_bar.setStyleSheet(u"QProgressBar{ \n"
                                        "font: 700 10pt \"Nirmala UI\";\n"
                                        "color: #8EC5FC; \n"
                                        "text-align:center; \n"
                                        "border:3px solid rgb(255, 255, 255);\n"
                                        "border-radius: 7px; \n"
                                        "background-color: rgba(215, 215, 215,100);\n"
                                        "} \n"
                                        "\n"
                                        "QProgressBar:chunk{ \n"
                                        "border-radius:0px; \n"
                                        "background:  #00A7B3;\n"
                                        "border-radius: 5px;\n"
                                        "}")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar_layout.addSpacing(20)
        self.progress_bar_layout.addWidget(self.progress_bar)

        self.mainLayout.addLayout(self.progress_bar_layout)

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

    # Layout에 포함된 모든 위젯을 숨기는 함수
    def hide_layout_widgets(self, layout: QVBoxLayout, hide: bool):
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setVisible(not hide)

    def checkbox_state_changed(self, state, feature):
        if self.checkboxes[Features.classify_zero_textsize_images].isChecked():
            self.zero_size_classify_directory_path_edit.setEnabled(True)
            self.zero_size_classify_select_directory.setEnabled(True)
        else:
            self.zero_size_classify_directory_path_edit.setEnabled(False)
            self.zero_size_classify_select_directory.setEnabled(False)

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

        if self.checkboxes[Features.image_evaluation_classification].isChecked():
            self.image_evaluation_classify_directory_path_edit.setEnabled(True)
            self.image_evaluation_classify_select_directory.setEnabled(True)

            self.brisque_radiobutton.setEnabled(True)
            self.brisque_radiobutton.setChecked(True)
            self.brisque_edit.setEnabled(True)

            self.niqe_radiobutton.setEnabled(True)
            self.niqe_edit.setEnabled(True)

            self.piqe_radiobutton.setEnabled(True)
            self.piqe_edit.setEnabled(True)

        else:
            self.image_evaluation_classify_directory_path_edit.setEnabled(False)
            self.image_evaluation_classify_select_directory.setEnabled(False)

            self.brisque_radiobutton.setEnabled(False)
            self.brisque_radiobutton.setChecked(False)
            self.brisque_edit.setEnabled(False)

            self.niqe_radiobutton.setEnabled(False)
            self.niqe_edit.setEnabled(False)

            self.piqe_radiobutton.setEnabled(False)
            self.piqe_edit.setEnabled(False)

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
        directory_layout.setContentsMargins(0, 10, 0, 10)

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
        if type == "curation":
            select_directory_button.setEnabled(True)
        else:
            select_directory_button.setEnabled(False)

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

        if type == "zero_size_classify" and self.checkboxes[Features.classify_zero_textsize_images].isChecked():
            self.zero_size_classify_directory_path_edit.setText(selected_directory)
        if type == "image_evaluation_classify" and self.checkboxes[Features.image_evaluation_classification].isChecked():
            self.image_evaluation_classify_directory_path_edit.setText(selected_directory)


    def proceed_action(self):

        threshold = 0.0
        metric_name = ""

        if self.brisque_radiobutton.isChecked():
            metric_name = "BRISQUE"
            threshold = float(self.brisque_edit.text()) if self.brisque_edit.text() != "0" else 50.0
        elif self.niqe_radiobutton.isChecked():
            metric_name = "NIQE"
            threshold = float(self.niqe_edit.text()) if self.niqe_edit.text() != "0" else 5.0
        else:
            metric_name = "PIQE"
            threshold = float(self.piqe_edit.text()) if self.piqe_edit.text() != "0" else 50.0

        train_ratio = float(self.train_ratio_edit.text() if self.train_ratio_edit.isEnabled() else 0.7)
        valid_ratio = float(self.valid_ratio_edit.text() if self.valid_ratio_edit.isEnabled() else 0.15)
        test_ratio = float(self.test_ratio_edit.text() if self.test_ratio_edit.isEnabled() else 0.15)

        current_path = self.curation_directory_path_edit.text()
        classify_path = self.zero_size_classify_directory_path_edit.text()
        img_eval_path = self.image_evaluation_classify_directory_path_edit.text()

        remove_class_id = list(
            map(int, self.delete_classid_edit.text().split(','))) if self.delete_classid_edit.text() else []

        change_target_classid = list(
            map(int, self.target_classid_edit.text().split(','))) if self.target_classid_edit.text() else []

        change_new_classid = int(self.new_classid_edit.text()) if self.new_classid_edit.text() else None

        if not current_path or not os.path.exists(current_path):
            return  # Exit the method if the directory path is empty or does not exist

        # 선택된 체크박스 개수 계산
        selected_features = [feature for feature, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        total_checked = len(selected_features)
        print(f"체크된 기능 수: {total_checked}")

        # 프로그레스바 초기화
        self.progress_bar.setValue(0)

        # 각 기능 수행 시마다 프로그레스 바 업데이트
        processed = 0

        subfolders = ['train', 'valid', 'test']

        is_delete = True
        if not current_path or not os.path.exists(classify_path):
            is_delete = False

        image_quality = None
        dataset_sorting = DatasetSorting(current_path, subfolders)
        dataset_balance = DatasetDistributionbalance(current_path,train_ratio,test_ratio,valid_ratio)
        cleaner = DatasetCleaner(current_path, subfolders, is_delete, classify_path)
        change_class_id = DatasetChangeClassId(current_path, subfolders, change_target_classid, change_new_classid)

        for feature in selected_features:
            # 여기서 실제 기능 처리 로직을 수행:
            if feature.name == Features.remove_mismatched_label_image_data.name:
                cleaner.remove_mismatch_images_and_labels()
            if feature.name == Features.remove_duplicate_label_image_data.name:
                cleaner.remove_duplicate_images()
            if feature.name == Features.classify_zero_textsize_images.name:
                cleaner.remove_zero_label()
            if feature.name == Features.remove_segmentation.name:
                cleaner.remove_segments()
            if feature.name == Features.remove_bounding_box.name:
                cleaner.remove_bounding_boxes()
            if feature.name == Features.remove_images_by_class_id.name:
                cleaner.remove_labels_and_images_by_class_ids(remove_class_id)
            if feature.name == Features.change_class_id.name:
                change_class_id.update_class_id_processing()
            if feature.name == Features.adjust_data_split_ratio.name:
                dataset_balance.adjust_dataset_splits()
            if feature.name == Features.sort_images_labels:
                dataset_sorting.sort_files_to_match_processing()
            if feature.name == Features.image_evaluation_classification.name:
                image_quality = ImageQualityAssessmentReorganizer(current_path, img_eval_path, metric_name, threshold)
                image_quality.move_files_by_metric()

            # 기능 완료 후 프로그레스바 업데이트
            processed += 1
            progress_value = int((processed / total_checked) * 100)
            self.progress_bar.setValue(progress_value)
            QApplication.processEvents()  # UI 갱신

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
