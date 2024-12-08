# coding: utf-8
import sys
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QDialog, QFileDialog, QLineEdit
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
        self.setFixedSize(600, 500)  # 창 크기 고정

        # Main layout
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setAlignment(Qt.AlignTop)  # 모든 컨트롤을 위쪽으로 정렬
        self.mainLayout.setSpacing(5)  # 컨트롤 간의 간격 조정
        self.mainLayout.setContentsMargins(20, 20, 20, 20)  # 여백 조정

        # Add directory path selection section with top margin
        self.directory_layout = QHBoxLayout()
        self.directory_layout.setContentsMargins(0, 0, 0, 20)  # 하단 마진 추가
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

        # Add qfluentwidgets styled checkboxes for features
        self.feature_layout = QVBoxLayout()
        self.feature_layout.setSpacing(15)  # 체크박스 간 간격 조정
        features = [
            "라벨 및 이미지 이름이 매칭 안되는 데이터 제거",  # DatasetCleaner.py
            "라벨 텍스트 사이즈 0인 경우 매칭되는 이미지와 같이 제거",
            "라벨 및 이미지 파일 이름을 매칭하여 변경 (파일명 패딩 추가)",
            "이미지 품질 평가 후 낮은 품질 제거",  # DatasetImageQualityEvaluator.py
            "Segmentation 제거",  # DatasetCleaner.py
            "Bounding Box 제거",  # DatasetCleaner.py
            "제거할 클래스 id를 통해서 이미지 및 라벨 제거",
            "클래스 ID 변경",  # DatasetChangeClassId.py
            "데이터 분할 비율 조정",  # DatasetDistributionbalance.py
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
        self.button_layout.setContentsMargins(0, 30, 0, 0)  # 버튼 위쪽 마진 추가
        self.button_layout.setSpacing(10)
        self.button_layout.setAlignment(Qt.AlignRight)  # 버튼을 오른쪽으로 정렬

        self.proceed_button = PrimaryPushButton("Progress", self)
        self.proceed_button.clicked.connect(self.proceed_action)

        self.cancel_button = PushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_action)

        self.button_layout.addWidget(self.proceed_button)
        self.button_layout.addWidget(self.cancel_button)

        self.mainLayout.addLayout(self.button_layout)

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
