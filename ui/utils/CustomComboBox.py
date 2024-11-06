from PySide6.QtCore import (QPoint, Qt)
from PySide6.QtWidgets import (QVBoxLayout,QWidget,QScrollArea,QCheckBox)
from qfluentwidgets import ComboBox

class MultiSelectComboBox(ComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Placeholder text for the ComboBox
        self.addItem("Select categories...")

        self.selected_items = list()

        # Container for checkboxes with scrolling capability
        self.checkbox_container = QWidget(self)
        self.scroll_area = QScrollArea(self.checkbox_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(200)  # Set fixed height for scrolling

        # Inner widget to hold checkboxes inside the scroll area
        self.inner_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.inner_widget)

        # Add inner widget to the scroll area
        self.scroll_area.setWidget(self.inner_widget)

        # Main layout for the checkbox container
        layout = QVBoxLayout(self.checkbox_container)
        layout.addWidget(self.scroll_area)

        # Apply custom styling to the checkbox container
        self.checkbox_container.setStyleSheet("""
                  QWidget {
                      background-color: white;                     
                      border-radius: 8px;
                  }
                  QCheckBox {
                      padding: 3px;
                      font: 600 9pt "Segoe UI";
                  }
                  QCheckBox::indicator {
                      width: 14px;
                      height: 14px;
                  }
                  QCheckBox::indicator:unchecked {
                      border: 1px solid #c0c0c0;
                      background-color: #f9f9f9;
                      border-radius: 3px;
                  }
                  QCheckBox::indicator:checked {
                      border: 1px solid #009faa;
                      background-color: #009faa;
                      border-radius: 3px;
                  }
                  QScrollBar:vertical {
                      border: none;
                      background: #f0f0f0;
                      width: 10px;
                      margin: 5px 0 5px 0;
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

        # Configure the checkbox container as a floating widget
        self.checkbox_container.setWindowFlags(Qt.Popup)
        self.checkbox_container.setFixedWidth(self.width())  # Match the ComboBox width

    def addCategory(self, categories):
        for category in categories:
            checkbox = QCheckBox(category, self.inner_widget)
            checkbox.stateChanged.connect(lambda state, chk=checkbox: self.on_checkbox_state_changed(chk))

            # If the category is 'reset', set its object name for easy retrieval
            if category == 'reset':
                checkbox.setObjectName('reset')

            self.checkbox_layout.addWidget(checkbox)

        # Retrieve the reset checkbox after adding categories
        self.reset_checkbox = self.inner_widget.findChild(QCheckBox, 'reset')

    def removeCategory(self, category):
        # Find the checkbox with the given text and remove it
        for checkbox in self.inner_widget.findChildren(QCheckBox):
            if checkbox.text() == category:
                self.checkbox_layout.removeWidget(checkbox)
                checkbox.deleteLater()  # Remove and delete the checkbox
                self.update_display()  # Update display after removal
                break

    def clearCategories(self):
        # Remove all checkboxes
        for checkbox in self.inner_widget.findChildren(QCheckBox):
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()  # Properly delete each checkbox

        # Clear the selected items list and reset the ComboBox display
        self.selected_items.clear()
        self.setCurrentText("Select categories...")  # Reset placeholder text

    def mousePressEvent(self, event):
        # Override mouse press to toggle dropdown immediately on click
        if self.checkbox_container.isVisible():
            self.checkbox_container.hide()
        else:
            self.show_dropdown()
        # Call the parent class's mousePressEvent to maintain ComboBox behavior
        super().mousePressEvent(event)

    def show_dropdown(self):
        # Position and show the dropdown container
        pos = self.mapToGlobal(QPoint(0, self.height()))

        self.checkbox_container.setFixedWidth(245)
        self.checkbox_container.move(pos)
        self.checkbox_container.show()

    def update_display(self):
        # Collect selected items from checked checkboxes, removing duplicates
        self.selected_items = list({
            checkbox.text()
            for checkbox in self.inner_widget.findChildren(QCheckBox)
            if checkbox.isChecked()
        })

        # Update ComboBox display text with comma-separated selected items
        if self.selected_items:
            self.setItemText(0, ", ".join(self.selected_items))
        else:
            self.setItemText(0, "Select categories...")

    def reset_display_text(self):
        self.setItemText(0, "Select categories...")
        self.selected_items.clear()  # Clear the current selection list

    def get_selected_items(self):
        """Return the list of selected items."""
        return self.selected_items

    def check_categories_by_dict(self, category_dict):
        """
        Set the check state of each category based on the provided dictionary.

        Parameters:
        category_dict (dict): Dictionary where keys are category names and values are booleans indicating
                              whether the category should be checked (True) or unchecked (False).
        """
        for checkbox in self.inner_widget.findChildren(QCheckBox):
            # Set the check state based on the dictionary values, defaulting to False if the category isn't specified
            checkbox.setChecked(category_dict.get(checkbox.text(), False))

        # Update the display text based on the new selections
        self.update_display()

    def on_checkbox_state_changed(self, checkbox):
        if checkbox.text() == 'reset' and checkbox.isChecked():
            for cb in self.inner_widget.findChildren(QCheckBox):
                if cb.text() != 'reset':
                    cb.setChecked(False)
        elif checkbox.text() != 'reset' and checkbox.isChecked():
            self.reset_checkbox.setChecked(False)

        self.update_display()

class MultiSelectComboBox2(ComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 다중 선택된 항목을 저장할 리스트
        self.selected_items = []

        # 항목 선택 시 신호 연결
        self.activated.connect(self.toggle_item_selection)

    def addCategory(self, categories):
        # ComboBox에 카테고리를 추가
        self.addItems(categories)

    def clearCategory(self):
        self.clear()

    def toggle_item_selection(self, index):
        # 클릭된 항목의 텍스트 가져오기
        item_text = self.itemText(index)

        if item_text == 'reset':
            self.selected_items.clear()
        else:
            # 선택된 항목을 토글
            if item_text in self.selected_items:
                self.selected_items.remove(item_text)
            else:
                self.selected_items.append(item_text)

        # Placeholder 대신 버튼 텍스트에 결과 표시
        self.update_display()

    def update_display(self):
        # 선택된 항목을 콤마로 구분된 텍스트로 표시
        if self.selected_items:
            result_text = ", ".join(self.selected_items)
            self.setText(result_text)  # 버튼 텍스트에 결과 표시
        else:
            self.setText("Select categories...")  # 선택된 항목이 없으면 기본 텍스트로 설정

    def get_selected_items(self):
        """Return the list of selected items."""
        return self.selected_items