import PySide6
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget, QScrollArea, QCheckBox, QLabel
from qfluentwidgets import ComboBox

class MultiSelectComboBox(ComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Placeholder text for the ComboBox
        self.addItem("Select categories...")

        self.selected_items = []

        # Container for checkboxes with scrolling capability
        self.checkbox_container = QWidget(self)
        self.scroll_area = QScrollArea(self.checkbox_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(200)  # Set fixed height for scrolling

        # Inner widget to hold category checkboxes inside the scroll area
        self.inner_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.inner_widget)

        # Add inner widget to the scroll area
        self.scroll_area.setWidget(self.inner_widget)

        # QLabel to display the count of checked items
        self.count_label = QLabel("Checked: 0", self.checkbox_container)
        self.count_label.setStyleSheet("""
            QLabel {
                padding: 0px;
                font: 600 9pt "Segoe UI";
            }
        """)

        # "Selected All" 체크박스 추가
        self.select_all_checkbox = QCheckBox("Selected All", self.checkbox_container)
        self.select_all_checkbox.setTristate(False)  # Ensure it's two-state
        self.select_all_checkbox.toggled.connect(self.on_select_all_toggled)

        # Main layout for the checkbox container
        layout = QVBoxLayout(self.checkbox_container)
        layout.addWidget(self.count_label)           # QLabel을 상단에 추가
        layout.addWidget(self.select_all_checkbox)   # "Selected All" 체크박스 추가
        layout.addWidget(self.scroll_area)

        # Apply custom styling to the checkbox container
        self.checkbox_container.setStyleSheet("""
            QWidget {
                background-color: white;                     
                border-radius: 8px;
            }
            QCheckBox {
                padding: 3px 0px;  /* Adjust padding to make better use of space */
                font: 600 9pt "Segoe UI";
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                margin-left: 0px;  /* Reduce left margin to make use of full width */
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
        # Ensure that the layout margins are set to zero
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox_layout.setSpacing(0)

        # Configure the checkbox container as a floating widget
        self.checkbox_container.setWindowFlags(Qt.Popup)
        self.checkbox_container.setFixedWidth(self.width())  # Match the ComboBox width

        # 리스트로 카테고리 체크박스를 관리
        self.category_checkboxes = []

    def addCategory(self, categories):
        for category in categories:
            checkbox = QCheckBox(category, self.inner_widget)
            checkbox.toggled.connect(self.on_checkbox_state_changed)
            self.checkbox_layout.addWidget(checkbox)
            self.category_checkboxes.append(checkbox)

    def removeCategory(self, category):
        # Find the checkbox with the given text and remove it
        for checkbox in self.category_checkboxes:
            if checkbox.text() == category:
                self.checkbox_layout.removeWidget(checkbox)
                checkbox.deleteLater()  # Remove and delete the checkbox
                self.category_checkboxes.remove(checkbox)
                self.update_display()  # Update display after removal
                break

    def clearCategories(self):
        # Remove all category checkboxes
        for checkbox in self.category_checkboxes:
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()  # Properly delete each checkbox
        self.category_checkboxes.clear()

        # Clear the selected items list and reset the ComboBox display
        self.selected_items.clear()
        self.setItemText(0, "Select categories...")  # Reset placeholder text

        # Reset the 'Selected All' checkbox
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(False)
        self.select_all_checkbox.blockSignals(False)

        # Update the count label
        self.count_label.setText("Checked: 0")

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
        pos = self.mapToGlobal(QPoint(0, self.height() + 5))

        self.checkbox_container.setFixedWidth(self.width())
        self.checkbox_container.move(pos)
        self.checkbox_container.show()

    def update_display(self):
        # Collect selected items from checked category checkboxes
        self.selected_items = [
            checkbox.text()
            for checkbox in self.category_checkboxes
            if checkbox.isChecked()
        ]

        # Update ComboBox display text with comma-separated selected items
        if self.selected_items:
            self.setItemText(0, ", ".join(self.selected_items))
        else:
            self.setItemText(0, "Select categories...")

        self.count_label.setText(f"Checked: {len(self.selected_items)}")

    def reset_display_text(self):
        self.setItemText(0, "Select categories...")
        self.selected_items.clear()  # Clear the current selection list
        # Reset the 'Selected All' checkbox
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(False)
        self.select_all_checkbox.blockSignals(False)
        # Uncheck all category checkboxes
        for checkbox in self.category_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        # Update the count label
        self.count_label.setText("Checked: 0")

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
        all_checked = True
        for checkbox in self.category_checkboxes:
            should_check = category_dict.get(checkbox.text(), False)
            checkbox.blockSignals(True)
            checkbox.setChecked(should_check)
            checkbox.blockSignals(False)
            if not should_check:
                all_checked = False

        # Set 'Selected All' checkbox based on all categories
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked)
        self.select_all_checkbox.blockSignals(False)

        # Update the display text based on the new selections
        self.update_display()

    def on_select_all_toggled(self, checked):
        # Determine whether to check or uncheck all category checkboxes
        should_check = checked
        # Block signals to prevent recursive calls
        for checkbox in self.category_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(should_check)
            checkbox.blockSignals(False)

        # Update the display and count
        self.update_display()

    def on_checkbox_state_changed(self, checked):
        # Check if all category checkboxes are checked
        all_checked = all(checkbox.isChecked() for checkbox in self.category_checkboxes)

        # Update 'Selected All' checkbox state accordingly
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked)
        self.select_all_checkbox.blockSignals(False)

        # Update the display and count
        self.update_display()

class TristateMultiSelectComboBox(ComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Placeholder text for the ComboBox
        self.addItem("Select categories...")

        self.selected_items = []

        # Container for checkboxes with scrolling capability
        self.checkbox_container = QWidget(self)
        self.scroll_area = QScrollArea(self.checkbox_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(200)  # Set fixed height for scrolling

        # Inner widget to hold category checkboxes inside the scroll area
        self.inner_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.inner_widget)

        # Add inner widget to the scroll area
        self.scroll_area.setWidget(self.inner_widget)

        # QLabel to display the count of checked items
        self.count_label = QLabel("Checked: 0", self.checkbox_container)
        self.count_label.setStyleSheet("""
            QLabel {
                padding: 0px;
                font: 600 9pt "Segoe UI";
            }
        """)

        # "Selected All" 체크박스 추가
        self.select_all_checkbox = QCheckBox("Selected All", self.checkbox_container)
        self.select_all_checkbox.setTristate(True)  # Enable tri-state
        # 기존 stateChanged 시그널 연결을 클릭 시그널로 변경
        self.select_all_checkbox.clicked.connect(self.on_select_all_clicked)

        # Main layout for the checkbox container
        layout = QVBoxLayout(self.checkbox_container)
        layout.addWidget(self.count_label)           # QLabel을 상단에 추가
        layout.addWidget(self.select_all_checkbox)   # "Selected All" 체크박스 추가
        layout.addWidget(self.scroll_area)

        # Apply custom styling to the checkbox container
        self.checkbox_container.setStyleSheet("""
            QWidget {
                background-color: white;                     
                border-radius: 8px;
            }
            QCheckBox {
                padding: 3px 0px;  /* Adjust padding to make better use of space */
                font: 600 9pt "Segoe UI";
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                margin-left: 0px;  /* Reduce left margin to make use of full width */
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
            QCheckBox::indicator:indeterminate {
                background-color: #009faa;
                border: 1px solid #009faa;
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
        # Ensure that the layout margins are set to zero
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox_layout.setSpacing(0)

        # Configure the checkbox container as a floating widget
        self.checkbox_container.setWindowFlags(Qt.Popup)
        self.checkbox_container.setFixedWidth(self.width())  # Match the ComboBox width

        # 리스트로 카테고리 체크박스를 관리
        self.category_checkboxes = []

    def addCategory(self, categories):
        for category in categories:
            checkbox = QCheckBox(category, self.inner_widget)
            checkbox.stateChanged.connect(self.on_checkbox_state_changed)
            self.checkbox_layout.addWidget(checkbox)
            self.category_checkboxes.append(checkbox)

    def removeCategory(self, category):
        # Find the checkbox with the given text and remove it
        for checkbox in self.category_checkboxes:
            if checkbox.text() == category:
                self.checkbox_layout.removeWidget(checkbox)
                checkbox.deleteLater()  # Remove and delete the checkbox
                self.category_checkboxes.remove(checkbox)
                self.update_display()  # Update display after removal
                break

    def clearCategories(self):
        # Remove all category checkboxes
        for checkbox in self.category_checkboxes:
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()  # Properly delete each checkbox
        self.category_checkboxes.clear()

        # Clear the selected items list and reset the ComboBox display
        self.selected_items.clear()
        self.setItemText(0, "Select categories...")  # Reset placeholder text

        # Reset the 'Selected All' checkbox
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)

        # Update the count label
        self.count_label.setText("Checked: 0")

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
        pos = self.mapToGlobal(QPoint(0, self.height() + 5))

        self.checkbox_container.setFixedWidth(self.width())
        self.checkbox_container.move(pos)
        self.checkbox_container.show()

    def update_display(self):
        # Collect selected items from checked category checkboxes
        self.selected_items = [
            checkbox.text()
            for checkbox in self.category_checkboxes
            if checkbox.isChecked()
        ]

        # Update ComboBox display text with comma-separated selected items
        if self.selected_items:
            self.setItemText(0, ", ".join(self.selected_items))
        else:
            self.setItemText(0, "Select categories...")

        self.count_label.setText(f"Checked: {len(self.selected_items)}")

    def reset_display_text(self):
        self.setItemText(0, "Select categories...")
        self.selected_items.clear()  # Clear the current selection list
        # Reset the 'Selected All' checkbox
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)
        # Uncheck all category checkboxes
        for checkbox in self.category_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Unchecked)
            checkbox.blockSignals(False)
        # Update the count label
        self.count_label.setText("Checked: 0")

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
        all_checked = True
        any_checked = False
        for checkbox in self.category_checkboxes:
            should_check = category_dict.get(checkbox.text(), False)
            checkbox.blockSignals(True)
            if should_check:
                checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Checked)
                any_checked = True
            else:
                checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Unchecked)
            checkbox.blockSignals(False)
            if not should_check:
                all_checked = False

        # Set 'Selected All' checkbox based on all categories
        self.select_all_checkbox.blockSignals(True)
        if all_checked:
            self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Checked)
        elif any_checked:
            self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.PartiallyChecked)
        else:
            self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)

        # Update the display text based on the new selections
        self.update_display()

    def on_select_all_clicked(self, checked):
        """
        Handle the 'Select All' checkbox click event to toggle all category checkboxes.
        """
        # Determine if we need to check or uncheck all based on current state
        if self.select_all_checkbox.checkState() == PySide6.QtCore.Qt.CheckState.Checked:
            should_check = False
        else:
            should_check = True

        # Block signals to prevent recursive calls
        for checkbox in self.category_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Checked if should_check else PySide6.QtCore.Qt.CheckState.Unchecked)
            checkbox.blockSignals(False)

        # Update the 'Select All' checkbox state
        self.select_all_checkbox.blockSignals(True)
        if should_check:
            self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Checked)
        else:
            self.select_all_checkbox.setCheckState(PySide6.QtCore.Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)

        # Update the display and count
        self.update_display()

    def on_checkbox_state_changed(self, state):
        """
        Handle individual category checkbox state changes to update the 'Select All' checkbox.
        """
        # Determine the overall state
        checked_count = sum(1 for checkbox in self.category_checkboxes if checkbox.isChecked())
        total_count = len(self.category_checkboxes)

        if checked_count == total_count:
            overall_state = PySide6.QtCore.Qt.CheckState.Checked
        elif checked_count == 0:
            overall_state = PySide6.QtCore.Qt.CheckState.Unchecked
        else:
            overall_state = PySide6.QtCore.Qt.CheckState.PartiallyChecked

        # Update 'Select All' checkbox state accordingly
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(overall_state)
        self.select_all_checkbox.blockSignals(False)

        # Update the display and count
        self.update_display()
