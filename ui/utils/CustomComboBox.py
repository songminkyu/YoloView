from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QVBoxLayout, QWidget, QScrollArea, QCheckBox, QLabel, QLineEdit
)
from PySide6.QtGui import QFontMetrics
from qfluentwidgets import ComboBox

class TristateMultiSelectComboBox(ComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Placeholder text for the ComboBox
        self.addItem("Select categories...")

        self.selected_items = []
        self.categories = dict()
        self.categories_count = 0
        # Container for checkboxes with scrolling capability
        self.checkbox_container = QWidget(self)
        self.scroll_area = QScrollArea(self.checkbox_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(200)  # Set fixed height for scrolling

        # Inner widget to hold category checkboxes inside the scroll area
        self.inner_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.inner_widget)
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox_layout.setSpacing(0)
        self.checkbox_layout.setAlignment(Qt.AlignTop)  # Align checkboxes to the top

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

        # Search bar for filtering categories
        self.search_bar = QLineEdit(self.checkbox_container)
        self.search_bar.setPlaceholderText("Search categories...")
        self.search_bar.textChanged.connect(self.filter_categories)
        self.search_bar.setStyleSheet("""
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

        # "Selected All" checkbox
        self.select_all_checkbox = QCheckBox("Selected All", self.checkbox_container)
        self.select_all_checkbox.setTristate(True)  # Enable tri-state
        self.select_all_checkbox.clicked.connect(self.on_select_all_clicked)

        # Main layout for the checkbox container
        layout = QVBoxLayout(self.checkbox_container)
        layout.addWidget(self.search_bar)           # Add search bar at the top
        layout.addWidget(self.count_label)          # QLabel below the search bar
        layout.addWidget(self.select_all_checkbox)  # "Selected All" checkbox
        layout.addWidget(self.scroll_area)

        # Apply custom styling to the checkbox container
        self.checkbox_container.setStyleSheet("""
            QWidget {
                background-color: white;                     
                border-radius: 8px;
            }
            QCheckBox {
                padding: 3px 0px;
                font: 600 9pt "Segoe UI";
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                margin-left: 0px;
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
                background-color: #DB7093;
                border: 1px solid #DB7093;
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

        # Configure the checkbox container as a floating widget
        self.checkbox_container.setWindowFlags(Qt.Popup)
        self.checkbox_container.setFixedWidth(self.width())  # Match the ComboBox width

        # List to manage category checkboxes
        self.category_checkboxes = []

    def addCategories(self, categories):
        # Update the count label
        self.categories = categories
        self.categories_count = len(categories)
        self.count_label.setText(f"Checked: 0 / {self.categories_count}")

        for class_name in categories.values():
            checkbox = QCheckBox(class_name, self.inner_widget)
            checkbox.stateChanged.connect(self.on_checkbox_state_changed)
            self.checkbox_layout.addWidget(checkbox)
            self.category_checkboxes.append(checkbox)

    def removeCategories(self, category):
        # Find the checkbox with the given text and remove it
        for checkbox in self.category_checkboxes:
            if checkbox.text() == category:
                self.checkbox_layout.removeWidget(checkbox)
                checkbox.deleteLater()
                self.category_checkboxes.remove(checkbox)
                self.update_display()
                break

    def clearCategories(self):
        # Remove all category checkboxes
        for checkbox in self.category_checkboxes:
            self.checkbox_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.category_checkboxes.clear()

        # Clear the selected items list and reset the ComboBox display
        self.selected_items.clear()
        self.setItemText(0, "Select categories...")

        # Reset the 'Selected All' checkbox
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)

    def mousePressEvent(self, event):
        # Override mouse press to toggle dropdown immediately on click
        if self.checkbox_container.isVisible():
            self.checkbox_container.hide()
        else:
            self.show_dropdown()
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

        # Calculate the display width using font metrics
        font_metrics = QFontMetrics(self.font())
        max_width = 200  # Maximum width for the placeholder text in pixels

        # Build display text and truncate with "..." if it exceeds max_width
        if self.selected_items:
            display_text = ", ".join(self.selected_items)
            if font_metrics.horizontalAdvance(display_text) > max_width:
                # Truncate text to fit within max_width
                display_text = font_metrics.elidedText(display_text, Qt.ElideRight, max_width)

            self.setItemText(0, display_text)
            self.setToolTip(", ".join(self.selected_items))  # Tooltip shows full list
        else:
            self.setItemText(0, "Select categories...")
            self.setToolTip("Select categories...")

        # Update the count label
        self.count_label.setText(f"Checked: {len(self.selected_items)} / {self.categories_count}")

    def reset_display_text(self):
        self.setItemText(0, "Select categories...")
        self.selected_items.clear()
        # Reset the 'Selected All' checkbox
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)
        # Uncheck all category checkboxes
        for checkbox in self.category_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setCheckState(Qt.CheckState.Unchecked)
            checkbox.blockSignals(False)

    def get_selected_categories(self):
        """Return the dictionary of selected items."""
        selected_categories = {
            key: value for key, value in self.categories.items()
            if any(checkbox.text() == value and checkbox.isChecked() for checkbox in self.category_checkboxes)
        }
        return selected_categories

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
                checkbox.setCheckState(Qt.CheckState.Checked)
                any_checked = True
            else:
                checkbox.setCheckState(Qt.CheckState.Unchecked)
            checkbox.blockSignals(False)
            if not should_check:
                all_checked = False

        # Set 'Selected All' checkbox based on all categories
        self.select_all_checkbox.blockSignals(True)
        if all_checked:
            self.select_all_checkbox.setCheckState(Qt.CheckState.Checked)
        elif any_checked:
            self.select_all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self.select_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.select_all_checkbox.blockSignals(False)

        # Update the display text based on the new selections
        self.update_display()

    def on_select_all_clicked(self, checked):
        """
        Handle the 'Select All' checkbox click event to toggle all category checkboxes.
        """
        # Determine if we need to check or uncheck all based on current state
        if self.select_all_checkbox.checkState() == Qt.CheckState.Unchecked:
            should_check = False
        else:
            should_check = True

        # Block signals to prevent recursive calls
        for checkbox in self.category_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setCheckState(Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked)
            checkbox.blockSignals(False)

        # Update the 'Select All' checkbox state
        self.select_all_checkbox.blockSignals(True)
        if should_check:
            self.select_all_checkbox.setCheckState(Qt.CheckState.Checked)
        else:
            self.select_all_checkbox.setCheckState(Qt.CheckState.Unchecked)
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
            overall_state = Qt.CheckState.Checked
        elif checked_count == 0:
            overall_state = Qt.CheckState.Unchecked
        else:
            overall_state = Qt.CheckState.PartiallyChecked

        # Update 'Select All' checkbox state accordingly
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setCheckState(overall_state)
        self.select_all_checkbox.blockSignals(False)

        # Update the display and count
        self.update_display()

    def filter_categories(self, text):
        """
        Filter the category checkboxes based on the search text.
        """
        for checkbox in self.category_checkboxes:
            if text.lower() in checkbox.text().lower():
                checkbox.show()
            else:
                checkbox.hide()
