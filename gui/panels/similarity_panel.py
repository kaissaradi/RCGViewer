from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView, QPushButton, QHBoxLayout, QAbstractItemView
from qtpy.QtCore import Signal, Qt
from gui.widgets import PandasModel
import numpy as np
import pandas as pd

class SimilarityPanel(QWidget):
    # Signal emitted when the selection changes; sends list of selected cluster IDs
    selection_changed = Signal(list)
    # Signal emitted when the "Mark as Duplicates" button is pressed
    mark_duplicates = Signal(list)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("Similar Clusters")
        layout.addWidget(self.label)

        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        # Button row
        button_layout = QHBoxLayout()
        self.duplicate_button = QPushButton("Mark as Duplicates")
        button_layout.addWidget(self.duplicate_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.duplicate_button.clicked.connect(self._on_mark_duplicates)

        self.similarity_model = None

        # Connect selection change after model is set (see set_data)
        self.table_selection_connected = False
    
    def set_data(self, similarity_df):
        """Set the DataFrame for the similarity table."""
        self.similarity_model = PandasModel(similarity_df)
        self.table.setModel(self.similarity_model)
        self.table.resizeColumnsToContents()
        self.table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.table_selection_connected = True

    def _on_selection_changed(self):
        """Emit the list of selected cluster IDs."""
        indexes = self.table.selectionModel().selectedRows()
        if self.similarity_model is not None:
            selected_ids = [self.similarity_model._dataframe.iloc[idx.row()]['cluster_id'] for idx in indexes]
            self.selection_changed.emit(selected_ids)

    def _on_mark_duplicates(self):
        """Emit the selected clusters as a duplicate group."""
        indexes = self.table.selectionModel().selectedRows()
        if self.similarity_model is not None:
            selected_ids = [self.similarity_model._dataframe.iloc[idx.row()]['cluster_id'] for idx in indexes]
            self.mark_duplicates.emit(selected_ids)

    def clear(self):
        """Clear the table."""
        self.table.setModel(None)
        self.similarity_model = None

    def update_main_cluster_id(self, cluster_id):
        # Get EI correlation values from data_manager
        if self.main_window.data_manager is None or self.main_window.data_manager.ei_corr_dict is None:
            print("Error: DataManager or EI correlation data not available.")
            self.clear()
            return

        ei_corr_dict = self.main_window.data_manager.ei_corr_dict
        cluster_ids = np.array(list(self.main_window.data_manager.vision_eis.keys())) - 1
        main_idx = np.where(cluster_ids == cluster_id)[0][0]
        other_idx = np.where(cluster_ids != cluster_id)[0]
        other_ids = cluster_ids[other_idx]
        d_df = {
            'cluster_id': other_ids,
            'full_ei_corr': ei_corr_dict['full'][main_idx, other_idx],
            'space_ei_corr': ei_corr_dict['space'][main_idx, other_idx],
            'power_ei_corr': ei_corr_dict['power'][main_idx, other_idx]

        }
        df = pd.DataFrame(d_df)
        df = df.sort_values(by='full_ei_corr', ascending=False).reset_index(drop=True)
        self.set_data(df)