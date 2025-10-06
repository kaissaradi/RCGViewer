from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView, QPushButton, QHBoxLayout, QAbstractItemView
from qtpy.QtCore import Signal, Qt
from gui.widgets import PandasModel
import random
import pandas as pd

class SimilarityPanel(QWidget):
    # Signal emitted when the selection changes; sends list of selected cluster IDs
    selection_changed = Signal(list)
    # Signal emitted when the "Mark as Duplicates" button is pressed
    mark_duplicates = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
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

    def compute_dummy_similarity(self, id1, id2):
        """
        Dummy similarity function (replace with EI correlation later).
        """
        return random.uniform(0, 1)
    
    def set_data(self, similarity_df):
        """Set the DataFrame for the similarity table."""
        self.similarity_model = PandasModel(similarity_df)
        self.table.setModel(self.similarity_model)
        self.table.resizeColumnsToContents()
        # # Connect selectionChanged signal after setting the model
        # if self.table_selection_connected:
        #     try:
        #         self.table.selectionModel().selectionChanged.disconnect(self._on_selection_changed)
        #     except Exception as e:
        #         print(f"Error disconnecting selectionChanged signal: {e}")
        #         pass
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

    def update_main_cluster_id(self, cluster_id, cluster_df):
        
        cluster_ids = cluster_df['cluster_id'].values
        # All cluster IDs except the main one
        all_ids = cluster_ids[cluster_ids != cluster_id]
        # Compute dummy similarity scores
        similarities = [self.compute_dummy_similarity(cluster_id, other_id) for other_id in all_ids]
        # Sort by descending
        sorted_pairs = sorted(zip(all_ids, similarities), key=lambda x: -x[1])
        df = pd.DataFrame(sorted_pairs, columns=['cluster_id', 'similarity'])
        self.set_data(df)