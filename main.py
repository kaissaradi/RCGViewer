import sys
from qtpy.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    # Set your default paths here for testing
    DEFAULT_KILOSORT_DIR = "/Volumes/Vyom MEA/analysis/20250306C/data026/kilosort2.5"
    DEFAULT_DAT_FILE = "/Volumes/Vyom MEA/data/raw/20250306C/data026.bin"
    app = QApplication(sys.argv)
    window = MainWindow(DEFAULT_KILOSORT_DIR, DEFAULT_DAT_FILE)
    window.show()
    sys.exit(app.exec())
