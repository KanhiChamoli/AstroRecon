"Capture simulation and sources windows for visual verification."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["capture"]
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.search_modal import LIBRARY
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "renders")
os.makedirs(OUT, exist_ok=True)
app = QApplication(sys.argv)
w = MainWindow()
w.show()
def capture():
    w.open_simulation(LIBRARY[0])
    sv = w.sim_view
    sv.showMaximized()
    def done():
        try:
            sv.grab().save(os.path.join(OUT, "ui_sim_final.png"))
            sv.meta_tabs.setCurrentIndex(0)
            w.sources_modal.show()
            w.sources_modal.grab().save(os.path.join(OUT, "ui_sources_final.png"))
            w.settings_window.show()
            w.settings_window.grab().save(os.path.join(OUT, "ui_settings_final.png"))
            print("captured simulation, sources, settings")
        finally:
            w.close()
            app.quit()
    QTimer.singleShot(1200, done)
QTimer.singleShot(500, capture)
sys.exit(app.exec_())
