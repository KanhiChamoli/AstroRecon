"Open the simulation + Sources windows and capture actual Qt widgets."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["capture"]
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.search_modal import LIBRARY
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "renders")
os.makedirs(out, exist_ok=True)
app = QApplication(sys.argv)
w = MainWindow()
w.show()
def launch():
    w.open_simulation(LIBRARY[0])
    v = w.sim_view
    v.show()
    v.raise_()
    v.activateWindow()
    def capture():
        app.processEvents()
        print("sim geometry", v.geometry().getRect(), "grab", v.grab().size().width(), v.grab().size().height(),
              "tabs", v.meta_tabs.geometry().getRect(), "pause", v.pause_btn.geometry().getRect(),
              "snapshot", v.snapshot_btn.geometry().getRect())
        v.grab().save(os.path.join(out, "astrorecon_simulation_latest.png"))
        v.sources_window.show()
        v.sources_window.raise_()
        app.processEvents()
        v.sources_window.grab().save(os.path.join(out, "astrorecon_sources_latest.png"))
        print("captured simulation and sources")
        w.close()
        app.quit()
    QTimer.singleShot(1500, capture)
QTimer.singleShot(500, launch)
sys.exit(app.exec_())
