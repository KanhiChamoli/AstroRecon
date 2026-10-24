"""Capture a fresh UI snapshot of the running simulation window (headless-safe).

Grabs the simulation dialog after a short settle delay and writes PNGs to
outputs/renders/. Also exercises the three panels so the frames are real
pipeline output rather than idle placeholders.

Run:  python gui/capture_snapshot.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["t"]

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.search_modal import LIBRARY  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "renders")
os.makedirs(OUT, exist_ok=True)

app = QApplication(sys.argv)
w = MainWindow()
w.show()


def capture() -> None:
    try:
        # main menu first
        w.menu_page.grab().save(os.path.join(OUT, "ui_menu_latest.png"))
        print("saved ui_menu_latest.png")

        # simulation window, mid-run so panels have real frames
        w.open_simulation(LIBRARY[0])
        sv = w.sim_view
        sv.resize(1360, 900)
        sv.show()

        def snap_sim() -> None:
            try:
                sv.grab().save(os.path.join(OUT, "ui_sim_latest.png"))
                print("saved ui_sim_latest.png")
                # switch to residual mode and re-grab for the export/view evidence
                sv._set_view_mode("Absolute Residual Error")
                sv.grab().save(os.path.join(OUT, "ui_sim_residual_latest.png"))
                print("saved ui_sim_residual_latest.png")
                # drawer / sources panel
                w.drawer.show_drawer()
                w.drawer.grab().save(os.path.join(OUT, "ui_drawer_latest.png"))
                print("saved ui_drawer_latest.png")
                w.close()
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                app.quit()

        QTimer.singleShot(1800, snap_sim)
    except Exception:
        import traceback
        traceback.print_exc()
        app.quit()


QTimer.singleShot(700, capture)
sys.exit(app.exec_())
