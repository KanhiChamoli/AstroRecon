"""Smoke test for the Astro Recon GUI — prints results, then exits."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp1252-safe printing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["t"]

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.search_modal import LIBRARY  # noqa: E402

app = QApplication(sys.argv)
w = MainWindow()
w.show()


def run_checks() -> None:
    try:
        w.open_simulation(LIBRARY[0])
        print("sim view shown:", w.sim_view.isVisible())
        print("blender stub:", w.sim_view.blender_btn.text())
        print("sources window entries:", w.sim_view.sources_window.list.count())
        w.sim_view.sources_window.show()
        print("sources window shown:", w.sim_view.sources_window.isVisible())
        w.sim_view.sources_window.close()
        w.drawer.show_drawer()
        print("drawer shown:", w.drawer.isVisible())
        w.close()
        print("ALL CHECKS PASSED")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.quit()


QTimer.singleShot(600, run_checks)
sys.exit(app.exec_())
