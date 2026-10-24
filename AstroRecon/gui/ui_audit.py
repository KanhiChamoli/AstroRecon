"""UI audit — opens the Sgr A* simulation view, cycles tabs/menus, and dumps
screenshots to outputs/renders/audit_*.png so the real rendering can be
inspected. Run:  python gui/ui_audit.py"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["audit"]

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.search_modal import LIBRARY  # noqa: E402

OUT = os.path.join("outputs", "renders")
os.makedirs(OUT, exist_ok=True)

app = QApplication(sys.argv)
w = MainWindow()
w.show()

steps = []
_shots = []


def shot(name):
    p = os.path.join(OUT, f"audit_{name}.png")
    _shots.append(p)
    return p


def step_menu():
    w.menu_page.grab().save(shot("menu"))
    print("[1] menu captured")
    w.open_simulation(LIBRARY[0])
    QTimer.singleShot(500, step_sim)


def step_sim():
    v = w.sim_view
    v.resize(1280, 820)
    v.grab().save(shot("sim_view"))
    print("[2] sim view captured; view_mode =", getattr(v, "_view_mode", "?"))

    # find tab widget
    tabs = getattr(v, "meta_tabs", None)
    if tabs is None:
        from PyQt5.QtWidgets import QTabWidget
        tws = v.findChildren(QTabWidget)
        tabs = tws[0] if tws else None
    if tabs is not None:
        print("    tabs count:", tabs.count())
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            app.processEvents()
            v.grab().save(shot(f"tab_{i}"))
            print(f"[3] tab {i} ({tabs.tabText(i)}) captured")

    # scrollbar audit on sources box
    src = getattr(v, "sources", None)
    if src is not None:
        from PyQt5.QtWidgets import QScrollArea
        outer = src.parent()
        print("    sources widget:", type(src).__name__,
              "| parent:", type(outer).__name__ if outer else None,
              "| is QScrollArea:", isinstance(outer, QScrollArea))
    QTimer.singleShot(300, step_done)


def step_done():
    w.close()
    print("AUDIT DONE")
    print("shots:", _shots)
    app.quit()


QTimer.singleShot(700, step_menu)
sys.exit(app.exec_())
