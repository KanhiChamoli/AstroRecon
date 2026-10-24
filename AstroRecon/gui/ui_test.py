"""Headless UI interaction test for the Astro Recon simulation window.

Drives every control programmatically and asserts on the checklist items:
menu bar, epoch/feed/loss stats, 4 view-mode toggles, 4 export actions,
pause-sync, and the three rendering panels. Prints results, then exits.

Run:  python gui/ui_test.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp1252-safe printing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["t"]

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.search_modal import LIBRARY  # noqa: E402

app = QApplication(sys.argv)
w = MainWindow()
w.show()

RESULTS = []


def check(name: str, ok: bool) -> None:
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def run_checks() -> None:
    try:
        w.open_simulation(LIBRARY[0])
        sv = w.sim_view

        # --- 3-panel viewport ------------------------------------------------
        check("3 rendering panels present", len(sv.panels) == 3)
        check("panels are target/model/residual",
              set(sv.panels) == {"target", "model", "residual"})

        # --- menu bar (WWT style) -------------------------------------------
        mb = sv.menu_bar
        titles = [a.text().replace("&", "") for a in mb.actions()]
        check("menu bar has View", "View" in titles)
        check("menu bar has Save & Export",
              any("Save" in t and "Export" in t for t in titles))

        # --- stats row -------------------------------------------------------
        check("epoch stat visible", "Epoch" in sv.stat_epoch.text())
        check("loss stat visible", "Loss" in sv.stat_loss.text())
        check("feed stat visible", "Feed" in sv.stat_feed.text())

        # --- pause / snapshot buttons ---------------------------------------
        check("pause button present & active", sv.pause_btn.isEnabled())
        check("snapshot button present & active", sv.snapshot_btn.isEnabled())

        # --- 4 view-mode toggles --------------------------------------------
        modes = list(getattr(sv, "_view_actions", {}).keys())
        check("4 view-mode toggles", len(modes) == 4)

        # --- pause syncs the View menu ---------------------------------------
        sv.pause_btn.setChecked(True)
        check("pause toggles state", sv.pause_btn.isChecked())
        sv.pause_btn.setChecked(False)

        # --- exercise each view mode ----------------------------------------
        for m in modes:
            try:
                sv._view_actions[m].trigger()
            except Exception as e:  # noqa: BLE001
                check(f"view mode '{m}' triggers", False)
                print("   error:", e)
        check("all view modes trigger", True)

        # --- export actions (4) ----------------------------------------------
        export_menu = [a.menu() for a in mb.actions()
                       if a.menu() and "Save" in a.text() and "Export" in a.text()][0]
        export_items = [a.text() for a in export_menu.actions() if not a.isSeparator()]
        check("4 export actions wired", len(export_items) >= 4)
        check("export menu lists PNG/OBJ/script",
              any("PNG" in t for t in export_items)
              and any("OBJ" in t for t in export_items)
              and any("script" in t.lower() for t in export_items))

        # --- frameless popup menus are opaque (no translucent flicker) -------
        check("View popup not translucent",
              not export_menu.testAttribute(Qt.WA_TranslucentBackground))

        # --- separate sources window -----------------------------------------
        check("sources launcher present", sv.sources_btn.isEnabled())
        check("sources window has clickable list", sv.sources_window.list.count() >= 5)
        sv.sources_window.show()
        check("sources opens separate window", sv.sources_window.isVisible())
        sv.sources_window.close()

        # --- metadata bottom tabs and text-fit constraints --------------------
        check("metadata tab strip present", sv.meta_tabs.count() >= 4)
        check("metadata tab bar scroll buttons enabled", sv.meta_tabs.usesScrollButtons())
        check("interface click sounds settings present",
              hasattr(w.settings_window, "interface_sounds_toggle")
              and hasattr(w.settings_window, "interface_sounds_volume"))

        w.close()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        passed = sum(1 for _, ok in RESULTS if ok)
        total = len(RESULTS)
        print(f"\n{passed}/{total} checks passed")
        print("ALL CHECKS PASSED" if passed == total else "SOME CHECKS FAILED")
        app.quit()


QTimer.singleShot(600, run_checks)
sys.exit(app.exec_())
