"""Astro Recon â€” top-level window wiring every page/modal together."""
from __future__ import annotations

import os
import webbrowser

from PyQt5.QtCore import Qt, QEvent, QObject
from PyQt5.QtGui import QKeySequence, QCloseEvent, QCursor, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QShortcut, QApplication, QAbstractButton, QWidget, QTabBar

from .theme import ThemeManager, build_qss
from .audio import AudioController
from src.hardware import get_manager
from .main_menu import MainMenuPage
from .search_modal import SearchModal
from .settings_drawer import SettingsWindow, SettingsDrawer
from .simulation_view import SimulationView
from .sources_modal import SourcesModal

GITHUB_URL = "https://github.com/"  # TODO: replace with the Aphellion Studios repo URL

# Reference telescope sprite used for the custom cursor. Scaled up at load
# time so the telescope is clearly visible on screen.
CURSOR_SPRITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "images", "cursor_reference.png"
)
CURSOR_SIZE = 96  # final on-screen cursor size (px) — big enough to be clearly visible


class InterfaceFeedback(QObject):
    'Global theme-aware pixel telescope cursor and synthesized click sounds.'

    def __init__(self, theme_manager, audio, parent=None):
        super().__init__(parent)
        self._audio = audio
        self._cursor = None
        self._hover_cursor = None
        self._theme_manager = theme_manager
        theme_manager.add_listener(self.apply_palette)
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

    @staticmethod
    def _render_telescope_pixmap(color: str) -> QPixmap:
        """Load the reference telescope sprite, scaled up for visibility."""
        src = QPixmap(CURSOR_SPRITE_PATH)
        if src.isNull():
            return QPixmap(48, 48)  # graceful fallback if the asset goes missing
        # scale to a fixed on-screen size so the telescope is clearly visible
        return src.scaled(CURSOR_SIZE, CURSOR_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    @classmethod
    def _make_telescope_cursor(cls, color: str) -> QCursor:
        # Hotspot at the scope's upper-right tip (objective end) so the
        # telescope "aims" at the point under the cursor.
        return QCursor(cls._render_telescope_pixmap(color),
                       int(CURSOR_SIZE * 0.82), int(CURSOR_SIZE * 0.15))

    def apply_palette(self, palette) -> None:
        # Dark themes get cyan/blue telescope, Light gets pale gold.
        color = "#38bdf8" if palette.name.lower() != "light" else "#d6a83e"
        self._cursor = self._make_telescope_cursor(color)
        self._hover_cursor = self._make_telescope_cursor(
            "#a78bfa" if palette.name.lower() != "light" else "#f0d27a"
        )

    def eventFilter(self, obj, event):
        # Give newly shown widgets the theme-aware pixel telescope cursor.
        if isinstance(obj, QWidget):
            if event.type() == QEvent.Show and self._cursor:
                obj.setCursor(self._cursor)
            if isinstance(obj, QAbstractButton):
                if event.type() == QEvent.Enter and self._hover_cursor:
                    obj.setCursor(self._hover_cursor)
                elif event.type() == QEvent.Leave and self._cursor:
                    obj.setCursor(self._cursor)
            if isinstance(obj, (QAbstractButton, QTabBar)) and event.type() == QEvent.MouseButtonRelease:
                if getattr(obj, "isEnabled", lambda: True)() and self._audio:
                    self._audio.play_interface_click()
        return False
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Astro Recon")
        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(1100, 700)

        # shared services
        self.tm = ThemeManager()
        self.audio = AudioController()
        self.interface_feedback = InterfaceFeedback(self.tm, self.audio, self)

        # root stack (main menu is the only full page right now)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # shared strict-GPU device context: the settings dropdown retargets
        # this manager, and any tensor-owning simulation reads
        # get_manager().device() at launch (or receives it via listener).
        self.hardware_manager = get_manager()
        self.hardware_manager.add_listener(self._on_device_changed)
        self.menu_page = MainMenuPage(
            self.tm,
            on_run=self.open_search,
            on_sources=self.open_sources,
            on_github=lambda: webbrowser.open(GITHUB_URL),
            on_settings=self.toggle_settings,
        )
        self.stack.addWidget(self.menu_page)

        # standalone pop-out settings window
        self.settings_window = SettingsWindow(self.tm, self.audio, parent=None)
        self.drawer = self.settings_window  # alias for backward compat
        self.tm.add_listener(self._apply_qss)
        self.tm.add_listener(self.settings_window.apply_palette)

        # modals
        self.search_modal = SearchModal(self)
        self.tm.add_listener(self.search_modal.apply_palette)
        self.search_modal.object_selected.connect(self._start_simulation)
        self.search_modal.nu_upload.connect(self._start_simulation_from_upload)

        self.sources_modal = SourcesModal(self)
        self.tm.add_listener(self.sources_modal.apply_palette)

        # keyboard shortcut: Ctrl+K search, Ctrl+, settings
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.open_search)
        QShortcut(QKeySequence("Ctrl+,"), self, activated=self.toggle_settings)

        # start ambient music (if available)
        self.audio.play()

    # ------------------------------------------------------------ wiring --
    def _on_device_changed(self, identifier: str) -> None:
        """React to live compute-device retargeting from the settings UI."""
        kind = "GPU" if identifier.startswith("cuda") else "CPU fallback"
        self.statusBar().showMessage(f"Compute device: {identifier} ({kind})", 4000)

    def _apply_qss(self, p) -> None:
        QApplication.instance().setStyleSheet(build_qss(p))

    def toggle_settings(self) -> None:
        self.settings_window.toggle_near(self.menu_page.settings_btn)

    def open_search(self) -> None:
        self.search_modal._populate("")  # reset view
        self.search_modal.search.setFocus()
        self.search_modal.exec_()

    def open_sources(self) -> None:
        self.sources_modal.exec_()

    def _start_simulation(self, entry: dict) -> None:
        dlg = SimulationView(entry, [entry.get("default_data")] if entry.get("default_data") else [], self)
        self.tm.add_listener(dlg.apply_palette)
        dlg.exec_()

    def open_simulation(self, entry: dict) -> None:
        """Non-modal open â€” used by tests and programmatic launches."""
        self.sim_view = SimulationView(entry, [entry.get("default_data")] if entry.get("default_data") else [], self)
        self.tm.add_listener(self.sim_view.apply_palette)
        self.sim_view.show()

    def _start_simulation_from_upload(self, paths: list) -> None:
        entry = {
            "name": f"Custom Object (NU) â€” {paths[0]}",
            "kind": "User-uploaded dataset",
            "default_data": None,
        }
        dlg = SimulationView(entry, paths, self)
        self.tm.add_listener(dlg.apply_palette)
        dlg.exec_()

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)

    def closeEvent(self, e: QCloseEvent):  # noqa: N802
        self.settings_window.close()
        self.audio.stop()
        super().closeEvent(e)
