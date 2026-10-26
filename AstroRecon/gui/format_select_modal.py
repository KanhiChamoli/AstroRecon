"""Data-format selection modal — "Select Data Format & Pipeline".

Appears when the user triggers a simulation: lists every data format the
dispatcher can route (FITS / MHD / UV-FITS / Optical + runtime-registered
custom formats), lets the user pick one, and hands the choice back to the
main window which routes the data through
:func:`src.pipeline_router.route_data` into the matching backend pipeline.

Layout (spec):
  * scrollable grid of selectable format cards,
  * bottom bar with styled **Cancel** (close, return to previous menu) and
    **Simulate Data** (trigger routing) buttons.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QWidget, QGridLayout, QApplication, QSizePolicy,
)

from src.pipeline_router import list_formats, FormatRoute


class FormatCard(QPushButton):
    """One selectable data-format tile in the grid."""

    def __init__(self, route: FormatRoute, parent=None) -> None:
        super().__init__(parent)
        self.route = route
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(92)
        self.setText(
            f"◧  {route.title}\n\n{route.description}\n"
            f"→ pipeline: {route.pipeline_module}"
        )
        self.setStyleSheet(
            "QPushButton {"
            "  text-align: left; white-space: pre; padding: 12px 14px;"
            "  border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px;"
            "  background: rgba(14, 25, 45, 0.65); color: #e2e8f0; font-size: 12px;}"
            "QPushButton:hover {"
            "  border-color: rgba(56, 189, 248, 0.7);"
            "  background: rgba(23, 42, 74, 0.85);}"
            "QPushButton:checked {"
            "  border: 2px solid #38bdf8;"
            "  background: rgba(23, 52, 92, 0.95); color: #f0f9ff;}"
        )


class FormatSelectModal(QDialog):
    """'Select Data Format & Pipeline' — modal data-format chooser."""

    # Emitted with the chosen FormatRoute + the data path (may be None).
    simulate_requested = pyqtSignal(object, object)

    def __init__(self, parent=None, data_paths=None) -> None:
        super().__init__(parent)
        self.data_paths = list(data_paths or [])
        self.setWindowTitle("Astro Recon — Select Data Format & Pipeline")
        self.setModal(True)
        self.setMinimumSize(640, 480)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(760, screen.width()), min(560, screen.height()))
        self._selected: FormatRoute | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ---- header -------------------------------------------------------
        head = QHBoxLayout()
        title = QLabel("◈  SELECT DATA FORMAT & PIPELINE")
        title.setObjectName("Section")
        close = QPushButton("✕")
        close.setFixedSize(34, 34)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.reject)          # = Cancel behaviour
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(close)
        root.addLayout(head)

        subtitle = QLabel(
            "Choose which pipeline the workspace data is routed through. "
            "Formats are auto-detected from FITS headers and file types; "
            "your selection overrides the detection.")
        subtitle.setObjectName("Dim")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # ---- scrollable grid of format cards ------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 4, 0, 4)
        grid.setSpacing(10)

        self.cards: list[FormatCard] = []
        for i, route in enumerate(list_formats()):
            card = FormatCard(route, content)
            card.clicked.connect(
                lambda _=False, c=card: self._select(c))
            grid.addWidget(card, i // 2, i % 2)     # 2-column grid
            self.cards.append(card)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # Pre-select the detected format for the first data path, if any.
        if self.data_paths:
            self._preselect()

        # ---- bottom action bar --------------------------------------------
        bar = QHBoxLayout()
        self.hint = QLabel("")
        self.hint.setObjectName("Dim")
        cancel = QPushButton("Cancel")
        cancel.setMinimumHeight(44)
        cancel.setMinimumWidth(140)
        cancel.setCursor(Qt.PointingHandCursor)
        cancel.clicked.connect(self.reject)         # close, back to menu
        simulate = QPushButton("▶  Simulate Data")
        simulate.setObjectName("MenuButton")
        simulate.setMinimumHeight(44)
        simulate.setMinimumWidth(200)
        simulate.setCursor(Qt.PointingHandCursor)
        simulate.clicked.connect(self._emit_simulate)
        bar.addWidget(self.hint, 1)
        bar.addWidget(cancel)
        bar.addWidget(simulate)
        root.addLayout(bar)

    # ------------------------------------------------------------ helpers --
    def _select(self, card: FormatCard) -> None:
        """Exclusive selection: one card checked at a time."""
        for c in self.cards:
            c.setChecked(c is card)
        self._selected = card.route
        self.hint.setText(
            f"Selected: {card.route.title}  →  {card.route.pipeline_module}")

    def _preselect(self) -> None:
        """Auto-detect the format of the first data path and preselect it."""
        try:
            from src.pipeline_router import detect_format
            detected = detect_format(self.data_paths[0])
            for card in self.cards:
                if card.route.fmt == detected.fmt:
                    self._select(card)
                    self.hint.setText(
                        f"Auto-detected: {detected.title} "
                        f"({detected.reason}) — adjust if needed")
                    return
        except Exception:
            pass  # detection is best-effort only

    def _emit_simulate(self) -> None:
        if self._selected is None:
            self.hint.setText("Select a data format first.")
            return
        self.simulate_requested.emit(self._selected, self.data_paths)
        self.accept()

    # ------------------------------------------------------------ theme ----
    def apply_palette(self, palette) -> None:
        """ThemeManager hook — required so the modal can be theme-listened.

        The card stylesheets are inline (fixed dark-theme colors), so the
        palette only needs to be accepted; without this method the theme
        manager raises AttributeError on open and the app aborts.
        """
        self._palette = palette  # noqa: F841  (kept for future theming)
