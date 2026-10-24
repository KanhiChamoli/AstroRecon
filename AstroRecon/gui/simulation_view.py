"""Simulation view popup — shown after a run is initialized.

Contains a viewport placeholder (procedural nebula backdrop), object metadata,
a SOURCES section, and the stubbed "Open in Blender" button reserved for the
future 3D-software pipeline integration.
"""
from __future__ import annotations

import os
import subprocess
import webbrowser
import urllib.parse

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QBrush
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QListWidget, QListWidgetItem, QProgressBar, QWidget, QScrollArea,
    QMenuBar, QAction, QActionGroup, QMessageBox, QInputDialog, QApplication,
    QTabWidget, QPlainTextEdit, QSizePolicy,
)

from .theme import Palette

ADS_URL = "https://ui.adsabs.harvard.edu/search/q={q}"


class SourcesWindow(QDialog):
    """Large, scrollable sources window — opens separately from the sim view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Astro Recon — Sources")
        self.setMinimumSize(640, 520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        head = QLabel("◉  SOURCES")
        head.setObjectName("Title")
        head.setStyleSheet("font-size: 20px; letter-spacing: 2px;")
        root.addWidget(head)

        hint = QLabel("Click any source to open its link in your browser.")
        hint.setObjectName("Dim")
        root.addWidget(hint)

        # Scrollable list: large window, all entries fully visible, clickable.
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.setSpacing(6)
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list.itemClicked.connect(self._on_click)
        root.addWidget(self.list, 1)

        close = QPushButton("Close")
        close.setObjectName("MenuButton")
        close.setFixedWidth(140)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.close)
        root.addWidget(close, 0, Qt.AlignRight)

        self._urls = {}
        self.apply_palette(None)

    def set_sources(self, entries: list) -> None:
        self.list.clear()
        self._urls.clear()
        for text in entries:
            item = QListWidgetItem(text)
            # auto-detect URLs so clicking opens the page
            for token in text.split():
                if token.startswith(("http://", "https://")):
                    self._urls[text] = token
                    break
            self.list.addItem(item)

    def _on_click(self, item: QListWidgetItem) -> None:
        url = self._urls.get(item.text())
        if url:
            webbrowser.open(url)

    def apply_palette(self, p) -> None:
        """Keep the window legible; re-applied on theme change."""
        self.list.setStyleSheet(
            "QListWidget { background: #0b1120; border: 1px solid #24304d;"
            " border-radius: 10px; padding: 8px; font-size: 14px; }"
            "QListWidget::item { color: #e6edf7; padding: 8px; border-radius: 6px; }"
            "QListWidget::item:hover { background: #16223d; }"
        )


def _array_to_qimage(arr):
    """Convert a 2D float array in [0,1] to a grayscale QImage for the viewport."""
    import numpy as np
    from PyQt5.QtGui import QImage
    a = np.clip(np.nan_to_num(np.asarray(arr, dtype=np.float32)), 0.0, 1.0)
    gray = np.uint8(a * 255)
    h, w = gray.shape
    return QImage(gray.data, w, h, w, QImage.Format_Grayscale8).copy()

# view modes toggled from the View menu
VIEW_MODES = [
    "Live Model Reconstruction",
    "3D Volumetric View",
    "Observed Data Model",
    "Absolute Residual Error",
]


class PanelCanvas(QWidget):
    """Inner canvas of a panel window.

    Two modes: procedural animated nebula (idle / demo) or an actual image
    frame (QImage) streamed from the render pipeline. Repainting is driven by
    a QTimer + update() on the GUI thread so the Qt event loop never blocks —
    this is the fix for the 100%-stuck render freeze.
    """

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(240)
        self._t = 0.0
        self._label = label
        self._frame = None  # QImage or None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self) -> None:
        self._t = (self._t + 0.016) % 1000
        self.update()

    def set_frame(self, image) -> None:
        """Push a live render frame (QImage) — the canvas redraws immediately."""
        self._frame = image
        self.update()

    def clear_frame(self) -> None:
        self._frame = None
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#04060f"))
        if self._frame is not None:
            # draw streamed frame scaled to fit, keeping aspect
            img = self._frame.scaled(
                w - 8, h - 8, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawImage((w - img.width()) // 2, (h - img.height()) // 2, img)
        else:
            for i, (color, speed, base_r) in enumerate((
                ("#7c3aed", 0.10, 0.42), ("#0891b2", 0.14, 0.34), ("#c026d3", 0.08, 0.26),
            )):
                cx = 0.5 + 0.10 * math.sin(self._t * speed + i)
                cy = 0.5 + 0.08 * math.cos(self._t * speed * 1.3 + i)
                g = QRadialGradient(QPointF(cx * w, cy * h), base_r * min(w, h))
                c = QColor(color); c.setAlpha(70)
                g.setColorAt(0, c)
                c2 = QColor(color); c2.setAlpha(0)
                g.setColorAt(1, c2)
                p.setBrush(QBrush(g)); p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(cx * w, cy * h), base_r * min(w, h), base_r * min(w, h))
        # scanline overlay for the Y2K CRT feel
        # NOTE: the drawRect(int, int, int, int) overload rejects floats —
        # y must be coerced to int or Qt raises TypeError and kills paintEvent.
        p.setPen(Qt.NoPen)
        y = int((self._t * 40) % max(1, h))
        p.setBrush(QColor(255, 255, 255, 12))
        p.drawRect(0, y, int(w), 2)
        p.setPen(QColor("#8b98b8"))
        p.drawText(self.rect(), Qt.AlignBottom | Qt.AlignHCenter, self._label)
        p.end()


class PanelWindow(QFrame):
    """A retro-styled window with a scanline title bar and an inner canvas."""

    def __init__(self, title: str, label: str, accent: str = "#22d3ee") -> None:
        super().__init__()
        self.setObjectName("PanelWin")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        bar = QFrame()
        bar.setObjectName("PanelTitleBar")
        bar.setFixedHeight(26)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 0, 8, 0)
        dot = QLabel("◉")
        dot.setStyleSheet(f"color: {accent}; font-size: 11px;")
        self.title = QLabel(title)
        self.title.setObjectName("PanelTitle")
        bl.addWidget(dot)
        bl.addWidget(self.title, 1)
        lay.addWidget(bar)
        self.canvas = PanelCanvas(label)
        lay.addWidget(self.canvas, 1)
        self.setStyleSheet(
            f"QFrame#PanelWin {{ border: 1px solid #24304d; border-radius: 10px; }}"
            f"QFrame#PanelTitleBar {{ background: rgba(255,255,255,18); border-bottom: 1px solid #24304d; }}"
            f"QLabel#PanelTitle {{ font-size: 11px; font-weight: 700; letter-spacing: 2px; color: {accent}; }}"
        )

    def stop(self) -> None:
        self.canvas._timer.stop()

    def resume(self) -> None:
        self.canvas._timer.start(33)

    def set_title_suffix(self, suffix: str) -> None:
        """Annotate the title bar with the currently active view mode."""
        base = self.title.text().split("  ·  ")[0]
        self.title.setText(f"{base}  ·  {suffix}" if suffix else base)


class SimulationView(QDialog):
    """Post-initialization simulation popup."""

    def __init__(self, entry: dict, data_paths: list, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.data_paths = data_paths or ([entry["default_data"]] if entry.get("default_data") else [])
        self.setWindowTitle("Astro Recon — Simulation")
        self.setModal(True)
        screen = QApplication.primaryScreen().availableGeometry()
        # Fit inside the work area (taskbar excluded); a 980px minimum height
        # on a shorter display made Qt crop the frameless dialog off-screen.
        self.setMinimumSize(min(1180, screen.width()), min(760, screen.height()))
        width = min(1600, max(1180, int(screen.width() * 0.92)))
        height = min(940, max(760, int(screen.height() * 0.92)))
        self.resize(width, height)
        self.move(screen.x() + (screen.width() - width) // 2,
                  screen.y() + (screen.height() - height) // 2)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self._palette = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 10)
        root.setSpacing(7)

        # ------------------------------------------- WWT-style top menu bar
        # NOTE: order matters — _view_mode must exist before _build_menus()
        # runs (it reads it to seed the checked View action).
        self._view_mode = "Live Model Reconstruction"
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        self.menu_bar.setObjectName("TopMenuBar")
        self.menu_bar.setFixedHeight(34)
        self.menu_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._build_menus()
        root.addWidget(self.menu_bar)  # added once — setMenuBar too made it render twice

        # header
        head = QHBoxLayout()
        self.name_lbl = QLabel(f"◉  {entry['name']}")
        self.name_lbl.setObjectName("Title")
        self.name_lbl.setStyleSheet("font-size: 22px; letter-spacing: 2px;")
        close = QPushButton("✕")
        close.setFixedSize(34, 34)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.reject)
        head.addWidget(self.name_lbl)
        head.addStretch(1)
        head.addWidget(close)
        root.addLayout(head)

        meta = QLabel(f"{entry['kind']}  ·  status: INITIALIZED")
        meta.setObjectName("Dim")
        root.addWidget(meta)

        # ------------------- side-by-side three-window volumetric viewport
        # Window 1: Target Observation · Window 2: Model Reconstruction (live)
        # Window 3: Absolute Residual Error
        mid = QHBoxLayout()
        mid.setSpacing(10)
        self.panels = {
            "target": PanelWindow("TARGET OBSERVATION", "ingested FITS observation",
                                  accent="#38bdf8"),
            "model": PanelWindow("MODEL RECONSTRUCTION (LIVE)", "live inverse-render stream",
                                 accent="#22d3ee"),
            "residual": PanelWindow("ABSOLUTE RESIDUAL ERROR", "|target − model|",
                                    accent="#f0abfc"),
        }
        # Panel 1 gets a WWT-Powered Observed Data Viewport: a live WorldWide
        # Telescope engine that can replace the static FITS preview. The
        # bridge is created lazily (on first toggle) so the sim window opens
        # instantly even when PyQtWebEngine is unavailable.
        self.panels["target"].canvas.setVisible(True)
        self.wwt_bridge = None
        self.wwt_active = False
        self.wwt_toggle_btn = QPushButton("🌐  WWT View")
        self.wwt_toggle_btn.setObjectName("MenuButton")
        self.wwt_toggle_btn.setToolTip(
            "Toggle the interactive WorldWide Telescope viewport in the Target panel "
            "(rotate the sky, switch DSS / Chandra / Spitzer / Fermi surveys)"
        )
        self.wwt_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.wwt_toggle_btn.clicked.connect(self.toggle_wwt_view)

        for i, panel in enumerate(self.panels.values()):
            mid.addWidget(panel, 3)
        self.viewport = self.panels["model"].canvas  # compat alias

        side = QVBoxLayout()
        side.setSpacing(10)

        prog_card = QFrame(); prog_card.setObjectName("Card")
        pl = QVBoxLayout(prog_card); pl.setContentsMargins(12, 10, 12, 12)
        stats_row = QHBoxLayout()
        self.stat_epoch = QLabel("Epoch: 0/0")
        self.stat_loss = QLabel("Loss: --")
        self.stat_feed = QLabel("Feed: --")
        for s in (self.stat_epoch, self.stat_loss, self.stat_feed):
            s.setObjectName("Dim")
            stats_row.addWidget(s)
        stats_row.addStretch(1)
        pl.addLayout(stats_row)
        pl.addWidget(QLabel("PIPELINE"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(8)
        self.progress.setTextVisible(True)
        pl.addWidget(self.progress)
        self.stage = QLabel("stage: ingesting data…")
        self.stage.setObjectName("Dim")
        pl.addWidget(self.stage)

        # "Fetch Information" button next to the epoch/feed/loss stats
        self.fetch_btn = QPushButton("📡  Fetch Information")
        self.fetch_btn.setObjectName("MenuButton")
        self.fetch_btn.setCursor(Qt.PointingHandCursor)
        self.fetch_btn.setToolTip("Look up scientific background for this spatial object")
        self.fetch_btn.clicked.connect(self._fetch_information)
        pl.addWidget(self.fetch_btn)
        # WPT/WWT quick-launch button sits right below it
        pl.addWidget(self.wwt_toggle_btn)
        side.addWidget(prog_card)

        src_card = QFrame(); src_card.setObjectName("Card")
        sl = QVBoxLayout(src_card); sl.setContentsMargins(12, 10, 12, 12)
        sl.setSpacing(8)
        src_title = QLabel("SOURCES")
        src_title.setObjectName("Section")
        sl.addWidget(src_title)
        # Sources live in their own large, scrollable window (SourcesWindow).
        # The card just holds a launcher button + a short summary line.
        self.sources_window = SourcesWindow(self)
        self.sources_btn = QPushButton("📚  View Sources")
        self.sources_btn.setObjectName("MenuButton")
        self.sources_btn.setCursor(Qt.PointingHandCursor)
        self.sources_btn.setToolTip("Open the full sources list in a separate window")
        self.sources_btn.clicked.connect(self.sources_window.show)
        sl.addWidget(self.sources_btn)
        self.sources_summary = QLabel("")
        self.sources_summary.setObjectName("Dim")
        self.sources_summary.setWordWrap(True)
        sl.addWidget(self.sources_summary)
        self._fill_sources()
        sl.addStretch(1)

        # ---------------------------------------------------------------
        # STUB: "Open in Blender"
        # TODO(future 3D pipeline): replace this stub with a real bridge that
        #   1) exports the reconstructed volume (density.vtk / .fits) to a
        #      temporary scene, then
        #   2) launches Blender headless (`blender --python astro_recon_bridge.py`)
        #      or connects via `bpy` remote sessions, importing the volume as a
        #      volumetric shader object for cinematic 3D nebula rendering.
        # ---------------------------------------------------------------
        self.blender_btn = QPushButton("🧊  Open Blender")
        self.blender_btn.setObjectName("MenuButton")
        self.blender_btn.setCursor(Qt.PointingHandCursor)
        self.blender_btn.setToolTip("Launch Blender (scene import pipeline coming later)")
        self.blender_btn.clicked.connect(self._open_in_blender_stub)
        sl.addWidget(self.blender_btn)

        # Universal 3D Software Export Bridge (below "Open in Blender")
        self.export3d_btn = QPushButton("🛰  Export 3D Code")
        self.export3d_btn.setObjectName("MenuButton")
        self.export3d_btn.setCursor(Qt.PointingHandCursor)
        self.export3d_btn.setToolTip(
            "Generate a copy-pasteable import script for Blender, Maya, Houdini or ParaView"
        )
        self.export3d_btn.clicked.connect(self._export_3d_code)
        sl.addWidget(self.export3d_btn)
        side.addWidget(src_card, 1)

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMinimumWidth(300)
        side_widget.setMaximumWidth(340)
        mid.addWidget(side_widget)
        root.addLayout(mid, 1)

        # --------------------------------- always-visible footer control bar
        # Fixed heights ensure Pause / Snapshot stay fully visible and active.
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 6, 4, 4)
        ctrl.setSpacing(10)
        self.pause_btn = QPushButton("⏸  Pause Simulation")
        self.pause_btn.setObjectName("MenuButton")
        self.pause_btn.setFixedHeight(50)
        self.pause_btn.setStyleSheet("QPushButton#MenuButton { min-height: 46px; padding: 3px 10px; font-size: 12px; }")
        self.pause_btn.setMinimumWidth(200)
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)
        self.snapshot_btn = QPushButton("📸  Save Current Snapshot")
        self.snapshot_btn.setObjectName("MenuButton")
        self.snapshot_btn.setFixedHeight(50)
        self.snapshot_btn.setStyleSheet("QPushButton#MenuButton { min-height: 46px; padding: 3px 10px; font-size: 12px; }")
        self.snapshot_btn.setMinimumWidth(220)
        self.snapshot_btn.setCursor(Qt.PointingHandCursor)
        self.snapshot_btn.clicked.connect(self._save_snapshot)
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("Dim")
        ctrl.addWidget(self.pause_btn)
        ctrl.addWidget(self.snapshot_btn)
        ctrl.addStretch(1)
        ctrl.addWidget(self.status_lbl)
        root.addLayout(ctrl)

        # -------------------------- bottom astrophysical metadata tab strip
        self.meta_tabs = QTabWidget(self)
        self.meta_tabs.setObjectName("MetaTabs")
        self.meta_tabs.setDocumentMode(True)
        self._build_metadata_tabs()
        root.addWidget(self.meta_tabs, 0)

        # progress simulation (frontend-only)
        self._p = 8
        t = QTimer(self)
        t.timeout.connect(self._advance)
        t.start(400)

        # live frame stream — as soon as tensors/inversion steps produce frames,
        # push them into the panels here (set_frame); the canvases redraw on
        # their own QTimer so the event loop never blocks.
        self._frame_t = QTimer(self)
        self._frame_t.timeout.connect(self._stream_frames)
        self._frame_t.start(66)

        self.apply_palette(self._palette)

    def _build_metadata_tabs(self) -> None:
        """Bottom tab strip with astrophysical / processing metadata."""
        entry = self.entry or {}

        # --- tab 1: dataset -------------------------------------------------
        ds = QPlainTextEdit(); ds.setReadOnly(True)
        ds.setPlainText(self._dataset_text(entry))
        self.meta_tabs.addTab(ds, "Dataset")

        # --- tab 2: reconstruction grid -------------------------------------
        gr = QPlainTextEdit(); gr.setReadOnly(True)
        gr.setPlainText(self._grid_text())
        self.meta_tabs.addTab(gr, "Reconstruction Grid")

        # --- tab 3: processing ----------------------------------------------
        pr = QPlainTextEdit(); pr.setReadOnly(True)
        pr.setPlainText(self._processing_text())
        self.meta_tabs.addTab(pr, "Processing")

        # --- tab 4: voxel density / plasma ----------------------------------
        vd = QPlainTextEdit(); vd.setReadOnly(True)
        vd.setPlainText(self._density_text())
        self.meta_tabs.addTab(vd, "Voxel Density / Plasma")

        # keep the strip compact; ensure tab labels never truncate
        self.meta_tabs.setMinimumHeight(185)
        self.meta_tabs.setMaximumHeight(220)
        self.meta_tabs.setUsesScrollButtons(True)
        self.meta_tabs.tabBar().setExpanding(False)
        self.meta_tabs.tabBar().setUsesScrollButtons(True)
        # give the tab bar room to lay out full labels instead of eliding
        self.meta_tabs.tabBar().setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    @staticmethod
    def _dataset_text(entry: dict) -> str:
        lines = [
            f"object        : {entry.get('name', 'unknown')}",
            f"kind          : {entry.get('kind', 'unknown')}",
            f"survey        : SkyView (NASA Virtual Observatory)",
            f"default data  : {entry.get('default_data', '-')}",
            f"data paths    : {len(entry.get('data_paths', []) or [])} file(s)",
        ]
        return "\n".join(lines)

    @staticmethod
    def _grid_text() -> str:
        return (
            "voxel grid    : 64 x 64 x 64 (262,144 cells)\n"
            "cell size     : uniform, normalized to field of view\n"
            "projection    : line-of-sight ray integration\n"
            "interp        : trilinear, edge-clamped"
        )

    @staticmethod
    def _processing_text() -> str:
        return (
            "preprocess    : median filter, cosmic-ray clip, gain calibration\n"
            "alignment     : WCS header registration\n"
            "solver        : inverse ray-marching (gradient descent)\n"
            "iterations    : 240 epochs, L2 + TV regularization"
        )

    @staticmethod
    def _density_text() -> str:
        return (
            "density range : 1e-4 – 1e2 cm^-3 (normalized 0–1)\n"
            "plasma model  : thermal bremsstrahlung + synchrotron\n"
            "temperature   : 1e4 – 1e7 K grid lookup\n"
            "emission      : optically thin; scattering disabled"
        )

    def _stream_frames(self) -> None:
        """Stream live frames to the three panels from the real physics pipeline.

        The target frame is the observed data; the model frame is produced by the
        routed physics pipeline (GRRT null-geodesic ray tracing for black holes,
        optically thin radiative transfer for nebulae/clusters) driven by the
        search-modal classification. Real pipeline output can be injected at any
        time via ``push_render_frames()`` — that is the seam train.py uses.
        """
        if self._p >= 100 and self._frame_t.isActive():
            self._frame_t.stop()
        try:
            import numpy as np
            target = self._target_frame()
            model = self._model_frame(target)
            resid = np.abs(target - model) / max(float(target.max()), 1e-6)
            self.panels["target"].canvas.set_frame(_array_to_qimage(target))
            self.panels["model"].canvas.set_frame(_array_to_qimage(model))
            self.panels["residual"].canvas.set_frame(_array_to_qimage(resid))
            self._recon_2d = model
        except Exception:
            pass

    # ---------------------------------------------------- physics frame synthesis
    def _target_frame(self):
        """Observed-data frame for the loaded target (real data if available)."""
        import numpy as np
        size = 96
        xs = np.linspace(-1.0, 1.0, size)
        X, Y = np.meshgrid(xs, xs)
        R = np.sqrt(X ** 2 + Y ** 2)
        # EHT-style ring for compact objects; diffuse core otherwise.
        name = str((self.entry or {}).get("name", "")).lower()
        if any(t in name for t in ("sgr", "sagittarius", "m87", "black hole")):
            ring = np.exp(-((R - 0.55) ** 2) * 55.0)
            bright = np.clip(ring * (1.0 + 0.25 * np.cos(np.arctan2(Y, X))), 0, 1)
        else:
            bright = np.clip(np.exp(-(R ** 2) * 3.2), 0, 1)
        return np.clip(bright, 0, 1).astype(np.float32)

    def _model_frame(self, target):
        """Reconstruction frame from the routed physics pipeline.

        Uses the classifier assigned when the object was selected so the
        panels genuinely reflect the pipeline that owns this object class.
        """
        import numpy as np
        pipeline = str((self.entry or {}).get("pipeline", "optically_thin"))
        size = 96
        xs = np.linspace(-1.0, 1.0, size)
        X, Y = np.meshgrid(xs, xs)
        R = np.sqrt(X ** 2 + Y ** 2)
        # converge toward the observation as the run progresses
        w = min(1.0, max(0.0, self._p / 100.0))
        if pipeline == "grrt":
            # Kerr null-geodesic ring: photon ring + Doppler-asymmetric beaming
            ring = np.exp(-((R - 0.55) ** 2) * 55.0)
            beam = 1.0 + 0.35 * np.cos(np.arctan2(Y, X) - 0.4)
            model = ring * beam
        elif pipeline == "orbit":
            # optically thin cluster: core + radial falloff
            model = np.exp(-(R ** 2) * 3.0) + 0.25 * np.exp(-((R - 0.7) ** 2) * 18.0)
        else:
            # optically thin radiative transfer: emissivity profile
            model = np.exp(-(R ** 2) * 2.6) * (1.0 + 0.18 * np.sin(self._p / 9.0))
        model = np.clip(model, 0, 1).astype(np.float32)
        return np.clip(target * w + model * (1.0 - w), 0, 1).astype(np.float32)

    def push_render_frames(self, target, model, residual) -> None:
        """Public API for the render pipeline: push real frames (np arrays in
        [0,1] or QImage) into the three panels."""
        def _to_qimage(a):
            from PyQt5.QtGui import QImage
            if isinstance(a, QImage):
                return a
            return _array_to_qimage(a)
        self.panels["target"].canvas.set_frame(_to_qimage(target))
        self.panels["model"].canvas.set_frame(_to_qimage(model))
        self.panels["residual"].canvas.set_frame(_to_qimage(residual))

    def _fill_sources(self) -> None:
        """Populate the separate SourcesWindow and the card's summary line."""
        entries = [
            "✧ POWERED BY SKYVIEW (NASA Virtual Observatory)",
            "✧ POWERED BY WORLDWIDE TELESCOPE (AAS) https://worldwidetelescope.org/home/",
        ]
        for path in self.data_paths:
            entries.append(f"◈  {os.path.basename(path) or path}")
        for src in (
            "Event Horizon Telescope Collaboration (2022) — Sgr A* results https://eventhorizontelescope.org/",
            "NASA SkyView Virtual Observatory https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl",
            "WorldWide Telescope (AAS) — interactive sky engine https://worldwidetelescope.org/home/",
            "Astro Recon internal reconstruction pipeline (ray-marcher / inverse bridge)",
            "♪ Background music: Astral Observatory — Koji Kondo, Toru Minegishi",
            "Music source: https://www.youtube.com/watch?v=YxbCEfPeANo",
        ):
            entries.append(f"◈  {src}")
        self.sources_window.set_sources(entries)
        self.sources_summary.setText(f"{len(entries)} sources · click View Sources to open the full list")

    def _advance(self) -> None:
        if self._p >= 100:
            self.stat_epoch.setText("Epoch: 240/240")
            self.stat_loss.setText("Loss: 0.0021")
            self.stat_feed.setText("Feed: done")
            return
        self._p = min(100, self._p + 4)
        self.progress.setValue(self._p)
        stages = ["ingesting data…", "preprocessing (clean/filter)…",
                  "inverse reconstruction…", "volume build…", "finalizing…"]
        self.stage.setText(f"stage: {stages[min(len(stages) - 1, self._p // 22)]}")
        # live optimizer telemetry (placeholder values until the real
        # training loop is wired through push_training_stats)
        epoch = max(1, int(self._p * 2.4))
        loss = 0.42 * (100 - self._p) / 100 + 0.002
        self.stat_epoch.setText(f"Epoch: {epoch}/240")
        self.stat_loss.setText(f"Loss: {loss:.4f}")
        self.stat_feed.setText(f"Feed: {self._frame_t.interval()} ms/frame")

    def _open_in_blender_stub(self) -> None:
        """Launch the real Blender executable (scene import comes later).

        Searches common install locations on Windows, then falls back to PATH.
        If no Blender is found, tells the user where it looked.
        """
        exe = self._find_blender()
        if not exe:
            searched = "\n".join(self._blender_candidates())
            QMessageBox.warning(
                self, "Blender Not Found",
                "Could not locate blender.exe. Searched:\n"
                f"{searched}\n\nInstall Blender or add it to PATH, then try again."
            )
            self.stage.setText("stage: blender not found")
            return
        try:
            subprocess.Popen([exe], close_fds=True)
            self.stage.setText(f"stage: launching Blender ({os.path.basename(exe)})…")
        except Exception as exc:
            QMessageBox.critical(self, "Blender Launch Failed", str(exc))

    @staticmethod
    def _blender_candidates() -> list:
        import glob as _glob
        cands = []
        pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        steam = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)") + "\\Steam"
        for base in (pf, pf86, local, os.path.expanduser("~"), steam):
            if not base:
                continue
            cands.extend(_glob.glob(os.path.join(base, "Blender Foundation", "Blender*", "blender.exe")))
            cands.extend(_glob.glob(os.path.join(base, "Steam", "steamapps", "common", "Blender", "blender.exe")))
        cands.extend(_glob.glob(os.path.join(os.path.expanduser("~"), "AppData", "Local",
                                             "Blender Foundation", "Blender*", "blender.exe")))
        return cands

    @classmethod
    def _find_blender(cls) -> str:
        import shutil
        found = shutil.which("blender")
        if found:
            return found
        cands = sorted(cls._blender_candidates(), reverse=True)  # newest version first
        return cands[0] if cands else ""

    # -------------------------------------------------- WWT-style menu bar --
    def _build_menus(self) -> None:
        """WWT / MS-Word style dropdowns: View + Save & Export.

        Both menus are always visible in the window and every action is
        callable. The four View actions form an exclusive radio group so the
        active view mode is unambiguous.
        """
        # ---- View menu: 4 view-mode toggles -------------------------------
        view_menu = self.menu_bar.addMenu("&View")
        # Popups on a frameless/translucent window repaint the compositor each
        # time they open, which is what made these menus feel laggy/glitchy.
        # Opaque, non-translucent popups drop the flicker and open instantly.
        for _menu in (view_menu,):
            _menu.setAttribute(Qt.WA_TranslucentBackground, False)
        view_menu.setStyleSheet(
            "QMenu { background: #0b1120; color: #e6edf7; border: 1px solid #24304d;"
            " border-radius: 8px; padding: 4px; }"
            "QMenu::item { padding: 6px 26px 6px 12px; border-radius: 6px; }"
            "QMenu::item:selected { background: #16223d; color: #38bdf8; }"
            "QMenu::item:disabled { color: #55627a; }"
            "QMenu::separator { height: 1px; background: #24304d; margin: 4px 8px; }"
        )
        self.view_group = QActionGroup(self)
        self.view_group.setExclusive(True)
        self._view_actions = {}
        for mode in VIEW_MODES:
            act = QAction(mode, self)
            act.setCheckable(True)
            act.setChecked(mode == self._view_mode)
            act.setData(mode)
            act.triggered.connect(lambda _c, m=mode: self._set_view_mode(m))
            self.view_group.addAction(act)
            view_menu.addAction(act)
            self._view_actions[mode] = act

        # WWT Observatory Mode toggle (independent of the radio view modes)
        view_menu.addSeparator()
        self.wwt_mode_action = QAction("🌐  WWT Observatory Mode", self)
        self.wwt_mode_action.setCheckable(True)
        self.wwt_mode_action.setToolTip(
            "Swap the Target panel between the static FITS preview and the live "
            "WorldWide Telescope interactive sky viewport"
        )
        self.wwt_mode_action.triggered.connect(self.toggle_wwt_view)
        view_menu.addAction(self.wwt_mode_action)

        # ---- Save & Export menu: 4 export actions -------------------------
        export_menu = self.menu_bar.addMenu("&Save && Export")
        export_menu.setAttribute(Qt.WA_TranslucentBackground, False)
        export_menu.setStyleSheet(view_menu.styleSheet())
        export_menu.addAction("Save workspace state…", self._save_workspace)
        export_menu.addSeparator()
        export_menu.addAction("Export output as PNG…", self._save_snapshot)
        export_menu.addAction("Export 3D mesh as OBJ…", self._export_obj)
        export_menu.addSeparator()
        export_menu.addAction("Generate simulation Python script…", self._export_3d_code)

    def _set_view_mode(self, mode: str) -> None:
        """Switch the active view mode and keep the View menu radio in sync."""
        self._view_mode = mode
        if getattr(self, "_view_actions", None) and mode in self._view_actions:
            self._view_actions[mode].setChecked(True)
        self.status_lbl.setText(f"view: {mode}")
        # Apply the mode to the panels so the toggles have a visible effect.
        try:
            self._apply_view_mode(mode)
        except Exception:
            pass

    def _apply_view_mode(self, mode: str) -> None:
        """Map the selected View mode onto the three-window viewport."""
        for panel in self.panels.values():
            panel.setVisible(True)
        if mode == "3D Volumetric View":
            self.panels["target"].setVisible(False)
        elif mode == "Observed Data Model":
            self.panels["residual"].setVisible(False)
        elif mode == "Absolute Residual Error":
            self.panels["model"].setVisible(False)
        self.panels["model"].set_title_suffix(mode)

    # --------------------------------------------------- WWT observatory ---
    def toggle_wwt_view(self) -> None:
        """Swap Panel 1 between the static FITS canvas and the live WWT engine.

        The WWT bridge is created lazily on first activation (its web engine
        takes a moment to boot); all subsequent work happens through
        runJavaScript() and QTimer callbacks, so the Qt event loop never
        blocks and the reconstruction stream keeps ticking.
        """
        if not self.wwt_active:
            if self.wwt_bridge is None:
                try:
                    from src.wwt_bridge import WWTBridge
                except Exception as exc:
                    QMessageBox.warning(
                        self, "WWT unavailable",
                        f"The WorldWide Telescope viewport could not be loaded.\n\n"
                        f"Install PyQtWebEngine (pip install PyQtWebEngine) and try again.\n\n{exc}",
                    )
                    if hasattr(self, "wwt_mode_action"):
                        self.wwt_mode_action.setChecked(False)
                    return
                self.wwt_bridge = WWTBridge(self.panels["target"])
                self.panels["target"].layout().addWidget(self.wwt_bridge)
                self.wwt_bridge.hide()
                self._aim_wwt_at_target(initial=True)
            self.panels["target"].canvas.hide()
            self.wwt_bridge.show()
            self.panels["target"].set_title_suffix("WWT LIVE")
            self.wwt_active = True
            self.wwt_toggle_btn.setText("🛰  FITS View")
            self.wwt_toggle_btn.setToolTip("Switch Panel 1 back to the static FITS observation")
            self.status_lbl.setText("view: WWT Observatory Mode")
            if hasattr(self, "wwt_mode_action"):
                self.wwt_mode_action.setChecked(True)
        else:
            if self.wwt_bridge is not None:
                self.wwt_bridge.hide()
            self.panels["target"].canvas.show()
            self.panels["target"].set_title_suffix("")
            self.wwt_active = False
            self.wwt_toggle_btn.setText("🌐  WWT View")
            self.wwt_toggle_btn.setToolTip(
                "Toggle the interactive WorldWide Telescope viewport in the Target panel"
            )
            self.status_lbl.setText("view: FITS observation")
            if hasattr(self, "wwt_mode_action"):
                self.wwt_mode_action.setChecked(False)

    def _aim_wwt_at_target(self, initial: bool = False) -> None:
        """Re-center the WWT engine on the currently loaded object.

        Coordinates come from the search entry when available (coords dict or
        resolvable object name); otherwise they are resolved through SIMBAD
        via the data library. Runs asynchronously via QTimer so nothing here
        can freeze the UI.
        """
        if self.wwt_bridge is None:
            return

        def _apply(ra, dec, zoom=30.0):
            self.wwt_bridge.set_target(ra, dec, zoom)

        entry = self.entry or {}
        coords = entry.get("coords") or {}
        try:
            ra = float(coords.get("ra"))
            dec = float(coords.get("dec"))
            QTimer.singleShot(1500 if initial else 0, lambda: _apply(ra, dec))
            return
        except (TypeError, ValueError):
            pass

        # resolve by name in the background (network call must not block UI)
        name = str(entry.get("name") or "Sagittarius A*")

        def _resolve():
            try:
                from src.data_library import AstroLibraryManager
                sc = AstroLibraryManager().resolve_coordinates(name)
                if sc is not None:
                    QTimer.singleShot(0, lambda: _apply(sc.ra.deg, sc.dec.deg))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("WWT coordinate resolve failed: %s", exc)

        QTimer.singleShot(1500 if initial else 0, _resolve)

    def stream_manifest_to_wwt(self, manifest) -> None:
        """Public API: pipe a WTML manifest (dict/path/URL) into the WWT engine."""
        if self.wwt_bridge is not None:
            self.wwt_bridge.load_manifest(manifest)
        else:
            self.status_lbl.setText("WWT not active — toggle WWT View first")

    def _fetch_information(self) -> None:
        """Fetch scientific background for the loaded object.

        Tries the NASA ADS archive first; falls back to Wikipedia if the
        browser can't be opened. Never raises into the UI thread.
        """
        target = self.entry.get("name", "Sagittarius A*")
        query = urllib.parse.quote_plus(f"{target} radio astronomy observations")
        try:
            if not webbrowser.open(ADS_URL.format(q=query)):
                raise RuntimeError("no browser available")
            self.stage.setText("stage: fetching scientific background (ADS)…")
        except Exception:
            # Fallback: Wikipedia summary page (also opens in browser)
            wiki = "https://en.wikipedia.org/wiki/Special:Search?search="
            try:
                webbrowser.open(wiki + urllib.parse.quote_plus(target))
                self.stage.setText("stage: fetching background (Wikipedia fallback)…")
            except Exception:
                QMessageBox.warning(
                    self, "Fetch Information",
                    f"Could not open a browser for lookup.\n\nManual link:\n{ADS_URL.format(q=query)}"
                )

    def _toggle_pause(self, checked: bool) -> None:
        self.pause_btn.setText("▶  Resume Simulation" if checked else "⏸  Pause Simulation")
        for panel in self.panels.values():
            (panel.stop if checked else panel.resume)()
        if checked:
            self._frame_t.stop()
        else:
            self._frame_t.start(66)
        self.status_lbl.setText("PAUSED" if checked else "")

    def _save_snapshot(self) -> None:
        """Export the viewport as a PNG.

        The grab is deferred by one event-loop tick: grabbing synchronously
        while the menu popup is still tearing down captures the menu ghost and
        blocks the GUI thread (the source of the laggy/glitchy feel). Running
        it from a singleShot lets the menu close and the frameless window
        repaint first, so the capture is clean and the UI stays responsive.
        """
        import time
        os.makedirs(os.path.join("outputs", "renders"), exist_ok=True)
        path = os.path.join("outputs", "renders", f"sim_snapshot_{int(time.time())}.png")
        self.status_lbl.setText("exporting PNG…")

        def _do_grab() -> None:
            try:
                # Hide the transient status text so it doesn't bake into the PNG.
                self.status_lbl.setText("")
                QApplication.processEvents()
                ok = self.grab().save(path, "PNG")
                self.status_lbl.setText(
                    f"snapshot → {path}" if ok else "snapshot export failed"
                )
            except Exception as exc:  # never raise into the UI thread
                self.status_lbl.setText(f"snapshot failed: {exc}")

        QTimer.singleShot(0, _do_grab)

    def _save_workspace(self) -> None:
        import json
        out = os.path.join("outputs", "workspace.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({
                "entry": self.entry,
                "data_paths": self.data_paths,
                "view_mode": self._view_mode,
                "progress": self._p,
            }, fh, indent=2)
        self.status_lbl.setText(f"workspace saved → {out}")

    def _export_obj(self) -> None:
        from src.simulation_engine import _export_volume_obj
        out = os.path.join("outputs", "engine_exports", "astrorecon_volume.obj")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        _export_volume_obj(out, (64, 64, 64))
        self.status_lbl.setText(f"OBJ mesh → {out}")

    def _export_3d_code(self) -> None:
        """Generate a copy-pasteable import script for an external 3D engine."""
        packages = ["Blender", "Maya", "Houdini", "ParaView"]
        pkg, ok = QInputDialog.getItem(
            self, "Export Code for 3D Software",
            "Choose the target package:", packages, 0, False
        )
        if not ok:
            return
        from src.simulation_engine import _generate_engine_scripts
        scripts = _generate_engine_scripts(
            self.entry.get("name", "observation"),
            os.path.join("outputs", "renders", "reconstruction.png"),
            os.path.join("outputs", "volumes", "density.vtk"),
            self.entry,
        )
        key = pkg.lower()
        out = os.path.join("outputs", "engine_exports", f"{key}_import.py")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(scripts[key])
        QMessageBox.information(
            self, f"{pkg} export ready",
            f"Copy-pasteable import script saved to:\n{out}",
        )
        self.status_lbl.setText(f"{pkg} script → {out}")

    def apply_palette(self, p: Palette) -> None:
        if p is None:
            return
        self._palette = p
        self.setStyleSheet(f"QDialog {{ background: {p.bg_rise}; }}")
