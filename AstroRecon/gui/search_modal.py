"""Object library search modal + New/Upload (NU) fallback dropzone.

Preset library ships with Sagittarius A* and other classic targets. When a
search finds nothing, the NU panel appears so users can drop raw data
(.fits / .uvfits / folders) into the simulation pipeline.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QFrame, QFileDialog, QWidget,
)

from .theme import Palette

try:
    from ..src.physics_dispatcher import dispatch as dispatch_physics
except ImportError:  # running as top-level package
    from src.physics_dispatcher import dispatch as dispatch_physics

# ------------------------------------------------------------- library ------

LIBRARY = [
    {
        "name": "Sagittarius A*",
        "kind": "Supermassive Black Hole",
        "detail": "Sgr A* · Milky Way Galactic Center · EHT 2022 image data",
        "tags": "sgr a sgra sagittarius black hole galactic center eht milky way",
        "default_data": "data/raw/sgr_a_real.fits",
    },
    {
        "name": "Messier 13 (Hercules Cluster)",
        "kind": "Globular Cluster",
        "detail": "M13 · ~30k stars · sample FITS frame included",
        "tags": "m13 messier hercules globular cluster",
        "default_data": "data/raw/M13_sample.fits",
    },
    {
        "name": "Orion Nebula (M42)",
        "kind": "H II Star-Forming Nebula",
        "detail": "3D volumetric nebula demo target",
        "tags": "orion m42 nebula star forming",
        "default_data": None,
    },
    {
        "name": "Crab Nebula (M1)",
        "kind": "Supernova Remnant + Pulsar",
        "detail": "SN 1054 remnant · volumetric filament test",
        "tags": "crab m1 supernova remnant pulsar",
        "default_data": None,
    },
    {
        "name": "M87* (Virgo A)",
        "kind": "Supermassive Black Hole",
        "detail": "First-ever EHT black hole image (2019)",
        "tags": "m87 virgo black hole eht jet",
        "default_data": None,
    },
    {
        "name": "Cygnus X-1",
        "kind": "Stellar-Mass Black Hole X-ray Binary",
        "detail": "Orbit simulation demo target (see orbit_data.csv)",
        "tags": "cygnus x1 binary orbit xray",
        "default_data": "orbit_data.csv",
    },
]

SUPPORTED_UPLOAD_EXTS = (".fits", ".fit", ".uvfits", ".tar", ".tgz", ".npy", ".csv")


class Dropzone(QFrame):
    """Drag-&-drop / click-to-browse upload zone for the NU fallback."""

    files_chosen = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Dropzone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self._icon = QLabel("⇪")
        self._icon.setAlignment(Qt.AlignCenter)
        f = self._icon.font(); f.setPixelSize(34); self._icon.setFont(f)
        self._text = QLabel("Drop raw data here  ·  .fits · .uvfits · .npy · .csv · folders")
        self._text.setObjectName("Dim")
        self._text.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._icon)
        lay.addWidget(self._text)

        self.setCursor(Qt.PointingHandCursor)

    # drag & drop ---------------------------------------------------------
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("dragOver", True)
            self._refresh()

    def dragLeaveEvent(self, _e):  # noqa: N802
        self.setProperty("dragOver", False)
        self._refresh()

    def dropEvent(self, e):  # noqa: N802
        self.setProperty("dragOver", False)
        self._refresh()
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        if paths:
            self.files_chosen.emit(paths)

    def mousePressEvent(self, _e):  # noqa: N802
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select raw data files", "", "Astronomy data (*.fits *.fit *.uvfits *.npy *.csv *.tgz *.tar);;All files (*)"
        )
        if paths:
            self.files_chosen.emit(paths)

    def _refresh(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)


class SearchModal(QDialog):
    """Interactive search bar modal over the object library, with NU fallback."""

    object_selected = pyqtSignal(dict)   # library entry dict
    nu_upload = pyqtSignal(list)         # list of local file/folder paths

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Astro Recon — Object Library")
        self.setModal(True)
        self.setMinimumSize(640, 620)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self._palette = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        t = QLabel("◈  OBJECT LIBRARY")
        t.setObjectName("Section")
        close = QPushButton("✕")
        close.setFixedSize(34, 34)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.reject)
        header.addWidget(t)
        header.addStretch(1)
        header.addWidget(close)
        root.addLayout(header)

        # --- search bar ---
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search spatial objects…  (e.g. 'Sagittarius A*', 'nebula', 'M13')")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search)
        self.search.returnPressed.connect(self._activate_first)
        root.addWidget(self.search)

        # --- results list ---
        self.results = QListWidget()
        self.results.itemDoubleClicked.connect(self._activate_item)
        root.addWidget(self.results, 1)

        # --- NU fallback dropzone (hidden until search misses) ---
        self.nu_box = QFrame()
        self.nu_box.setObjectName("Card")
        nu_lay = QVBoxLayout(self.nu_box)
        nu_lay.setContentsMargins(14, 12, 14, 14)
        nu_lay.setSpacing(8)
        nu_title = QLabel("NEW / UPLOAD (NU) — not in library?")
        nu_title.setObjectName("Section")
        nu_hint = QLabel("Feed raw data (FITS, UVFITS, NP volumes, folders) straight into the simulation pipeline.")
        nu_hint.setObjectName("Dim")
        nu_hint.setWordWrap(True)
        self.dropzone = Dropzone()
        self.dropzone.files_chosen.connect(self._on_upload)
        nu_lay.addWidget(nu_title)
        nu_lay.addWidget(nu_hint)
        nu_lay.addWidget(self.dropzone)
        self.nu_box.hide()
        root.addWidget(self.nu_box)

        footer = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("Dim")
        browse = QPushButton("NU · Browse files / folder")
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(self._browse_any)
        footer.addWidget(self.status)
        footer.addStretch(1)
        footer.addWidget(browse)
        root.addLayout(footer)

        self._populate("")
        self._theme(self._palette)

    # ------------------------------------------------------- behavior ----
    def _populate(self, query: str) -> None:
        self.results.clear()
        q = query.strip().lower()
        hits = [e for e in LIBRARY if not q or q in e["tags"] or q in e["name"].lower()]
        for e in hits:
            item = QListWidgetItem(f"★  {e['name']}   ·   {e['kind']}\n      {e['detail']}")
            item.setData(Qt.UserRole, e)
            self.results.addItem(item)
        self.nu_box.setVisible(bool(q) and not hits)
        self.status.setText(
            f"{len(hits)} result(s)" if hits or not q
            else "No match in library — use NU to upload raw data."
        )

    def _on_search(self, text: str) -> None:
        self._populate(text)

    def _activate_first(self) -> None:
        if self.results.count():
            self._activate_item(self.results.item(0))

    def _activate_item(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.UserRole)
        # Physics dispatcher hook: classify the object and route to the right
        # rendering pipeline (Track B GRRT vs optically-thin radiative transfer).
        route = dispatch_physics(entry=entry)
        entry = dict(entry)
        entry["pipeline"] = route.pipeline
        entry["pipeline_track"] = route.track
        entry["object_class"] = route.object_class
        entry["pipeline_params"] = route.params
        self.status.setText(f"routed → {route.track}")
        self.object_selected.emit(entry)
        self.accept()

    def _browse_any(self) -> None:
        # folder OR file — folders are handed straight through as paths
        path = QFileDialog.getExistingDirectory(self, "Select data folder (NU)")
        if not path:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select data files (NU)", "",
                "Astronomy data (*.fits *.fit *.uvfits *.npy *.csv *.tgz *.tar);;All files (*)",
            )
            paths = paths
        else:
            paths = [path]
        if paths:
            self._on_upload(paths)

    def _on_upload(self, paths: list) -> None:
        valid = [p for p in paths if os.path.splitext(p)[1].lower() in SUPPORTED_UPLOAD_EXTS or os.path.isdir(p)]
        if not valid:
            self.status.setText("Unsupported format — use .fits / .uvfits / .npy / .csv or a folder.")
            return
        self.nu_upload.emit(valid)
        self.accept()

    # ------------------------------------------------------------ theming
    def _theme(self, p: Palette) -> None:
        self._palette = p
        if p is None:
            return
        self.setStyleSheet("")  # window inherits app stylesheet; fine-tune accents:
        # window backdrop uses the rise surface via QSS on QDialog
        self.setStyleSheet(f"""
            QDialog {{ background: {p.bg_rise}; border-radius: 16px; }}
        """)

    def apply_palette(self, p: Palette) -> None:
        self._theme(p)
