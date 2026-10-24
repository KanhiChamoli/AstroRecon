"""Multi-Epoch Selection & Batch Combination Modal for AstroRecon.

A modular, reusable PyQt5 dialog that lets the user select multiple
observation epochs of a dynamic astrophysical source (e.g. Sagittarius A*'s
orbiting hot spots) and batch-combine them for dynamical tomography.

Design
------
* **Fully data-driven**: the dialog is built from a metadata configuration
  (list of epoch dicts, or a metadata JSON path / dict) so it adapts to any
  target or simulation -- nothing about Sgr A* is hardcoded in the widget.
  Out of the box it defaults to the 2017 EHT Sgr A* campaign epochs:
  Day 095 (April 5), Day 096 (April 6), Day 097 (April 7).
* **Y2K retro-futuristic styling**: deep-space background (#0b0b16),
  electric cyan borders (#38bdf8), terminal green text (#00ff66).
* **Backend hook**: the primary action button calls
  :func:`combine_epochs` (importable standalone) which resolves the
  ``.uvfits`` / ``.fits`` files for the selected days under ``data/raw/``,
  validates them, and returns a manifest dict ready for the multi-epoch
  spatial-temporal optimization pipeline.

Typical usage from the viewport
-------------------------------
>>> from src.ui_epochs import EpochSelectionDialog, combine_epochs
>>> dlg = EpochSelectionDialog(parent=self.viewport, metadata=self.metadata)
>>> if dlg.exec_() == EpochSelectionDialog.Accepted:
...     manifest = dlg.result_manifest   # selected days + resolved file paths

Standalone (no GUI) use:

>>> from src.ui_epochs import combine_epochs
>>> manifest = combine_epochs(["Day 095", "Day 097"])
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:  # static-analysis only; PyQt5 never imported at runtime here
    from PyQt5.QtWidgets import QCheckBox  # noqa: F401

from .log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- constants --

#: AstroRecon Y2K palette
Y2K_BG = "#0b0b16"          # deep space background
Y2K_BORDER = "#38bdf8"      # electric cyan borders / accents
Y2K_TEXT = "#00ff66"        # terminal green text
Y2K_DIM = "#0e7a3f"         # dimmed terminal green
Y2K_WARN = "#ffb347"        # amber warnings
Y2K_PANEL = "#101024"       # elevated panel surface

#: Project-root-relative raw data directory
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

#: Accepted data product extensions (uvfits visibility tables + FITS images)
DATA_EXTENSIONS = (".uvfits", ".fits", ".fit", ".uvfits.gz")

#: Default configuration: 2017 EHT Sagittarius A* observational campaign.
#: 3C 279 was observed in the same window; Sgr A* scan windows are noted.
DEFAULT_EPOCH_METADATA: Dict[str, Any] = {
    "target": "Sagittarius A*",
    "campaign": "EHT 2017 April observational campaign",
    "description": (
        "Dynamical tomography epochs for the Milky Way Galactic Center. "
        "Orbiting hot spot candidates peak near 230 GHz; select multiple "
        "epochs to enable spatial-temporal reconstruction."
    ),
    "epochs": [
        {
            "id": "Day 095",
            "label": "Day 095 (April 5)",
            "mjd": 57848.0,
            "file": "sgra_day095.uvfits",
            "state": "flare rise",
            "tooltip": "Day 095 (2017-04-05, MJD 57848): flare rise phase - "
                       "orbiting hot spot inbound; strong short-baseline "
                       "variability.",
            "default_selected": True,
        },
        {
            "id": "Day 096",
            "label": "Day 096 (April 6)",
            "mjd": 57849.0,
            "file": "sgra_day096.uvfits",
            "state": "flare peak",
            "tooltip": "Day 096 (2017-04-06, MJD 57849): flare peak - "
                       "maximum compact emission; ideal for tomography "
                       "anchor frame.",
            "default_selected": True,
        },
        {
            "id": "Day 097",
            "label": "Day 097 (April 7)",
            "mjd": 57850.0,
            "file": "sgra_day097.uvfits",
            "state": "decay / baseline",
            "tooltip": "Day 097 (2017-04-07, MJD 57850): decay / baseline "
                       "state - symmetric ring structure; good reference "
                       "epoch.",
            "default_selected": False,
        },
    ],
}

# Metadata keys understood per epoch entry (with aliases so external
# metadata JSONs written by other tools still work):
_EPOCH_KEY_ALIASES = {
    "id": ("id", "day", "name", "epoch_id"),
    "label": ("label", "display", "title"),
    "mjd": ("mjd", "modified_julian_date"),
    "file": ("file", "path", "uvfits", "fits", "filename", "data"),
    "state": ("state", "phase", "physical_state"),
    "tooltip": ("tooltip", "description", "notes", "info"),
    "default_selected": ("default_selected", "selected", "default"),
}

QSS = f"""
QWidget#EpochDialogRoot {{
    background-color: {Y2K_BG};
    border: 2px solid {Y2K_BORDER};
    border-radius: 8px;
}}
QLabel#EpochTitle {{
    color: {Y2K_BORDER};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QLabel#EpochSubtitle {{
    color: {Y2K_TEXT};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}}
QLabel#EpochFooter {{
    color: {Y2K_DIM};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 10px;
}}
QLabel#EpochStatus {{
    color: {Y2K_WARN};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}}
QCheckBox {{
    color: {Y2K_TEXT};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    spacing: 8px;
    padding: 4px 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {Y2K_BORDER};
    background: {Y2K_PANEL};
}}
QCheckBox::indicator:checked {{
    background: {Y2K_BORDER};
}}
QCheckBox:hover {{
    color: {Y2K_BORDER};
}}
QPushButton#EpochPrimary {{
    background-color: {Y2K_PANEL};
    color: {Y2K_TEXT};
    border: 2px solid {Y2K_BORDER};
    border-radius: 4px;
    padding: 8px 18px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton#EpochPrimary:hover {{
    background-color: {Y2K_BORDER};
    color: {Y2K_BG};
}}
QPushButton#EpochSecondary {{
    background-color: transparent;
    color: {Y2K_DIM};
    border: 1px solid {Y2K_DIM};
    border-radius: 4px;
    padding: 8px 14px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}}
QPushButton#EpochSecondary:hover {{
    color: {Y2K_TEXT};
    border-color: {Y2K_BORDER};
}}
QFrame#EpochDivider {{
    background-color: {Y2K_BORDER};
    max-height: 1px;
    border: none;
}}
QScrollArea {{
    border: 1px solid {Y2K_BORDER};
    background: {Y2K_PANEL};
}}
"""


# ---------------------------------------------------------------- backend ----

def _candidate_files(raw_dir: Path, epoch: Dict[str, Any]) -> List[Path]:
    """Return plausible data files for *epoch*, best match first.

    Resolution order:
    1. Explicit ``file``/``path`` entry in the metadata (relative to the
       project root or raw dir; absolute paths pass through).
    2. Any ``*.uvfits`` / ``*.fits`` in ``raw_dir`` whose filename contains
       the epoch id (e.g. ``"Day 095"`` -> ``sgra_day095.uvfits``).
    """
    candidates: List[Path] = []
    explicit = epoch.get("file") or epoch.get("path")
    if explicit:
        p = Path(str(explicit))
        if not p.is_absolute():
            for base in (raw_dir, raw_dir.parent):
                probe = base / p
                if probe.is_file():
                    candidates.append(probe)
                    break
            else:
                candidates.append(raw_dir / p)  # best guess even if missing
        else:
            candidates.append(p)

    # Fuzzy match: epoch id -> slug plus any numeric tokens ("Day 095" ->
    # "day095" and "095") matched against flattened filenames recursively.
    slug = str(epoch.get("id", "")).lower().replace(" ", "").replace("_", "")
    tokens = [slug] if slug else []
    digits = re.findall(r"\d{2,4}", str(epoch.get("id", "")))
    tokens += [d for d in digits if len(d) >= 3]
    if tokens:
        try:
            hits = [p for ext in DATA_EXTENSIONS for p in raw_dir.rglob(f"*{ext}")]
        except OSError as exc:
            logger.warning("cannot scan %s: %s", raw_dir, exc)
            hits = []
        for hit in hits:
            stem = hit.name.lower()
            stem = re.sub(r"[\s_\-.]+", "", stem)
            if any(tok in stem for tok in tokens):
                candidates.append(hit)
    return candidates


def combine_epochs(
    selected_days: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
    raw_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Gather and validate the data files for the selected observation epochs.

    This is the backend hook invoked by the modal
    Selected Epochs"** button.  It resolves the ``.uvfits`` / ``.fits``
    paths under ``data/raw/`` for each selected day and returns a manifest
    ready for multi-epoch spatial-temporal optimization.

    Parameters
    ----------
    selected_days:
        Epoch identifiers exactly as shown in the modal (e.g.
        ``["Day 095", "Day 097"]``).  Also accepts labels ("Day 095
        (April 5)") -- matching is lenient.
    metadata:
        Epoch configuration; defaults to :data:`DEFAULT_EPOCH_METADATA`.
        Accepts either ``{"epochs": [...]}`` or a bare list of epoch dicts.
    raw_dir:
        Directory searched for data files.  Defaults to ``data/raw/``.

    Returns
    -------
    dict
        Manifest with ``selected``, ``epochs`` (per-day resolved metadata
        incl. ``path`` and ``exists``), ``missing``, and ``ready`` flag.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    meta = metadata or DEFAULT_EPOCH_METADATA
    epoch_list: List[Dict[str, Any]] = (
        meta["epochs"] if isinstance(meta, dict) else list(meta)
    )

    def _norm(text: str) -> str:
        return str(text).lower().replace(" ", "").replace("_", "")

    manifest: Dict[str, Any] = {
        "selected": list(selected_days),
        "raw_dir": str(raw_dir),
        "epochs": [],
        "missing": [],
        "ready": False,
    }
    if not selected_days:
        logger.warning("combine_epochs: no epochs selected")
        return manifest

    wanted = {_norm(d) for d in selected_days}
    for epoch in epoch_list:
        keys = [str(epoch.get(k, "")) for k in ("id", "label")]
        if not any(_norm(k) in wanted or wanted & {_norm(k)} for k in keys):
            # lenient substring match both directions
            if not any(_norm(k) and (_norm(k) in wanted or any(w in _norm(k) for w in wanted))
                       for k in keys):
                continue

        entry = dict(epoch)
        files = _candidate_files(raw_dir, epoch)
        existing = next((f for f in files if f.is_file()), None)
        entry["path"] = str(existing) if existing else str(files[0]) if files else ""
        entry["exists"] = existing is not None
        if entry["exists"]:
            logger.info("combine_epochs: %s -> %s", epoch.get("id"), entry["path"])
        else:
            manifest["missing"].append(epoch.get("id"))
            logger.warning("combine_epochs: %s -> file not found (%s)",
                           epoch.get("id"), entry["path"])
        manifest["epochs"].append(entry)

    manifest["ready"] = bool(manifest["epochs"]) and not manifest["missing"]
    logger.info("combine_epochs: %d epoch(s) prepared, ready=%s",
                len(manifest["epochs"]), manifest["ready"])
    return manifest


def load_metadata(source: Union[str, Path, Dict[str, Any], None]) -> Dict[str, Any]:
    """Normalize a metadata source into the standard config dict.

    Accepts a JSON file path, an already-loaded dict, or ``None`` (uses the
    Sgr A* 2017 campaign default).  Missing keys are filled from defaults.
    """
    if source is None:
        return json.loads(json.dumps(DEFAULT_EPOCH_METADATA))  # deep copy
    if isinstance(source, dict):
        meta = source
    else:
        try:
            meta = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("cannot load epoch metadata from %s: %s", source, exc)
            return json.loads(json.dumps(DEFAULT_EPOCH_METADATA))

    base = json.loads(json.dumps(DEFAULT_EPOCH_METADATA))
    if isinstance(meta, list):
        base["epochs"] = meta
    elif isinstance(meta, dict):
        base.update(meta)
        base.setdefault("epochs", DEFAULT_EPOCH_METADATA["epochs"])
    else:
        logger.error("unsupported metadata type %s", type(meta))
        return base
    return base


def _normalize_epoch(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map alias keys from an external metadata dict to canonical keys."""
    out: Dict[str, Any] = {}
    for canonical, aliases in _EPOCH_KEY_ALIASES.items():
        for key in aliases:
            if key in raw and canonical not in out:
                out[canonical] = raw[key]
    return out


# ------------------------------------------------------------------- GUI -----

def _qt():
    """Import PyQt5 lazily so the module can be used headless."""
    try:
        from PyQt5 import QtCore, QtWidgets  # noqa: F401
        return QtCore, QtWidgets
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyQt5 is required for the epoch selection UI; "
            "combine_epochs() works without it."
        ) from exc


class EpochSelectionDialog:
    """Multi-epoch selection & batch combination modal (Y2K themed).

    Build it either directly around a ``QDialog``::

        dialog = EpochSelectionDialog()
        dialog.widget.exec_()

    The static convenience :meth:`select_epochs` runs the dialog modally and
    returns the combine manifest (or ``None`` if the user cancelled).
    """

    def __init__(
        self,
        parent: Any = None,
        metadata: Union[str, Path, Dict[str, Any], List[Dict[str, Any]], None] = None,
        raw_dir: Union[str, Path, None] = None,
        on_combine: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        QtCore, QtWidgets = _qt()
        self._qt = QtWidgets
        self._qcore = QtCore
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
        self._on_combine = on_combine

        meta = load_metadata(metadata)
        raw_epochs = meta.get("epochs", [])
        if raw_epochs and isinstance(raw_epochs[0], dict) and "id" not in raw_epochs[0]:
            raw_epochs = [_normalize_epoch(e) for e in raw_epochs]
        self.metadata = meta
        self.epochs = raw_epochs or DEFAULT_EPOCH_METADATA["epochs"]

        # --- dialog shell -------------------------------------------------
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("ASTRORECON // MULTI-EPOCH TOMOGRAPHY")
        dlg.setMinimumWidth(430)
        dlg.setObjectName("EpochDialogRoot")
        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        title = QtWidgets.QLabel("◆ MULTI-EPOCH SELECTION")
        title.setObjectName("EpochTitle")
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            f"{meta.get('target', 'target')} — {meta.get('campaign', 'campaign')}")
        subtitle.setObjectName("EpochSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        divider = QtWidgets.QFrame()
        divider.setObjectName("EpochDivider")
        divider.setFixedHeight(1)
        root.addWidget(divider)

        # --- scrollable checkbox list -------------------------------------
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(2)

        self._checkboxes: List["QCheckBox"] = []
        for epoch in self.epochs:
            cb = QtWidgets.QCheckBox(str(epoch.get("label", epoch.get("id", "epoch"))))
            cb.setToolTip(str(epoch.get("tooltip", epoch.get("state", ""))))
            cb.setChecked(bool(epoch.get("default_selected", False)))
            cb.setStyleSheet(QSS)
            state = epoch.get("state")
            if state:
                cb.setText(f"{cb.text()}   [{state}]")
            panel_layout.addWidget(cb)
            self._checkboxes.append(cb)
        panel_layout.addStretch(1)
        scroll.setWidget(panel)
        root.addWidget(scroll, stretch=1)

        self.status_label = QtWidgets.QLabel("// select epochs and initiate combination")
        self.status_label.setObjectName("EpochStatus")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # --- action row ----------------------------------------------------
        buttons = QtWidgets.QHBoxLayout()
        self.combine_btn = QtWidgets.QPushButton("⚡ Fetch & Combine Selected Epochs")
        self.combine_btn.setObjectName("EpochPrimary")
        self.combine_btn.clicked.connect(self._on_combine_clicked)
        cancel_btn = QtWidgets.QPushButton("ABORT")
        cancel_btn.setObjectName("EpochSecondary")
        cancel_btn.clicked.connect(dlg.reject)
        buttons.addWidget(self.combine_btn, stretch=1)
        buttons.addWidget(cancel_btn)
        root.addLayout(buttons)

        footer = QtWidgets.QLabel(
            "source: EHT 2017 campaign · dynamical tomography pipeline ready")
        footer.setObjectName("EpochFooter")
        root.addWidget(footer)

        self.widget = dlg
        self.result_manifest: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def selected_days(self) -> List[str]:
        """Return the ids of currently checked epochs (in list order)."""
        return [self.epochs[i].get("id", self.epochs[i].get("label", f"epoch{i}"))
                for i, cb in enumerate(self._checkboxes) if cb.isChecked()]

    def _on_combine_clicked(self) -> None:
        days = self.selected_days()
        if not days:
            self.status_label.setText("// ⚠ select at least one epoch")
            return
        manifest = combine_epochs(days, metadata=self.metadata, raw_dir=self.raw_dir)
        self.result_manifest = manifest
        if manifest.get("missing"):
            self.status_label.setText(
                "// ⚠ missing data for: " + ", ".join(manifest["missing"]))
            return
        self.status_label.setText(
            f"// ✓ combined {len(manifest['epochs'])} epoch(s) — dispatching...")
        if self._on_combine:
            try:
                self._on_combine(manifest)
            except Exception:
                logger.exception("on_combine callback failed")
        self.widget.accept()

    # ------------------------------------------------------------------
    @classmethod
    def select_epochs(
        cls,
        parent: Any = None,
        metadata: Union[str, Path, Dict[str, Any], List[Dict[str, Any]], None] = None,
        raw_dir: Union[str, Path, None] = None,
        on_combine: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the modal and return the combine manifest, or ``None``."""
        dialog = cls(parent=parent, metadata=metadata, raw_dir=raw_dir,
                     on_combine=on_combine)
        if dialog.widget.exec_() == dialog.widget.Accepted and dialog.result_manifest:
            return dialog.result_manifest
        return None


if __name__ == "__main__":  # pragma: no cover - smoke test
    # Headless backend check
    manifest = combine_epochs(["Day 095", "Day 097"])
    print(json.dumps(manifest, indent=2)[:1200])
    try:
        import sys
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        app.setStyleSheet(QSS)
        print("GUI OK:", EpochSelectionDialog.select_epochs())
    except ImportError:
        print("(PyQt5 unavailable - backend smoke test only)")
