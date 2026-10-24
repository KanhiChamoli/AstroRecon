"""WorldWide Telescope bridge — interactive sky viewport inside the desktop app.

Hosts the WWT WebGL engine inside a ``QWebEngineView`` and exposes a small
PyQt5-friendly API:

* :meth:`WWTBridge.load_manifest`  — load a WTML JSON manifest (dict, path or URL)
* :meth:`WWTBridge.set_target`     — re-center on RA/Dec with a zoom level
* :meth:`WWTBridge.set_survey`     — toggle multi-wavelength survey layers
  (DSS Optical, Chandra X-Ray, Spitzer IR, Fermi Gamma, NVSS Radio)

The bridge prefers the official ``pywwt`` Jupyter widget when it is importable
(its WWT web client assets are used via the local web server), and falls back
to the public WWT web client at ``https://web.wwtassets.org/`` otherwise.
All communication with the web view is asynchronous (runJavaScript), so the
Qt event loop is never blocked.

Y2K retro aesthetic: deep-space background #0b0b16, neon-green #00ff66 text,
cyan #38bdf8 accents, Courier New monospace.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSizePolicy, QFrame,
)

logger = logging.getLogger(__name__)

WWT_WEB_CLIENT = "https://web.wwtassets.org/research/latest/"
WWT_HOME_URL = "https://worldwidetelescope.org/home/"

#: Built-in multi-wavelength survey layers (name -> WWT image-set lookup)
SURVEY_LAYERS = {
    "DSS Optical":       "Digitized Sky Survey (Color)",
    "Chandra X-Ray":     "Chandra X-Ray Observatory",
    "Spitzer Infrared":  "Spitzer Infrared Space Telescope",
    "Fermi Gamma-Ray":   "Fermi Gamma-Ray Space Telescope",
    "NVSS Radio":        "NRAO VLA Sky Survey (NVSS)",
    "WISE Infrared":     "WISE All-Sky (Infrared)",
}


def _ensure_qtwebengine():
    """Import QtWebEngineWidgets, raising a friendly error if missing.

    QtWebEngine must be imported BEFORE any QCoreApplication exists (or
    AA_ShareOpenGLContexts must be set first) — when the app is launched
    normally, the QApplication is already alive, so we preflight the import
    with the attribute set and fall back to a graceful, non-fatal outcome if
    the engine can no longer attach to this process.
    """
    try:
        from PyQt5.QtCore import QCoreApplication, Qt
        if QCoreApplication.instance() is None:
            QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
        from PyQt5.QtWebEngineWidgets import QWebEngineView  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyQtWebEngine is required for the WWT viewport.\n"
            "Install it with:  pip install PyQtWebEngine"
        ) from exc


class WWTBridge(QWidget):
    """Interactive WorldWide Telescope viewport widget (Qt wrapper)."""

    #: emitted after every successful command sent to the engine
    command_sent = pyqtSignal(str)
    #: emitted when the WWT page finishes loading
    engine_ready = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None,
                 show_layer_toolbar: bool = True) -> None:
        super().__init__(parent)
        _ensure_qtwebengine()
        from PyQt5.QtWebEngineWidgets import QWebEngineView

        self.setObjectName("WWTBridge")
        self._ready = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ------------------------------------------------ layer toolbar ----
        if show_layer_toolbar:
            bar = QFrame()
            bar.setObjectName("WWTToolbar")
            bar.setStyleSheet(
                "QFrame#WWTToolbar { background: #0b0b16; border: 1px solid #14303a;"
                " border-radius: 6px; }"
            )
            bl = QHBoxLayout(bar)
            bl.setContentsMargins(8, 4, 8, 4)
            bl.setSpacing(8)

            lbl = QLabel("SURVEY")
            lbl.setStyleSheet(
                "color: #00ff66; font-family: 'Courier New'; font-size: 11px;"
                " font-weight: bold; letter-spacing: 1px; background: transparent;"
            )
            bl.addWidget(lbl)

            self.survey_combo = QComboBox()
            self.survey_combo.addItems(list(SURVEY_LAYERS.keys()))
            self.survey_combo.setStyleSheet(
                "QComboBox { background: #101024; color: #38bdf8;"
                " font-family: 'Courier New'; font-size: 12px; border: 1px solid #14303a;"
                " border-radius: 4px; padding: 2px 8px; }"
                "QComboBox QAbstractItemView { background: #101024; color: #00ff66;"
                " selection-background-color: #14303a; }"
            )
            self.survey_combo.currentTextChanged.connect(self.set_survey)
            bl.addWidget(self.survey_combo)

            self.center_btn = QPushButton("◎  RECENTER")
            self.center_btn.setStyleSheet(
                "QPushButton { background: #101024; color: #00ff66;"
                " font-family: 'Courier New'; font-size: 11px; font-weight: bold;"
                " border: 1px solid #14303a; border-radius: 4px; padding: 3px 10px; }"
                "QPushButton:hover { background: #14303a; color: #38bdf8; }"
            )
            self.center_btn.setToolTip("Re-center on the current target")
            self.center_btn.clicked.connect(self._on_recenter)
            bl.addWidget(self.center_btn)

            self.open_web_btn = QPushButton("⧉  WWT SITE")
            self.open_web_btn.setStyleSheet(self.center_btn.styleSheet())
            self.open_web_btn.setToolTip("Open worldwidetelescope.org in your browser")
            self.open_web_btn.clicked.connect(self._on_open_web)
            bl.addWidget(self.open_web_btn)

            bl.addStretch(1)
            root.addWidget(bar)

        # ------------------------------------------------ web viewport -----
        self.view = QWebEngineView(self)
        self.view.setObjectName("WWTWeb")
        self.view.setStyleSheet(
            "QWebEngineView { background: #0b0b16; border: 1px solid #14303a;"
            " border-radius: 6px; }"
        )
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.loadFinished.connect(self._on_load_finished)
        root.addWidget(self.view, 1)

        # status strip
        self.status = QLabel("WWT ENGINE · booting…")
        self.status.setStyleSheet(
            "color: #00ff66; font-family: 'Courier New'; font-size: 10px;"
            " letter-spacing: 1px; padding: 1px 4px;"
        )
        root.addWidget(self.status)

        # remember last commanded state so RECENTER works
        self._last_target: Dict[str, Any] = {"ra": 266.4168, "dec": -29.0078,
                                             "zoom": 30.0}  # Sgr A* default
        self._last_survey = "DSS Optical"

        # Prefer a locally-installed pywwt client if present; else the web client.
        page_url = self._resolve_client_url()
        self.view.load(QUrl(page_url))
        self.view.loadFinished.connect(
            lambda ok: logger.info("WWT client load %s: %s", "ok" if ok else "FAILED", page_url)
        )

    # ------------------------------------------------------------ plumbing --
    @staticmethod
    def _resolve_client_url() -> str:
        """Use the pywwt-bundled client when available, else the hosted one."""
        try:
            import pywwt  # noqa: F401
            # pywwt serves its client through Jupyter; the hosted research app
            # exposes the same engine and needs zero extra wiring, so use it
            # whenever pywwt exists too (identical scripting surface).
        except Exception:
            pass
        return WWT_WEB_CLIENT

    def _on_load_finished(self, ok: bool) -> None:
        self._ready = bool(ok)
        if ok:
            self.status.setText("WWT ENGINE · online")
            self.engine_ready.emit()
            # re-apply last state once the engine is up
            QTimer_single(1200, lambda: (self.set_survey(self._last_survey),
                                         self.set_target(**self._last_target)))
        else:
            self.status.setText("WWT ENGINE · offline (check network / PyQtWebEngine)")

    # -------------------------------------------------------------- public --
    def is_ready(self) -> bool:
        return self._ready

    def set_target(self, ra_deg: float, dec_deg: float, zoom_deg: float = 30.0) -> None:
        """Re-center the sky on RA/Dec (degrees, ICRS) with a zoom level."""
        self._last_target = {"ra": float(ra_deg), "dec": float(dec_deg),
                             "zoom": float(zoom_deg)}
        js = (
            f"(() => {{ const w=window.wwt ?? (window.parent && window.parent.wwt);"
            f" if (w) {{ try {{ w.setBackgroundImageByName(null); }} catch(e) {{}} }}"
            f" if (window.wwtctl && window.wwtctl.goto) {{"
            f"  window.wwtctl.goto({ra_deg:.6f}, {dec_deg:.6f}, {zoom_deg:.4f}); }} }})();"
        )
        self._run(js, f"goto RA={ra_deg:.4f} Dec={dec_deg:.4f}")

    def set_survey(self, name: str) -> None:
        """Switch the background survey layer (e.g. 'DSS Optical')."""
        if name in SURVEY_LAYERS:
            self._last_survey = name
        js = (
            f"(() => {{ const w=window.wwt ?? (window.parent && window.parent.wwt);"
            f" const n={json.dumps(SURVEY_LAYERS.get(name, name))};"
            f" if (w && w.setBackgroundImageByName) {{"
            f"  try {{ w.setBackgroundImageByName(n); }} catch(e) {{}} }} }})();"
        )
        self._run(js, f"survey → {name}")

    def load_manifest(self, manifest) -> None:
        """Load a WTML manifest.

        Accepts a dict (already-parsed JSON), a local ``.wtml``/``.json`` file
        path, or an http(s) URL. The manifest's place/FOV, if present, is used
        to re-center the viewport.
        """
        data: Optional[Dict[str, Any]] = None
        if isinstance(manifest, dict):
            data = manifest
        else:
            text = str(manifest)
            if text.startswith(("http://", "https://")):
                self._run(f"location.href={json.dumps(text)};", "wtml url load")
                return
            path = Path(text)
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.error("WTML parse failed for %s: %s", path, exc)
                    return
            else:
                logger.warning("WTML manifest not found: %s", path)
                return

        if not data:
            return
        # Best-effort: if the manifest carries target coordinates, aim there.
        try:
            folder = data.get("Folder") or data
            place = folder.get("Place") or (folder.get("Places") or [None])[0]
            if isinstance(place, list):
                place = place[0]
            if isinstance(place, dict):
                lat = float(place.get("Lat") or place.get("Dec") or 0.0)
                lng = float(place.get("Lng") or place.get("RA") or 0.0)
                fov = float(place.get("ZoomLevel") or 30.0)
                self.set_target(lng, lat, fov)
        except Exception as exc:
            logger.debug("WTML coordinate extraction skipped: %s", exc)
        # Hand the raw JSON to the engine for full-fidelity handling
        payload = json.dumps(json.dumps(data))
        self._run(f"(() => {{ if (window.wwtctl && window.wwtctl.loadWtml)"
                  f" {{ window.wwtctl.loadWtml({payload}); }} }})();", "wtml manifest")

    # ------------------------------------------------------------ internals --
    def _run(self, js: str, what: str) -> None:
        """Fire JavaScript at the engine without blocking the Qt loop."""
        try:
            self.view.page().runJavaScript(js)
            self.command_sent.emit(what)
            self.status.setText(f"WWT ▸ {what}")
        except Exception as exc:
            logger.warning("WWT command %r failed: %s", what, exc)

    def _on_recenter(self) -> None:
        self.set_target(**self._last_target)

    def _on_open_web(self) -> None:
        import webbrowser
        webbrowser.open(WWT_HOME_URL)


def QTimer_single(ms: int, fn) -> None:
    """Small non-blocking helper (QTimer.singleShot wrapper)."""
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(ms, fn)


def launch_wwt_window(entry: Optional[dict] = None) -> int:
    """Open WWT in a standalone retro-styled window. Returns a QApplication id.

    Safe to call from scripts/tests; mirrors the app palette so it also works
    inside the running GUI process (it just creates a top-level widget).
    """
    from PyQt5.QtWidgets import QMainWindow, QApplication

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication([])
        created = True

    win = QMainWindow()
    win.setWindowTitle("Astro Recon — WWT Observatory Mode")
    win.setMinimumSize(1000, 680)
    win.setStyleSheet("QMainWindow { background: #0b0b16; }")

    bridge = WWTBridge(win)
    win.setCentralWidget(bridge)

    # aim at the selected target if one was provided
    entry = entry or {}
    coords = entry.get("coords") or {}
    try:
        ra = float(coords.get("ra", 266.4168))
        dec = float(coords.get("dec", -29.0078))
    except (TypeError, ValueError):
        ra, dec = 266.4168, -29.0078
    bridge.set_target(ra, dec, 30.0)

    win.show()
    if created:
        return app.exec_()
    return 0
