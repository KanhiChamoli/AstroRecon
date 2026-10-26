"""Astro Recon — startup / main menu page.

Nebula aesthetic rendered procedurally on a QWidget (no image assets needed):
layered radial-gradient glows + parallax starfield that drifts slowly.
"""
from __future__ import annotations

import math
import random
import os
import webbrowser

from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QBrush, QPainterPath, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QFrame,
)

from .theme import Palette

GITHUB_URL = "https://github.com/"  # TODO: point at the real Aphellion Studios repo
SKYVIEW_URL = "https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl"
SKYVIEW_LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "images", "skyview_logo_bright.png"
)
WWT_URL = "https://worldwidetelescope.org/home/"
WWT_LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "images", "wwt_logo.png"
)


class NebulaCanvas(QWidget):
    """Paints a slow-drifting starfield over layered nebula glows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RootStack")
        self._t = 0.0
        rng = random.Random(2026)
        self._stars = [
            {
                "x": rng.random(), "y": rng.random(),
                "r": rng.uniform(0.4, 1.8), "tw": rng.uniform(0.5, 2.5),
                "ph": rng.uniform(0, math.tau),
            }
            for _ in range(220)
        ]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 fps

    def _tick(self) -> None:
        self._t = (self._t + 0.016) % 1000
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # deep space base
        base = QBrush(QColor("#050814"))
        painter.fillRect(self.rect(), base)

        # layered nebula glows (positions breathe slowly with time)
        def glow(cx, cy, radius, color, alpha):
            g = QRadialGradient(QPointF(cx * w, cy * h), radius * min(w, h))
            c = QColor(color)
            c.setAlpha(int(alpha))
            g.setColorAt(0.0, c)
            c2 = QColor(color)
            c2.setAlpha(0)
            g.setColorAt(1.0, c2)
            painter.setBrush(QBrush(g))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx * w, cy * h), radius * min(w, h), radius * min(w, h))

        breath = 0.06 * math.sin(self._t * 0.7)
        glow(0.22 + 0.02 * math.sin(self._t * 0.11), 0.30, 0.55 + breath, "#7c3aed", 66)
        glow(0.80 + 0.02 * math.cos(self._t * 0.09), 0.68, 0.50 + breath, "#0891b2", 55)
        glow(0.55, 0.12, 0.38, "#c026d3", 34)
        glow(0.10, 0.85, 0.30, "#2563eb", 28)

        # parallax starfield
        painter.setPen(Qt.NoPen)
        for s in self._stars:
            tw = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(self._t * s["tw"] + s["ph"]))
            x = (s["x"] + self._t * 0.004) % 1.0
            y = s["y"]
            c = QColor(230, 238, 255)
            c.setAlpha(int(200 * tw))
            painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(x * w, y * h), s["r"], s["r"])

        painter.end()


class MainMenuPage(NebulaCanvas):
    """Startup page: title, primary button stack, copyright footer."""

    def __init__(self, theme_manager, on_run, on_sources, on_github, on_settings) -> None:
        super().__init__()
        self._tm = theme_manager
        self._palette = self._tm.palette

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # top bar with settings gear + technical stack
        top = QHBoxLayout()
        top.setContentsMargins(20, 16, 20, 0)
        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tech_stack_btn = QPushButton("🧬  Technical Stack")
        self.tech_stack_btn.setObjectName("MenuButton")
        self.tech_stack_btn.setFixedWidth(180)
        self.tech_stack_btn.setCursor(Qt.PointingHandCursor)
        self.tech_stack_btn.setToolTip("Backend dependencies & data APIs")
        self.tech_stack_btn.clicked.connect(self._show_tech_stack)
        self.settings_btn = QPushButton("⚙  Settings")
        self.settings_btn.setObjectName("MenuButton")
        self.settings_btn.setFixedWidth(150)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.clicked.connect(on_settings)
        top.addWidget(spacer)
        top.addWidget(self.tech_stack_btn)
        top.addWidget(self.settings_btn)
        root.addLayout(top)

        # centered hero block
        root.addStretch(2)
        hero = QVBoxLayout()
        hero.setSpacing(6)
        hero.setAlignment(Qt.AlignHCenter)

        self.title = QLabel("ASTRO RECON")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignCenter)
        self.subtitle = QLabel("COMPUTATIONAL ASTROPHYSICS  ·  SPATIAL OBJECTS  ·  3D NEBULAE")
        self.subtitle.setObjectName("Subtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)

        hero.addWidget(self.title)
        hero.addWidget(self.subtitle)
        root.addLayout(hero)

        root.addStretch(1)

        # vertical stack of primary buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(14)
        btn_col.setAlignment(Qt.AlignHCenter)

        specs = [
            ("▶   RUN SIMULATION", "Launch the simulation pipeline on a spatial object", on_run),
            ("◈   SOURCES", "Credits, data provenance & background-music attribution", on_sources),
            ("⌥   GITHUB REPO & ADDITIONAL SOURCES", "Open the project repository and docs", on_github),
        ]
        for label, tip, cb in specs:
            b = QPushButton(label)
            b.setObjectName("MenuButton")
            b.setFixedSize(500, 60)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.clicked.connect(cb)
            btn_col.addWidget(b)
        root.addLayout(btn_col)

        root.addStretch(2)

        # "Powered by" banners: WWT on top, SkyView below it, both centered.
        # (Previously side by side — user wants a vertically stacked,
        # horizontally centered column above the footer.)
        banner_col = QVBoxLayout()
        banner_col.setSpacing(4)
        banner_col.setAlignment(Qt.AlignHCenter)

        self.wwt_banner = QLabel(
            f'<a href="{WWT_URL}" style="text-decoration: none; color: #38bdf8;">'
            f'<span style="font-family: \'Cinzel\', \'Cinzel Decorative\', \'Georgia\', serif; font-size: 14px; font-weight: bold; letter-spacing: 2px;">'
            f'✧ POWERED BY WORLDWIDE TELESCOPE ✧</span></a>'
        )
        self.wwt_banner.setAlignment(Qt.AlignCenter)
        self.wwt_banner.setOpenExternalLinks(True)
        self.wwt_banner.setCursor(Qt.PointingHandCursor)
        self.wwt_banner.setToolTip("Click to visit the WorldWide Telescope (AAS)")
        banner_col.addWidget(self.wwt_banner, 0, Qt.AlignHCenter)

        self.skyview_banner = QLabel(
            f'<a href="{SKYVIEW_URL}" style="text-decoration: none; color: #38bdf8;">'
            f'<span style="font-family: \'Cinzel\', \'Cinzel Decorative\', \'Georgia\', serif; font-size: 14px; font-weight: bold; letter-spacing: 2px;">'
            f'✧ POWERED BY SKYVIEW ✧</span></a>'
        )
        self.skyview_banner.setAlignment(Qt.AlignCenter)
        self.skyview_banner.setOpenExternalLinks(True)
        self.skyview_banner.setCursor(Qt.PointingHandCursor)
        self.skyview_banner.setToolTip("Click to visit NASA's SkyView Observatory")
        banner_col.addWidget(self.skyview_banner, 0, Qt.AlignHCenter)

        root.addLayout(banner_col)
        root.addSpacing(6)

        # Bottom footer row: logos left/right balanced so the copyright
        # text sits exactly centered in the window (not offset by the
        # left-side logo weights).
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(20, 0, 20, 16)

        # Clickable SkyView logo on bottom-left
        self.logo_btn = QPushButton()
        self.logo_btn.setFlat(True)
        self.logo_btn.setCursor(Qt.PointingHandCursor)
        self.logo_btn.setToolTip("Visit NASA SkyView: https://skyview.gsfc.nasa.gov")
        self.logo_btn.clicked.connect(lambda: webbrowser.open(SKYVIEW_URL))
        self.logo_btn.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; }")

        if os.path.isfile(SKYVIEW_LOGO_PATH):
            pix = QPixmap(SKYVIEW_LOGO_PATH)
            # scale cleanly to ~46px height for crisp high-contrast visibility against the dark space background
            scaled = pix.scaledToHeight(46, Qt.SmoothTransformation)
            self.logo_btn.setIcon(QIcon(scaled))
            self.logo_btn.setIconSize(scaled.size())
            self.logo_btn.setFixedSize(scaled.width() + 12, 50)
        else:
            self.logo_btn.setText("🔭 SkyView")
            self.logo_btn.setStyleSheet("color: #38bdf8; font-weight: bold;")

        footer_row.addWidget(self.logo_btn, 0, Qt.AlignLeft | Qt.AlignVCenter)

        # Clickable Worldwide Telescope logo next to the SkyView one
        self.wwt_logo_btn = QPushButton()
        self.wwt_logo_btn.setFlat(True)
        self.wwt_logo_btn.setCursor(Qt.PointingHandCursor)
        self.wwt_logo_btn.setToolTip("Visit the WorldWide Telescope: https://worldwidetelescope.org")
        self.wwt_logo_btn.clicked.connect(lambda: webbrowser.open(WWT_URL))
        self.wwt_logo_btn.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; }")

        if os.path.isfile(WWT_LOGO_PATH):
            pix = QPixmap(WWT_LOGO_PATH)
            scaled = pix.scaledToHeight(46, Qt.SmoothTransformation)
            self.wwt_logo_btn.setIcon(QIcon(scaled))
            self.wwt_logo_btn.setIconSize(scaled.size())
            self.wwt_logo_btn.setFixedSize(scaled.width() + 12, 50)
        else:
            self.wwt_logo_btn.setText("🌐 Worldwide Telescope")
            self.wwt_logo_btn.setStyleSheet("color: #38bdf8; font-weight: bold;")
        footer_row.addWidget(self.wwt_logo_btn, 0, Qt.AlignLeft | Qt.AlignVCenter)

        # Copyright truly centered: balance left logos with a stretch on
        # both sides rather than a fixed-width spacer guess.
        footer_row.addStretch(1)

        # footer copyright text
        self.footer = QLabel(
            "Copyright Astro Recon — Made and owned by Aaradhya Prasad Chamoli under Aphellion Studios"
        )
        self.footer.setObjectName("Footer")
        self.footer.setAlignment(Qt.AlignCenter)
        footer_row.addWidget(self.footer, 0, Qt.AlignCenter)

        footer_row.addStretch(1)

        root.addLayout(footer_row)

        self._tm.add_listener(self.apply_palette)

    # ------------------------------------------------- technical stack -----
    def _show_tech_stack(self) -> None:
        """Retro modal listing backend dependencies and data APIs.

        Content lives in a QTextBrowser with strictly vertical scrolling
        (overflow-y: auto; overflow-x: hidden) so the full dependency
        specification is readable without horizontal drift.
        """
        from PyQt5.QtWidgets import QDialog, QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle("Astro Recon — Technical Stack")
        dlg.setModal(True)
        dlg.setFixedSize(640, 620)
        dlg.setWindowFlags(dlg.windowFlags() | Qt.FramelessWindowHint)
        dlg.setStyleSheet(f"QDialog {{ background: {self._palette.bg_rise}; border-radius: 10px; }}")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)

        title = QLabel("◈  TECHNICAL STACK")
        title.setStyleSheet("font-size: 18px; font-weight: bold; letter-spacing: 2px; color: #38bdf8;")
        lay.addWidget(title)

        sub = QLabel("Computational backbone powering Astro Recon")
        sub.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lay.addWidget(sub)

        html = """
        <div style="font-family:'Courier New', monospace;">
        <p style="color:#7ee787; font-size:12px; font-weight:bold; letter-spacing:1px; margin:6px 0 2px;">▸ CORE & SCIENTIFIC COMPUTING</p>
        <p style="color:#c9d1d9; font-size:11px; margin:0 0 4px; line-height:1.5;">
        • Python 3.7 or later<br>
        • NumPy 1.9 or later<br>
        • Matplotlib 1.5 or later<br>
        • SciPy — sparse solvers, filtering & optimisation kernels<br>
        • PyTorch (CUDA) — differentiable inverse-rendering optimiser
        </p>
        <p style="color:#7ee787; font-size:12px; font-weight:bold; letter-spacing:1px; margin:6px 0 2px;">▸ ASTRONOMY &amp; DATA INGESTION</p>
        <p style="color:#c9d1d9; font-size:11px; margin:0 0 4px; line-height:1.5;">
        • Astropy 1.0 or later — FITS I/O, coordinates, units &amp; constants<br>
        • Astroquery — MAST, SkyView, VizieR, SIMBAD, HEASARC archive clients<br>
        • Requests, Beautiful Soup 4, Dateutil, lxml — HTTP &amp; scraping stack<br>
        • reproject, pytz — WCS reprojection &amp; timezone handling<br>
        • ehtim / MPoL — radio interferometry (VLBI) uv-plane bridge<br>
        • WorldWide Telescope (pywwt 0.26+) — ambient multi-wavelength sky data
        </p>
        <p style="color:#7ee787; font-size:12px; font-weight:bold; letter-spacing:1px; margin:6px 0 2px;">▸ UI &amp; DESKTOP BINDINGS</p>
        <p style="color:#c9d1d9; font-size:11px; margin:0 0 4px; line-height:1.5;">
        • PySide / PyQt (PyQt4 &amp; PyQt5 supported) — this Y2K desktop UI<br>
        • PyQtWebEngine — renders the live WWT web viewport in-app<br>
        • QtPy 1.2 or later — binding-agnostic shims<br>
        • Tornado — local web server bridging pywwt into the desktop shell
        </p>
        <p style="color:#7ee787; font-size:12px; font-weight:bold; letter-spacing:1px; margin:6px 0 2px;">▸ JUPYTER &amp; INTERACTIVE WIDGETS</p>
        <p style="color:#c9d1d9; font-size:11px; margin:0 0 4px; line-height:1.5;">
        • Jupyter 1.0 or later, Notebook 5.0 or later<br>
        • ipywidgets 7.0+ / 8.0+, ipyevents, traitlets (pywwt dependencies)<br>
        </p>
        <p style="color:#7ee787; font-size:12px; font-weight:bold; letter-spacing:1px; margin:6px 0 2px;">▸ OBSERVATIONAL ARCHIVES</p>
        <p style="color:#c9d1d9; font-size:11px; margin:0 0 4px; line-height:1.5;">
        • NASA SkyView — <a href="https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl" style="color:#38bdf8; text-decoration:none;">skyview.gsfc.nasa.gov</a><br>
        • WorldWide Telescope (AAS) — <a href="https://worldwidetelescope.org/home/" style="color:#38bdf8; text-decoration:none;">worldwidetelescope.org</a><br>
        • NASA MAST (Hubble / JWST), VizieR, SIMBAD, HEASARC / Chandra<br>
        • Event Horizon Telescope (EHT) Collaboration — Sgr A* 2022 release
        </p>
        </div>
        """

        browser = QTextBrowser()
        browser.setObjectName("TechStackBrowser")
        browser.setHtml(html)
        browser.setOpenExternalLinks(True)
        # strictly vertical scrolling: overflow-y auto, overflow-x hidden
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        browser.horizontalScrollBar().setFixedHeight(0)
        browser.horizontalScrollBar().setRange(0, 0)
        browser.setStyleSheet(
            "QTextBrowser { background: rgba(14, 25, 45, 0.55); border: 1px solid #24304d;"
            " border-radius: 10px; padding: 10px; }"
            "QScrollBar:vertical { background: #0b1120; width: 8px; border-radius: 4px; }"
            "QScrollBar::handle:vertical { background: #38bdf8; border-radius: 4px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )
        lay.addWidget(browser, 1)

        foot = QLabel("Copyright Astro Recon — Made and owned by Aaradhya Prasad Chamoli under Aphellion Studios")
        foot.setStyleSheet("color: #8b949e; font-size: 11px;")
        foot.setAlignment(Qt.AlignCenter)
        lay.addWidget(foot)

        close = QPushButton("✕  Close")
        close.setObjectName("MenuButton")
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)

        dlg.exec_()

    # ------------------------------------------------------ theming -----
    def apply_palette(self, p: Palette) -> None:
        self._palette = p
        # repaint nebula with themed hues
        self._stars_hue = None
        self.update()
