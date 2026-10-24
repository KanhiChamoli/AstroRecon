"""Sources / credits modal — data provenance and background-music attribution."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea,
    QLayout,
)

from .audio import MUSIC_CREDIT

DATA_SOURCES = [
    ("Event Horizon Telescope Collaboration (2022)", "First Sgr A* results — UVFITS archive"),
    ("Astro Recon pipeline", "Ray-marcher, inverse bridge, interferometry bridge (in-repo)"),
    ("Sample datasets", "data/raw/sgr_a_real.fits · data/raw/M13_sample.fits · orbit_data.csv"),
]


class SourcesModal(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Astro Recon — Sources & Credits")
        self.setModal(True)
        self.setMinimumSize(520, 420)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(12)

        head = QHBoxLayout()
        t = QLabel("◈  SOURCES & CREDITS")
        t.setObjectName("Section")
        close = QPushButton("✕")
        close.setFixedSize(34, 34)
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.accept)
        head.addWidget(t); head.addStretch(1); head.addWidget(close)
        root.addLayout(head)

        card = QFrame(); card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(8)

        # Scroll container for the sources content: strictly vertical scrolling
        # only. Qt QSS has no `overflow-x/y` properties (it logs "Unknown
        # property"), so the policy is enforced here in Python instead.
        scroll = QScrollArea()
        scroll.setObjectName("SourcesScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizeAdjustPolicy(QScrollArea.AdjustIgnored)
        # The horizontal scrollbar must never appear, even if a child tries to
        # exceed the viewport: pin its handle to zero size.
        scroll.horizontalScrollBar().setFixedHeight(0)
        scroll.horizontalScrollBar().setRange(0, 0)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            "QScrollBar:horizontal { height: 0px; width: 0px; background: transparent; }"
            "QScrollBar::handle:horizontal { width: 0px; min-width: 0px; }"
        )
        scroll_content = QFrame()
        scroll_content.setObjectName("SourcesScrollContent")
        scroll_content.setMinimumWidth(0)
        scroll_content.setMaximumWidth(16777215)
        body = QVBoxLayout(scroll_content)
        body.setContentsMargins(0, 0, 8, 0)
        body.setSpacing(8)
        body.setAlignment(Qt.AlignHCenter)
        body.setSizeConstraint(QLayout.SetNoConstraint)

        d = QLabel("DATA SOURCES")
        d.setObjectName("Section")
        d.setAlignment(Qt.AlignHCenter)
        body.addWidget(d)

        # Fancy clickable Powered by boxes (WWT first, SkyView below it) —
        # both centered.
        wwt_box = QLabel(
            '<div align="center">'
            '<a href="https://worldwidetelescope.org/home/" style="text-decoration: none; color: #38bdf8;">'
            '<span style="font-family: \'Cinzel\', \'Cinzel Decorative\', \'Georgia\', serif; font-size: 13px; font-weight: bold; letter-spacing: 1.5px;">'
            '✧ POWERED BY WORLDWIDE TELESCOPE ✧</span></a><br>'
            '<span style="color: #94a3b8; font-size: 11px;">American Astronomical Society\'s WorldWide Telescope — interactive multi-wavelength sky engine (pywwt 0.26+).</span>'
            '</div>'
        )
        wwt_box.setOpenExternalLinks(True)
        wwt_box.setWordWrap(True)
        wwt_box.setAlignment(Qt.AlignHCenter)
        wwt_box.setStyleSheet(
            "QLabel {"
            "    background: rgba(14, 25, 45, 0.7);"
            "    border: 1px solid rgba(56, 189, 248, 0.4);"
            "    border-radius: 8px;"
            "    padding: 8px 10px;"
            "}"
        )
        body.addWidget(wwt_box)

        skyview_box = QLabel(
            '<div align="center">'
            '<a href="https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl" style="text-decoration: none; color: #38bdf8;">'
            '<span style="font-family: \'Cinzel\', \'Cinzel Decorative\', \'Georgia\', serif; font-size: 13px; font-weight: bold; letter-spacing: 1.5px;">'
            '✧ POWERED BY SKYVIEW ✧</span></a><br>'
            '<span style="color: #94a3b8; font-size: 11px;">NASA\'s SkyView Virtual Observatory (multi-wavelength astronomical survey data engine).</span>'
            '</div>'
        )
        skyview_box.setOpenExternalLinks(True)
        skyview_box.setWordWrap(True)
        skyview_box.setAlignment(Qt.AlignHCenter)
        skyview_box.setStyleSheet(wwt_box.styleSheet())
        body.addWidget(skyview_box)

        for title, desc in DATA_SOURCES:
            row = QLabel(
                f"<div align='center'><b>{title}</b><br>"
                f"<span style='color:gray'>{desc}</span></div>"
            )
            row.setWordWrap(True)
            row.setAlignment(Qt.AlignHCenter)
            body.addWidget(row)

        body.addSpacing(8)
        m = QLabel("BACKGROUND MUSIC")
        m.setObjectName("Section")
        m.setAlignment(Qt.AlignHCenter)
        body.addWidget(m)
        artists = MUSIC_CREDIT.get('contributing_artists', 'Koji Kondo, Toru Minegishi')
        album = MUSIC_CREDIT.get('album', "The Legend of Zelda: Majora's Mask Original Soundtrack")
        music = QLabel(
            f"<div align='center'><b>{MUSIC_CREDIT['track']}</b><br>"
            f"Contributing Artists: {artists}<br>"
            f"Album: {album}<br>"
            f"Game: {MUSIC_CREDIT['game']}<br>"
            f"<a href='{MUSIC_CREDIT['url']}'>{MUSIC_CREDIT['url']}</a><br>"
            f"<span style='color:gray'>All rights belong to Nintendo and the original creators.</span></div>"
        )
        music.setOpenExternalLinks(True)
        music.setWordWrap(True)
        music.setAlignment(Qt.AlignHCenter)
        body.addWidget(music)
        body.addStretch(1)
        scroll.setWidget(scroll_content)
        lay.addWidget(scroll, 1)

        root.addWidget(card, 1)

        foot = QLabel("Copyright Astro Recon — Made and owned by Aaradhya Prasad Chamoli under Aphellion Studios")
        foot.setObjectName("Footer")
        foot.setAlignment(Qt.AlignCenter)
        root.addWidget(foot)

    def apply_palette(self, p) -> None:
        self.setStyleSheet(f"QDialog {{ background: {p.bg_rise}; }}")
