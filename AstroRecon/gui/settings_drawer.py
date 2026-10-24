"""Gamer-style collapsible settings side-drawer.

Sections:
  THEME     — quick chips: Light / Dark / Dark Purple
  ACCENTS   — live RGB sliders for Primary / Secondary / Interactive colors
  AUDIO     — background-music play/stop, volume slider, mute toggle
  HARDWARE  — Compute Device & Hardware: live device dropdown + VRAM inventory
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QScrollArea, QFrame, QSizePolicy, QApplication, QTextEdit, QComboBox,
)

from .theme import Palette, THEMES
from .audio import MUSIC_CREDIT
from src.hardware import detect_hardware, get_manager


class HardwarePanel(QFrame):
    """Compute Device & Hardware card inside the settings drawer."""

    device_selected = pyqtSignal(str)  # identifier, e.g. "cuda:0" or "cpu"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.manager = get_manager()

        col = QVBoxLayout(self)
        col.setContentsMargins(12, 10, 12, 12)
        col.setSpacing(8)

        title = QLabel("COMPUTE DEVICE")
        title.setObjectName("Section")
        col.addWidget(title)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.combo, 1)
        refresh = QPushButton("⟳")
        refresh.setObjectName("IconBtn")
        refresh.setFixedSize(34, 34)
        refresh.setToolTip("Re-scan GPUs and VRAM usage")
        refresh.clicked.connect(self.refresh_devices)
        row.addWidget(refresh)
        col.addLayout(row)

        self.status = QLabel()
        self.status.setObjectName("Dim")
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        self.refresh_devices()
        self.combo.currentIndexChanged.connect(self._on_select)

        # live VRAM readout while training runs
        self._tick = QTimer(self)
        self._tick.setInterval(2000)
        self._tick.timeout.connect(self._update_status)
        self._tick.start()

    # ------------------------------------------------------------- logic --
    def refresh_devices(self) -> None:
        idx = self.combo.currentIndex()
        self.combo.blockSignals(True)
        self.combo.clear()
        for spec in detect_hardware():
            self.combo.addItem(spec.label, spec.identifier)
        at = self.combo.findData(self.manager.selected)
        self.combo.setCurrentIndex(at if at >= 0 else 0)
        self.combo.blockSignals(False)
        self._update_status()

    def _on_select(self, index: int) -> None:
        identifier = self.combo.itemData(index)
        if not identifier:
            return
        try:
            self.manager.set_selected(identifier)
            self.device_selected.emit(identifier)
        except Exception as exc:
            self.status.setText(f"⚠ Device switch failed: {exc}")

    def _update_status(self) -> None:
        identifier = self.manager.selected
        spec = self.manager.specs_for(identifier)
        if spec is not None and spec.identifier.startswith("cuda"):
            self.status.setText(
                f"Active: {spec.identifier} — {spec.used_vram_gb:.1f} / "
                f"{spec.total_vram_gb:.1f} GB VRAM · GPU lock engaged (no silent CPU fallback)."
            )
        elif spec is not None:
            self.status.setText(
                "Active: CPU — explicit fallback selected; heavy ray marching "
                "runs without GPU acceleration."
            )
        else:
            self.status.setText("Active: unknown device")

ACCENT_KEYS = [
    ("primary", "PRIMARY"),
    ("secondary", "SECONDARY"),
    ("interactive", "INTERACTIVE"),
]


class AccentSliders(QFrame):
    """Three rows of R/G/B sliders bound to one accent color."""

    rgb_changed = pyqtSignal(str, tuple)  # key, (r, g, b)

    def __init__(self, key: str, label: str, palette: Palette) -> None:
        super().__init__()
        self.key = key
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(6)

        head = QHBoxLayout()
        self._label = QLabel(label)
        self._label.setObjectName("Section")
        self._hex = QLabel("#------")
        self._hex.setObjectName("Dim")
        reset = QPushButton("⟳")
        reset.setObjectName("IconBtn")
        reset.setFixedSize(26, 26)
        reset.setToolTip("Reset to theme default")
        reset.clicked.connect(lambda: self.rgb_changed.emit(self.key, None))
        head.addWidget(self._label)
        head.addStretch(1)
        head.addWidget(self._hex)
        head.addWidget(reset)
        lay.addLayout(head)

        self._sliders = []
        self._suppress = False
        start = Palette.rgb(getattr(palette, key))
        for ch, name in enumerate(("R", "G", "B")):
            row = QHBoxLayout()
            tag = QLabel(name)
            tag.setFixedWidth(16)
            tag.setObjectName("Dim")
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 255)
            sl.setValue(start[ch])
            sl.valueChanged.connect(self._on_change)
            self._sliders.append(sl)
            row.addWidget(tag)
            row.addWidget(sl, 1)
            lay.addLayout(row)

    def _on_change(self, _v: int) -> None:
        if self._suppress:
            return
        rgb = tuple(s.value() for s in self._sliders)
        self._hex.setText("#{:02x}{:02x}{:02x}".format(*rgb))
        self.rgb_changed.emit(self.key, rgb)

    def sync_from_palette(self, p: Palette) -> None:
        """Update slider positions when theme changes (without re-emitting)."""
        self._suppress = True
        rgb = Palette.rgb(getattr(p, self.key))
        for s, v in zip(self._sliders, rgb):
            s.setValue(v)
        self._hex.setText("#{:02x}{:02x}{:02x}".format(*rgb))
        self._suppress = False


class SettingsWindow(QWidget):
    """Standalone pop-out settings window (drag by its header)."""

    def __init__(self, theme_manager, audio, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Astro Recon — Settings")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedWidth(480)
        self.setMinimumHeight(760)
        self._tm = theme_manager
        self._audio = audio
        self._drag_pos = None
        self._palette = theme_manager.palette

        root = QVBoxLayout(self)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)

        # ------------------------------------------------------ shell card
        self.shell = QFrame()
        self.shell.setObjectName("Drawer")
        shell_lay = QVBoxLayout(self.shell)
        shell_lay.setContentsMargins(16, 12, 16, 16)
        shell_lay.setSpacing(12)
        root.addWidget(self.shell)

        # --- header (drag handle + close) ---
        head = QHBoxLayout()
        title = QLabel("⚙  SETTINGS")
        title.setObjectName("Section")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        close = QPushButton("✕")
        close.setObjectName("IconBtn")
        close.setFixedSize(34, 34)
        close.setCursor(Qt.PointingHandCursor)
        close.setToolTip("Close settings")
        close.clicked.connect(self.close)
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(close)
        shell_lay.addLayout(head)

        # scrollable body — vertical only, no horizontal scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body = QWidget()
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        scroll.setWidget(body)
        shell_lay.addWidget(scroll)

        # ---------------------------------------------------------- THEME
        t = QLabel("THEME")
        t.setObjectName("Section")
        lay.addWidget(t)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self._chips: dict[str, QPushButton] = {}
        for name in ("Light", "Dark", "Dark Purple"):
            b = QPushButton(name.upper())
            b.setObjectName("ThemeChip")
            b.setCursor(Qt.PointingHandCursor)
            b.setCheckable(True)
            b.clicked.connect(lambda _c, n=name: self._tm.set_theme(n))
            chips.addWidget(b, 1)
            self._chips[name] = b
        lay.addLayout(chips)

        # -------------------------------------------------------- ACCENTS
        a = QLabel("COLOR CUSTOMIZER  ·  RGB")
        a.setObjectName("Section")
        lay.addSpacing(6)
        lay.addWidget(a)
        self._accent_rows = []
        for key, label in ACCENT_KEYS:
            row = AccentSliders(key, label, self._palette)
            row.rgb_changed.connect(self._on_rgb)
            lay.addWidget(row)
            self._accent_rows.append(row)

        # ---------------------------------------------------------- AUDIO
        au = QLabel("AUDIO & BACKGROUND MUSIC")
        au.setObjectName("Section")
        lay.addSpacing(6)
        lay.addWidget(au)

        audio_card = QFrame()
        audio_card.setObjectName("Card")
        al = QVBoxLayout(audio_card)
        al.setContentsMargins(12, 10, 12, 12)
        al.setSpacing(8)

        track = QLabel(f"♪  {MUSIC_CREDIT['track']}")
        track.setObjectName("Section")
        credit = QLabel(
            f"Game: {MUSIC_CREDIT['game']}  ·  "
            f"<a href='{MUSIC_CREDIT['url']}'>Listen on YouTube</a>"
        )
        credit.setObjectName("Dim")
        credit.setOpenExternalLinks(True)
        credit.setWordWrap(True)
        al.addWidget(track)
        al.addWidget(credit)

        if audio is None or not audio.available:
            warn = QLabel(
                "⚠  No local audio file found. Place the track at assets/audio/background_music.mp3 "
                "to enable playback."
            )
            warn.setObjectName("Dim")
            warn.setWordWrap(True)
            al.addWidget(warn)

        transport = QHBoxLayout()
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.clicked.connect(self._toggle_play)
        self.mute_btn = QPushButton("🔇  Mute")
        self.mute_btn.setCursor(Qt.PointingHandCursor)
        self.mute_btn.setCheckable(True)
        self.mute_btn.toggled.connect(self._on_mute)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.mute_btn)
        al.addLayout(transport)

        # Detailed track metadata box (requested right below the mute button)
        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setFixedHeight(110)
        self.details_box.setStyleSheet(
            "QTextEdit {"
            "    background: rgba(8, 12, 24, 0.7);"
            "    border: 1px solid rgba(80, 100, 140, 0.4);"
            "    border-radius: 8px;"
            "    font-family: 'Segoe UI', sans-serif;"
            "    font-size: 11px;"
            "    padding: 6px;"
            "}"
        )
        metadata_html = f"""
        <div style='line-height: 140%; color: #d0d8ea;'>
            <b style='color: #38bdf8;'>Title:</b> {MUSIC_CREDIT.get('track', 'Astral Observatory')}<br>
            <b style='color: #a78bfa;'>Contributing Artists:</b> {MUSIC_CREDIT.get('contributing_artists', 'Koji Kondo, Toru Minegishi')}<br>
            <b style='color: #34d399;'>Album:</b> {MUSIC_CREDIT.get('album', "The Legend of Zelda: Majora's Mask Original Soundtrack")}<br>
            <b style='color: #f472b6;'>Game:</b> {MUSIC_CREDIT.get('game', "The Legend of Zelda: Majora's Mask")}<br>
            <b style='color: #94a3b8;'>Source / Notes:</b> {MUSIC_CREDIT.get('source_notes', 'Extended Brawl BRSTM Reupload')}
        </div>
        """
        self.details_box.setHtml(metadata_html)
        al.addWidget(self.details_box)

        vol_row = QHBoxLayout()
        dec = QPushButton("−")
        dec.setObjectName("IconBtn")
        dec.setFixedSize(34, 34)
        dec.setToolTip("Volume down")
        dec.clicked.connect(lambda: self._nudge_volume(-5))
        inc = QPushButton("+")
        inc.setObjectName("IconBtn")
        inc.setFixedSize(34, 34)
        inc.setToolTip("Volume up")
        inc.clicked.connect(lambda: self._nudge_volume(+5))
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        base_vol = audio.base_volume if audio is not None else 40
        self.volume.setValue(base_vol)
        if audio is not None:
            self.volume.valueChanged.connect(self._audio.set_volume)
        self.vol_label = QLabel(f"{base_vol}%")
        self.vol_label.setFixedWidth(42)
        self.vol_label.setObjectName("Dim")
        vol_row.addWidget(dec)
        vol_row.addWidget(self.volume, 1)
        vol_row.addWidget(inc)
        vol_row.addWidget(self.vol_label)
        al.addLayout(vol_row)

        lay.addWidget(audio_card)

        # -------------------------------------------------- INTERFACE SOUNDS
        sounds_title = QLabel("INTERFACE SOUNDS")
        sounds_title.setObjectName("Section")
        lay.addSpacing(6)
        lay.addWidget(sounds_title)
        sounds_card = QFrame()
        sounds_card.setObjectName("Card")
        sounds_lay = QVBoxLayout(sounds_card)
        sounds_lay.setContentsMargins(12, 10, 12, 12)
        sounds_lay.setSpacing(8)
        sounds_row = QHBoxLayout()
        sounds_row.addWidget(QLabel("Retro button bleeps"), 1)
        self.interface_sounds_toggle = QPushButton(
            "ON" if self._audio and self._audio.interface_sounds_enabled else "OFF"
        )
        self.interface_sounds_toggle.setCheckable(True)
        self.interface_sounds_toggle.setChecked(bool(self._audio and self._audio.interface_sounds_enabled))
        self.interface_sounds_toggle.toggled.connect(self._set_interface_sounds)
        sounds_row.addWidget(self.interface_sounds_toggle)
        sounds_lay.addLayout(sounds_row)
        volume_row = QHBoxLayout()
        volume_row.addWidget(QLabel("Click volume"))
        self.interface_sounds_volume = QSlider(Qt.Horizontal)
        self.interface_sounds_volume.setRange(0, 100)
        self.interface_sounds_volume.setValue(self._audio.interface_sounds_volume if self._audio else 35)
        self.interface_sounds_volume.valueChanged.connect(
            self._audio.set_interface_sounds_volume if self._audio else lambda _v: None
        )
        volume_row.addWidget(self.interface_sounds_volume, 1)
        sounds_lay.addLayout(volume_row)
        lay.addWidget(sounds_card)

        # ------------------------------------------------------ HARDWARE
        hw = QLabel("🖥️  COMPUTE DEVICE & HARDWARE")
        hw.setObjectName("Section")
        lay.addSpacing(6)
        lay.addWidget(hw)
        self.hardware = HardwarePanel()
        self.hardware.device_selected.connect(self._on_device_selected)
        lay.addWidget(self.hardware)

        lay.addStretch(1)

        self._tm.add_listener(self.apply_palette)

    # ------------------------------------------------ interface sound wiring --
    def _set_interface_sounds(self, enabled: bool) -> None:
        if self._audio:
            self._audio.set_interface_sounds_enabled(enabled)
        self.interface_sounds_toggle.setText("ON" if enabled else "OFF")

    # ------------------------------------------------- hardware wiring --
    def _on_device_selected(self, identifier: str) -> None:
        """Fan the device change out to listeners via HardwareManager."""
        # HardwareManager.set_selected already notified its listeners; kept as
        # a hook for future UI-side side effects (e.g. VRAM heatmap)."""

    # ----------------------------------------------------- audio wiring --
    def _toggle_play(self) -> None:
        if self._audio and self._audio.available:
            if self._audio.is_playing():
                self._audio.stop()
                self.play_btn.setText("▶  Play")
            else:
                self._audio.play()
                self.play_btn.setText("⏸  Pause")
        else:
            self.play_btn.setText("▶  No file")

    def _on_mute(self, checked: bool) -> None:
        if self._audio:
            self._audio.set_muted(checked)
        self.mute_btn.setText("🔊  Unmute" if checked else "🔇  Mute")

    def _nudge_volume(self, delta: int) -> None:
        self.volume.setValue(max(0, min(100, self.volume.value() + delta)))

    def _on_rgb(self, key: str, rgb) -> None:
        if rgb is None:
            self._tm.clear_custom(key)
        else:
            self._tm.set_custom_rgb(key, rgb)

    # ---------------------------------------------------- window chrome --
    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.LeftButton and e.pos().y() < 48:
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._drag_pos is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, _e):  # noqa: N802
        self._drag_pos = None

    # ---------------------------------------------------------- open/attach
    def toggle_near(self, anchor: QWidget | None = None) -> None:
        """Toggle visibility. When showing, anchor near the target widget."""
        if self.isVisible():
            self.hide()
            return
        self.show()
        self.raise_()
        self.activateWindow()
        if anchor:
            g = anchor.mapToGlobal(anchor.rect().topRight())
            x = g.x() + 16
            y = max(40, g.y() - 10)
            screen = QApplication.primaryScreen().availableGeometry()
            if x + self.width() > screen.right():
                x = max(10, g.x() - self.width() - anchor.width() - 16)
            self.move(x, min(y, max(40, screen.bottom() - self.height() - 30)))

    # Alias for backwards compatibility
    def show_drawer(self) -> None:
        self.toggle_near()

    def hide_drawer(self) -> None:
        self.hide()

    def toggle(self) -> None:
        self.toggle_near()

    # ---------------------------------------------------------- theming --
    def apply_palette(self, p: Palette) -> None:
        self._palette = p
        for row in self._accent_rows:
            row.sync_from_palette(p)
        for name, chip in self._chips.items():
            chip.setChecked(name == p.name)
        self.shell.setStyleSheet(
            f"""
            QFrame#Drawer {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 {p.bg_rise}, stop:1 {p.bg_deep});
                border: 2px solid {p.interactive};
                border-radius: 14px;
            }}
            """
        )


# Backward-compatibility alias
SettingsDrawer = SettingsWindow
