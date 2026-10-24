"""Theme engine for Astro Recon.

Three built-in themes (Light, Dark, Dark Purple) plus a live RGB customizer
for the Primary / Secondary / Interactive accent colors. Themes are expressed
as QSS (Qt stylesheets) generated from a palette dict, so accent changes
re-render instantly across the whole app.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------- palettes ---

@dataclass
class Palette:
    """All colors the UI is rendered from."""
    name: str
    # base surfaces
    bg_deep: str          # deepest background (space canvas)
    bg_mid: str           # panels / cards
    bg_rise: str          # elevated (modals, drawers)
    border: str
    text: str
    text_dim: str
    # accents (overrideable by the RGB customizer)
    primary: str          # main brand accent (buttons, title glow)
    secondary: str        # secondary accent (headings, panels edges)
    interactive: str      # hover / focus / interactive accent
    # derived
    glow: str = ""
    scrim: str = "rgba(0,0,0,0.55)"

    def __post_init__(self) -> None:
        if not self.glow:
            self.glow = self._glow_of(self.primary)

    @staticmethod
    def _glow_of(hex_color: str) -> str:
        r, g, b = Palette.rgb(hex_color)
        return f"rgba({r},{g},{b},0.35)"

    @staticmethod
    def rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def with_accents(self, primary: str, secondary: str, interactive: str) -> "Palette":
        return replace(self, primary=primary, secondary=secondary, interactive=interactive)


LIGHT = Palette(
    name="Light",
    bg_deep="#eef2f9", bg_mid="#ffffff", bg_rise="#f7f9fd",
    border="#c9d4e6", text="#10141f", text_dim="#5a6a85",
    primary="#3b82f6", secondary="#7c3aed", interactive="#06b6d4",
)

DARK = Palette(
    name="Dark",
    bg_deep="#070b16", bg_mid="#0d1322", bg_rise="#131b2e",
    border="#24304d", text="#e8edff", text_dim="#8b98b8",
    primary="#38bdf8", secondary="#a78bfa", interactive="#22d3ee",
)

DARK_PURPLE = Palette(
    name="Dark Purple",
    bg_deep="#0a0614", bg_mid="#140b26", bg_rise="#1d1236",
    border="#3b2566", text="#f2eaff", text_dim="#a892cf",
    primary="#c084fc", secondary="#f0abfc", interactive="#7dd3fc",
)

THEMES: dict[str, Palette] = {"Light": LIGHT, "Dark": DARK, "Dark Purple": DARK_PURPLE}

# ------------------------------------------------------------ QSS builder ---

def build_qss(p: Palette) -> str:
    """Render the full application stylesheet from a palette."""
    return f"""
* {{
    font-family: 'Segoe UI', 'Inter', 'Rajdhani', sans-serif;
    color: {p.text};
}}
QMainWindow, #RootStack {{ background: {p.bg_deep}; }}

/* ---------- global button (gamer HUD style) ---------- */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {p.bg_rise}, stop:0.48 {p.bg_mid}, stop:1 {p.bg_deep});
    border: 1px solid {p.border};
    border-top-color: {p.interactive};
    border-left-color: {p.interactive};
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 22px;
    font-weight: 600;
    letter-spacing: 0.25px;
}}
QPushButton:hover {{
    border-color: {p.interactive};
    color: {p.interactive};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {p.bg_rise}, stop:1 {p.bg_mid});
}}
QPushButton:pressed {{
    background: {p.bg_deep};
    border-top-color: {p.border};
    border-left-color: {p.border};
    padding-top: 9px;
}}

/* small square icon buttons (+ / - / close / refresh): no padding so
   the glyph fills the button instead of being clipped away */
QPushButton#IconBtn {{
    padding: 0px;
    min-height: 0px;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 0px;
}}

/* primary menu buttons — big, neon-edged */
QPushButton#MenuButton {{
    background: {p.bg_mid};
    border: 1px solid {p.primary};
    border-left: 4px solid {p.primary};
    border-radius: 12px;
    padding: 3px 10px;       /* compact enough for 46px footer controls */
    min-height: 28px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-align: left;
}}
QPushButton#MenuButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {p.bg_rise}, stop:1 {p.bg_mid});
    border-color: {p.interactive};
    border-left: 4px solid {p.interactive};
}}
QPushButton#MenuButton:pressed {{
    background: {p.bg_deep};
    border-top-color: {p.border};
    padding-top: 7px;
}}

/* ---------- WWT-style top menu bar ---------- */
QMenuBar#TopMenuBar {{
    background: {p.bg_mid};
    border: 1px solid {p.primary};
    border-radius: 10px;
    padding: 2px 6px;
    color: {p.text};
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QMenuBar#TopMenuBar::item {{
    background: transparent;
    padding: 6px 14px;
    border-radius: 6px;
    color: {p.text};
}}
QMenuBar#TopMenuBar::item:selected {{
    background: {p.bg_rise};
    color: {p.interactive};
}}
QMenuBar#TopMenuBar::item:pressed {{
    background: {p.primary};
    color: #04060f;
}}
QMenu {{
    background: {p.bg_mid};
    border: 1px solid {p.primary};
    border-radius: 8px;
    padding: 6px;
    color: {p.text};
    font-size: 13px;
}}
QMenu::item {{
    padding: 8px 26px 8px 22px;
    border-radius: 6px;
    color: {p.text};
}}
QMenu::item:selected {{
    background: {p.bg_rise};
    color: {p.interactive};
}}
QMenu::item:checked {{
    color: {p.interactive};
    font-weight: 700;
}}
QMenu::separator {{
    height: 1px;
    background: {p.primary};
    margin: 4px 8px;
}}

/* ---------- title ---------- */
QLabel#Title {{
    font-size: 64px;
    font-weight: 800;
    letter-spacing: 6px;
    color: {p.primary};
}}
QLabel#Subtitle {{
    font-size: 15px;
    color: {p.text_dim};
    letter-spacing: 3px;
}}
QLabel#Footer {{
    color: {p.text_dim};
    font-size: 12px;
    letter-spacing: 0.5px;
}}
QLabel#Section {{ font-size: 13px; font-weight: 700; letter-spacing: 2px; color: {p.secondary}; }}
QLabel#Dim {{ color: {p.text_dim}; }}

/* ---------- panels / cards ---------- */
QFrame#Card {{
    background: {p.bg_mid};
    border: 1px solid {p.border};
    border-radius: 14px;
}}
QFrame#Drawer {{
    background: {p.bg_rise};
    border-left: 1px solid {p.border};
}}

/* ---------- simulation three-panel windows ---------- */
QFrame#PanelWin {{
    background: {p.bg_deep};
    border: 1px solid {p.border};
    border-radius: 10px;
}}
QFrame#PanelTitleBar {{
    background: rgba(255,255,255,18);
    border-bottom: 1px solid {p.border};
}}
QLabel#PanelTitle {{
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    padding: 2px 0;  /* keep the glow letterspacing from clipping at the top */
}}

/* ---------- inputs ---------- */
QLineEdit {{
    background: {p.bg_mid};
    border: 1px solid {p.border};
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 14px;
}}
QLineEdit:focus {{ border-color: {p.interactive}; }}

/* ---------- list of library objects ---------- */
QListWidget {{
    background: {p.bg_mid};
    border: 1px solid {p.border};
    border-radius: 12px;
    padding: 4px;
}}
QListWidget::item {{
    border-radius: 8px;
    padding: 10px 12px;
    margin: 2px;
}}
QListWidget::item:selected {{ background: {p.bg_rise}; color: {p.interactive}; }}
QListWidget::item:hover {{ color: {p.interactive}; }}

/* ---------- sliders ---------- */
QSlider::groove:horizontal {{
    height: 6px; border-radius: 3px;
    background: {p.border};
}}
QSlider::sub-page:horizontal {{ background: {p.primary}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px;
    background: {p.interactive};
    border: 2px solid {p.bg_rise};
}}
QSlider::handle:horizontal:hover {{ background: {p.primary}; }}

/* ---------- theme toggle chips ---------- */
QPushButton#ThemeChip {{
    border-radius: 9px;
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
}}
QPushButton#ThemeChip:checked {{
    border-color: {p.interactive};
    color: {p.interactive};
    background: {p.bg_rise};
}}

/* ---------- dropzone ---------- */
QFrame#Dropzone {{
    background: {p.bg_deep};
    border: 2px dashed {p.secondary};
    border-radius: 14px;
}}
QFrame#Dropzone[dragOver="true"] {{
    border-color: {p.interactive};
    background: {p.bg_rise};
}}

/* ---------- sources modal scroll container ----------
   Strictly vertical scrolling only, full width, no horizontal overflow.
   NOTE: Qt QSS does NOT support the CSS properties `overflow-x`/`overflow-y`
   (Qt logs "Unknown property ..." for them). Scrolling policy is enforced in
   Python via setVerticalScrollBarPolicy / setHorizontalScrollBarPolicy and by
   constraining the inner widget's width instead. */
QScrollArea#SourcesScroll {{
    background: transparent;
}}
QScrollArea#SourcesScroll > QWidget > QWidget {{ background: transparent; }}
QScrollBar:horizontal {{ height: 0px; background: transparent; }}
QScrollBar::handle:horizontal {{ width: 0px; }}

/* ---------- tabs (Metadata / Specs) ----------
   Without these rules Qt falls back to its LIGHT default palette, which is why
   the tab strip and panes were rendering white-on-white in the dark shell.
   NOTE: styling only QTabWidget::pane leaves the tab widget's OWN background
   (the strip beside the tabs and the margin around the pane) painted with the
   default light base colour — that was the white block below the control bar.
   The QTabWidget rule below paints the whole strip dark. */
QTabWidget {{
    background: {p.bg_mid};
}}
QTabWidget::pane {{
    background: {p.bg_mid};
    border: 1px solid {p.border};
    border-radius: 10px;
    top: -1px;
}}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: {p.bg_deep};
    color: {p.text_dim};
    border: 1px solid {p.border};
    border-bottom: none;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    padding: 8px 18px 8px 18px;
    margin-right: 3px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QTabBar::tab:selected {{
    background: {p.bg_rise};
    color: {p.interactive};
    border-color: {p.interactive};
}}
QTabBar::tab:hover {{ color: {p.interactive}; }}

/* ---------- text panes (read-only metadata boxes) ---------- */
QPlainTextEdit, QTextEdit {{
    background: {p.bg_deep};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: 8px;
    selection-background-color: {p.primary};
    selection-color: {p.text};
}}
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {p.interactive}; }}

/* ---------- combo boxes ---------- */
QComboBox {{
    background: {p.bg_deep};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 5px 10px;
}}
QComboBox:hover {{ border-color: {p.interactive}; }}
QComboBox QAbstractItemView {{
    background: {p.bg_mid};
    color: {p.text};
    border: 1px solid {p.border};
    selection-background-color: {p.bg_rise};
    selection-color: {p.interactive};
    outline: none;
}}

/* ---------- progress bar ----------
   Qt's default "Fusion" progress bar paints a light groove with a white
   chunk, which read as a stray white band on the dark pipeline card. */
QProgressBar {{
    background: {p.bg_deep};
    border: 1px solid {p.border};
    border-radius: 7px;
    height: 16px;
    text-align: center;
    color: {p.text};
    font-size: 11px;
    font-weight: 600;
}}
QProgressBar::chunk {{
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {p.primary}, stop:1 {p.interactive});
}}

/* ---------- misc ---------- */
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 2px;
}}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p.interactive}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{
    background: {p.bg_rise}; color: {p.text};
    border: 1px solid {p.border}; padding: 4px;
}}
"""


# --------------------------------------------------------- theme manager ----

class ThemeManager:
    """Holds the current palette, applies QSS, and applies custom RGB accents."""

    def __init__(self) -> None:
        self.palette: Palette = DARK
        self.custom: dict[str, tuple[int, int, int] | None] = {
            "primary": None, "secondary": None, "interactive": None,
        }
        self.listeners: list = []  # callables(palette) invoked on change

    # -- theme selection ---------------------------------------------------
    def set_theme(self, name: str) -> None:
        self.palette = THEMES.get(name, DARK)
        self._apply()

    # -- RGB customizer ----------------------------------------------------
    def set_custom_rgb(self, key: str, rgb: tuple[int, int, int]) -> None:
        """key in {'primary','secondary','interactive'}; rgb = (r, g, b) 0-255."""
        self.custom[key] = rgb
        self._apply()

    def clear_custom(self, key: str | None = None) -> None:
        if key is None:
            for k in self.custom:
                self.custom[k] = None
        else:
            self.custom[key] = None
        self._apply()

    def _effective(self) -> Palette:
        p = self.palette
        overrides = {
            k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in self.custom.items() if v
        }
        if overrides:
            p = p.with_accents(
                overrides.get("primary", p.primary),
                overrides.get("secondary", p.secondary),
                overrides.get("interactive", p.interactive),
            )
        return p

    # -- wiring ------------------------------------------------------------
    def add_listener(self, fn) -> None:
        self.listeners.append(fn)
        fn(self._effective())  # sync new widget immediately

    def _apply(self) -> None:
        p = self._effective()
        for fn in self.listeners:
            try:
                fn(p)
            except Exception:  # a dead listener must never break the app
                pass
