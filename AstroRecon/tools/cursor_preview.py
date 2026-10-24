"""Render the telescope cursor sprite to a PNG for visual verification.

Usage:  python tools/cursor_preview.py [out_path] [scale]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PyQt5.QtWidgets import QApplication  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QImage, QPainter  # noqa: E402

from gui.main_window import InterfaceFeedback  # noqa: E402


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/renders/cursor_preview.png"
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    app = QApplication([])
    img = InterfaceFeedback._render_telescope_pixmap("#eec12a").toImage()
    img = img.convertToFormat(QImage.Format_ARGB32)

    # nearest-neighbour upscale onto a flat gray backdrop, done pixel-by-pixel
    w, h = img.width(), img.height()
    up = QImage(w * scale, h * scale, QImage.Format_RGB32)
    bg = img.pixelColor(0, 0) if False else None
    gray = 0xD8D8D8
    for y in range(h):
        for x in range(w):
            c = img.pixel(x, y)
            a = (c >> 24) & 0xFF
            r, g, b = (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF
            if a == 0:
                r, g, b = 0xD8, 0xD8, 0xD8
            up.setPixel(x * scale, y * scale, 0xFF000000 | (r << 16) | (g << 8) | b)
    up.save(out)
    print("saved", out, w, "x", h, "scale", scale)


if __name__ == "__main__":
    main()
