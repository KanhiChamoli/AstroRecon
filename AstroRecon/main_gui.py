"""Astro Recon — GUI entry point.

Run:  python main_gui.py
Requires: PyQt5 (see requirements.txt)

Optional background music:
    place the Zelda track (https://www.youtube.com/watch?v=YxbCEfPeANo) as
    assets/audio/background_music.mp3 — the app loops it automatically.
"""
import sys

from PyQt5.QtCore import Qt, QCoreApplication
# QtWebEngine requires shared OpenGL contexts — must be set before QApplication.
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Astro Recon")
    app.setOrganizationName("Aphellion Studios")
    window = MainWindow()
    window.show()
    print("Astro Recon window open — close it to exit.", flush=True)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
