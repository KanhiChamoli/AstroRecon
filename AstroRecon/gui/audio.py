"""Background-music controller for Astro Recon.

Uses pygame.mixer for reliable, immediate MP3 playback on Windows across all
environments (avoiding DirectShow / WMF backend issues in PyQt5).
Loops the user's requested track on loop from the moment the app opens.
"""
from __future__ import annotations

import os
import math
import struct
import threading
from PyQt5.QtCore import QObject, pyqtSignal

# Credit info required by the spec — surfaced in the Sources / credits modal.
MUSIC_CREDIT = {
    "track": "Astral Observatory",
    "game": "The Legend of Zelda: Majora's Mask",
    "contributing_artists": "Koji Kondo, Toru Minegishi",
    "album": "The Legend of Zelda: Majora's Mask Original Soundtrack",
    "source_notes": "Extended Brawl BRSTM re-upload (BGM)",
    "url": "https://www.youtube.com/watch?v=YxbCEfPeANo",
}

_EXPLICIT_AUDIO_PATH = os.path.join(
    os.path.dirname(__file__),
    "AUDIO",
    "Astral Observatory _ The Legend of Zelda Majoras Mask Music Extended (Brawl Brstm Reupload) [YxbCEfPeANo] (1).mp3"
)

_ASSET_DIRS = [
    os.path.join(os.path.dirname(__file__), "AUDIO"),
    os.path.join(os.path.dirname(__file__), "audio"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "audio"),
    os.path.join(os.getcwd(), "assets", "audio"),
]
_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")


def find_music_file() -> str | None:
    # 1. First priority: user-provided explicit track path
    if os.path.isfile(_EXPLICIT_AUDIO_PATH):
        return _EXPLICIT_AUDIO_PATH

    # 2. Search inside gui/AUDIO first
    gui_audio_dir = os.path.join(os.path.dirname(__file__), "AUDIO")
    if os.path.isdir(gui_audio_dir):
        for fn in sorted(os.listdir(gui_audio_dir)):
            if fn.lower().endswith(_EXTS):
                return os.path.join(gui_audio_dir, fn)

    # 3. Fallback search dirs
    for d in _ASSET_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(_EXTS) and ("astral" in fn.lower() or "zelda" in fn.lower()):
                return os.path.join(d, fn)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(_EXTS):
                return os.path.join(d, fn)
    return None


class AudioController(QObject):
    """Looping background music with volume / mute control powered by pygame."""

    state_changed = pyqtSignal(bool)   # playing?
    volume_changed = pyqtSignal(int)   # 0-100 effective volume

    def __init__(self) -> None:
        super().__init__()
        self.base_volume = 40   # slider value before mute
        self.muted = False
        self.available = False
        self.loaded_path: str | None = None
        self._pygame_ready = False
        self.interface_sounds_enabled = True
        self.interface_sounds_volume = 35
        self._click_sounds = []
        self._click_index = 0
        self._click_lock = threading.Lock()

        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._pygame_ready = True
        except Exception as e:
            print("Failed to initialize pygame.mixer:", e)

        self._build_click_sounds()
        path = find_music_file()
        if path and self._pygame_ready:
            try:
                import pygame
                self.loaded_path = path
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(self.base_volume / 100.0)
                self.available = True
            except Exception as e:
                print("Failed to load music into pygame:", e)

    def _build_click_sounds(self) -> None:
        'Build three tiny synthesized retro bleeps without external assets.'
        if not self._pygame_ready:
            return
        try:
            import pygame
            rate = 22050
            for frequency, duration in ((740, 0.045), (980, 0.050), (620, 0.065)):
                count = int(rate * duration)
                data = bytearray()
                for i in range(count):
                    envelope = min(1.0, i / 90.0) * max(0.0, 1.0 - i / count)
                    wave = math.sin(2.0 * math.pi * frequency * i / rate)
                    sample = int(32767 * 0.24 * envelope * (1.0 if wave >= 0 else -1.0))
                    data.extend(struct.pack("<h", sample))
                self._click_sounds.append(pygame.mixer.Sound(buffer=bytes(data)))
        except Exception:
            self._click_sounds = []

    def play_interface_click(self) -> None:
        if not self.interface_sounds_enabled or not self._click_sounds:
            return
        try:
            with self._click_lock:
                sound = self._click_sounds[self._click_index % len(self._click_sounds)]
                self._click_index += 1
            sound.set_volume(self.interface_sounds_volume / 100.0)
            sound.play()
        except Exception:
            pass
    def set_interface_sounds_enabled(self, enabled: bool) -> None:
        self.interface_sounds_enabled = bool(enabled)

    def set_interface_sounds_volume(self, value: int) -> None:
        self.interface_sounds_volume = max(0, min(100, int(value)))

    # ------------------------------------------------------------- API ---
    def play(self) -> None:
        if self.available and self._pygame_ready:
            try:
                import pygame
                # -1 loops infinitely
                pygame.mixer.music.play(-1)
                self.state_changed.emit(True)
            except Exception as e:
                print("pygame play error:", e)

    def stop(self) -> None:
        if self.available and self._pygame_ready:
            try:
                import pygame
                pygame.mixer.music.stop()
                self.state_changed.emit(False)
            except Exception as e:
                print("pygame stop error:", e)

    def pause(self) -> None:
        if self.available and self._pygame_ready:
            try:
                import pygame
                pygame.mixer.music.pause()
                self.state_changed.emit(False)
            except Exception as e:
                print("pygame pause error:", e)

    def unpause(self) -> None:
        if self.available and self._pygame_ready:
            try:
                import pygame
                pygame.mixer.music.unpause()
                self.state_changed.emit(True)
            except Exception as e:
                print("pygame unpause error:", e)

    def is_playing(self) -> bool:
        if self.available and self._pygame_ready:
            try:
                import pygame
                return pygame.mixer.music.get_busy()
            except Exception:
                return False
        return False

    def set_volume(self, value: int) -> None:
        self.base_volume = max(0, min(100, int(value)))
        self._apply()

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self._apply()

    def toggle_mute(self) -> None:
        self.set_muted(not self.muted)

    def _apply(self) -> None:
        eff = 0 if self.muted else self.base_volume
        if self.available and self._pygame_ready:
            try:
                import pygame
                pygame.mixer.music.set_volume(eff / 100.0)
            except Exception as e:
                print("Failed setting pygame volume:", e)
        self.volume_changed.emit(eff)

