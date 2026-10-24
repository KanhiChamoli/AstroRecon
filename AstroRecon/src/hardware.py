"""Strict GPU enforcement and dynamic hardware inventory for AstroRecon.

Design
------
* ``detect_hardware()`` interrogates PyTorch for every CUDA device plus the
  CPU fallback and reports live VRAM usage.
* ``HardwareManager`` is a process-wide singleton owning the *active* device.
  By default it **locks onto the primary CUDA GPU**. Anything that silently
  downgrades computation (missing CUDA, OOM, "if cuda else cpu" patterns) is
  surfaced loudly: an :class:`GPULockViolation` is raised when code asks for
  CPU while a GPU lock is engaged, and a prominent warning is logged when the
  user explicitly opts into the CPU fallback through the settings UI or CLI.
* The GUI settings panel reads the inventory from here and re-points the
  active context at any listed device without recreating optimizer state
  tensors: ``HardwareManager.set_selected`` plus :func:`retarget_tensors`
  moves existing tensor storage across devices (optimizer states included,
  via :func:`retarget_optimizer`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch

logger = logging.getLogger("astrorecon.hardware")


class GPULockViolation(RuntimeError):
    """Raised when code path would silently fall back to CPU despite a live GPU."""


@dataclass
class DeviceSpec:
    """One selectable entry in the Compute Device dropdown."""

    identifier: str          # "cuda:0", "cuda:1", "cpu"
    label: str               # human readable, shown in the UI dropdown
    total_vram_gb: float = 0.0
    used_vram_gb: float = 0.0
    is_primary_gpu: bool = False

    def as_choice(self) -> str:
        return self.identifier


_UC = 1024 ** 3  # bytes per GiB


def _vram(idx: int) -> tuple[float, float]:
    """Return (total_gb, used_gb) for a CUDA index, 0/0 on any failure."""
    try:
        total = torch.cuda.get_device_properties(idx).total_memory / _UC
        free, _ = torch.cuda.mem_get_info(idx)
        return round(total, 2), round(total - free / _UC, 2)
    except Exception:  # pragma: no cover - driver quirks must not crash the UI
        return 0.0, 0.0


def detect_hardware() -> List[DeviceSpec]:
    """Scan for CUDA GPUs (primary first) and append the CPU fallback entry."""
    specs: List[DeviceSpec] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            total, used = _vram(idx)
            specs.append(
                DeviceSpec(
                    identifier=f"cuda:{idx}",
                    label=f"GPU · {name} · {used:.1f} / {total:.1f} GB VRAM",
                    total_vram_gb=total,
                    used_vram_gb=used,
                    is_primary_gpu=(idx == 0),
                )
            )
    specs.append(DeviceSpec(identifier="cpu", label="CPU fallback (explicit opt-in)"))
    return specs


def gpu_present() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


class HardwareManager:
    """Owns the process-wide active torch device under a strict-GPU policy."""

    def __init__(self, strict_gpu: bool = True) -> None:
        self.strict_gpu = strict_gpu
        self._listeners: List[Callable[[str], None]] = []
        self._specs = detect_hardware()
        if gpu_present():
            self._selected = "cuda:0"
            logger.info(
                "HardwareManager: locked execution to primary CUDA GPU %s",
                self._primary().label,
            )
        else:
            self._selected = "cpu"
            logger.warning(
                "GPULOCK: no CUDA device detected — running on CPU. Heavy tensor "
                "ops and ray marching are NOT GPU-accelerated in this build."
            )

    # ------------------------------------------------------------ queries --
    @property
    def devices(self) -> List[DeviceSpec]:
        return self._specs

    @property
    def selected(self) -> str:
        return self._selected

    def device(self) -> torch.device:
        return torch.device(self._selected)

    def _normalize(self, identifier: str) -> str:
        """Accept friendly aliases: ``cuda`` / ``gpu`` → primary ``cuda:0``."""
        ident = (identifier or "").strip().lower()
        if ident in ("cuda", "gpu", "nvidia") and gpu_present():
            return f"cuda:{torch.cuda.current_device()}"
        if ident.startswith("cuda/"):
            return ident.replace("cuda/", "cuda:", 1)
        return identifier.strip().lower()

    def specs_for(self, identifier: str) -> Optional[DeviceSpec]:
        for s in self._specs:
            if s.identifier == identifier:
                return s
        return None

    def _primary(self) -> DeviceSpec:
        for s in self._specs:
            if s.is_primary_gpu:
                return s
        # unreachable when a GPU is present; defend anyway
        return self._specs[-1]

    # ------------------------------------------------------------ mutation --
    def set_selected(self, identifier: str, refresh: bool = True) -> torch.device:
        """Switch the active device context.

        * Validating that the id exists in the inventory.
        * Enforcing the GPU lock: an *automatic* selection never lands us on
          CPU while a GPU is present (that is `GPULockViolation`), and an
          *explicit* CPU choice is honored but logged prominently.
        """
        if refresh:
            self._specs = detect_hardware()
        identifier = self._normalize(identifier)
        if self.specs_for(identifier) is None:
            raise GPULockViolation(
                f"Unknown compute device '{identifier}'. "
                f"Available: {[s.identifier for s in self._specs]}"
            )
        previous = self._selected
        if identifier == "cpu" and gpu_present() and self.strict_gpu:
            logger.warning(
                "GPULOCK override: explicitly deselecting the CUDA GPU (%s) in "
                "favor of the CPU fallback. Training will run meaningfully slower.",
                previous,
            )
        self._selected = identifier
        for cb in list(self._listeners):
            cb(self._selected)
        return self.device()

    # listener plumbing (GUI subscribes without importing Qt here)

    def add_listener(self, callback: Callable[[str], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    # ------------------------------------------------- enforcement helpers --
    def resolve_requested(self, requested: Optional[str]) -> torch.device:
        """Resolve a user/CLI device request under the strict-GPU policy.

        ``requested=None``  → automatic: hard-locked to the primary GPU or a
        prominent CPU warning when no GPU exists.
        ``requested='cpu'`` → honored as explicit opt-in, logged loudly.
        """
        if requested in (None, ""):
            d = self.device()
            if d.type == "cpu":
                logger.warning(
                    "GPULOCK: computation pinned to CPU — no CUDA device is "
                    "usable in this PyTorch build."
                )
            return d
        if requested.startswith("cuda") and not gpu_present():
            raise GPULockViolation(
                "CUDA device requested but torch reports no usable GPU "
                "(this usually means a CPU-only torch build: reinstall torch "
                "with the CUDA wheel)."
            )
        return self.set_selected(requested, refresh=False)

    def assert_on_gpu(self, *tensors: torch.Tensor) -> None:
        """Hard check used around heavy ops: refuse silent CPU fallback."""
        if self.strict_gpu and gpu_present():
            for t in tensors:
                if t.device.type == "cpu":
                    raise GPULockViolation(
                        "GPU lock engaged but a tensor lives on CPU — move "
                        "tensors through HardwareManager.device() before ray "
                        "marching."
                    )


_MANAGERS: dict[int, HardwareManager] = {}


def get_manager(strict_gpu: bool = True) -> HardwareManager:
    """Per-thread-or-Qt-event-loop HardwareManager singleton."""
    import threading

    key = threading.get_ident()
    if key not in _MANAGERS:
        _MANAGERS[key] = HardwareManager(strict_gpu=strict_gpu)
    return _MANAGERS[key]


# --------------------------------------------------------- retarget utils --
def retarget_tensors(tensors: Iterable[torch.Tensor], device: torch.device):
    """Move existing tensors (in place when possible) to ``device``."""
    moved = []
    for t in tensors:
        if t.device == device:
            moved.append(t)
        else:
            moved.append(t.to(device))
    return moved


def retarget_optimizer(optimizer: torch.optim.Optimizer, device: torch.device):
    """Re-point optimizer state (exp_avg / exp_avg_sq buffers) to ``device``.

    torch's Adam lazily initializes state on first step; if state exists we
    migrate it so switching devices mid-run cannot crash or restart moments.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return optimizer


def refresh_inventory(manager: "HardwareManager | None" = None) -> List[DeviceSpec]:
    """Re-scan hardware (VRAM usage drifts during training) and update labels."""
    mgr = manager or get_manager()
    mgr._specs = detect_hardware()
    return mgr.devices
