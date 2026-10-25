"""Data-format dispatcher — route datasets to their specialized pipeline.

Complements :mod:`src.physics_dispatcher` (which classifies *objects* by
name/keywords) by classifying **data formats** and routing the raw data to
the correct backend processing pipeline:

* ``fits``       — standard astronomical imaging → inverse-rendering
                   ray-marcher (:mod:`src.ray_marcher`).
* ``mhd``        — magnetohydrodynamic simulation grids → Stokes Q/U
                   polarized-emission pipeline (:mod:`src.losses`).
* ``uvfits``     — EHT / radio interferometric visibilities →
                   interferometric reconstruction (:mod:`src.uvfits_imaging`).
* ``optical``    — optical / spectroscopic frames → spectral-cube mapper
                   (Track-A volumetric marcher with wavelength channels).
* custom formats — registered at runtime via :func:`register_pipeline`,
                   the modular extension hook for future physics models.

Format detection combines file extension, FITS-header keyword inspection
(OBJECT / TELESCOP / CTYPE / BUNIT via astropy when available) and plain
filename heuristics.  Pure-python + lazy astropy: importable without a GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class FormatRoute:
    """Result of the format dispatcher: pipeline id + human description."""
    fmt: str                  # "fits" | "mhd" | "uvfits" | "optical" | custom id
    title: str                # human-readable format name
    pipeline_module: str      # dotted module that implements the pipeline
    description: str          # one-line description for the UI
    reason: str = ""          # how this route was decided (extension/header)
    headers: Dict[str, str] = field(default_factory=dict)  # FITS keywords seen
    handler: Optional[Callable] = None  # set for runtime-registered formats


# ---- extension → format mapping ------------------------------------------
_EXT_MAP: Dict[str, str] = {
    ".fits": "fits", ".fit": "fits", ".fts": "fits",
    ".uvfits": "uvfits", ".uvf": "uvfits",
    ".mhd": "mhd", ".vti": "mhd", ".amrext": "mhd",
    ".npy": "optical", ".npz": "optical", ".png": "optical",
    ".jpg": "optical", ".jpeg": "optical", ".tif": "optical",
    ".tiff": "optical", ".csv": "optical", ".h5": "optical",
    ".hdf5": "optical", ".fits.gz": "fits",
}

# ---- FITS header keyword heuristics ---------------------------------------
_UVFITS_HINTS = ("uvfits", "visibility", "interferom", "baseline", "eht")
_MHD_HINTS = ("mhd", "stokes q", "stokes u", "magnetohydro", "b-field",
              "plasma grid", "simulation grid")
_OPTICAL_HINTS = ("spectrum", "spectral", "optical", "wavelength", "angstrom")


def _probe_fits_header(path: Path) -> Dict[str, str]:
    """Read a few header keywords via astropy (empty dict if unavailable)."""
    try:
        from astropy.io import fits  # lazy import: GUI stays launchable
        with fits.open(str(path), memmap=False) as hdus:
            hdr = hdus[0].header
            return {k: str(hdr[k]) for k in
                    ("OBJECT", "TELESCOP", "CTYPE1", "BUNIT") if k in hdr}
    except Exception:
        return {}


def detect_format(path) -> FormatRoute:
    """Detect the data format of *path* and return its pipeline route.

    Order of evidence: FITS header keywords (most specific) → file
    extension → filename heuristics → generic fits fallback.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    name = p.name.lower()

    # ---- 1. FITS header inspection (authoritative for .fits family) ------
    if suffix in (".fits", ".fit", ".fts") or name.endswith(".fits.gz"):
        headers = _probe_fits_header(p)
        blob = " ".join(headers.values()).lower()
        if any(h in blob for h in _UVFITS_HINTS):
            return FormatRoute("uvfits", "Interferometry / UV-FITS",
                               "src.uvfits_imaging",
                               "EHT / radio baseline visibility reconstruction",
                               reason="FITS header keywords", headers=headers)
        if any(h in blob for h in _MHD_HINTS):
            return FormatRoute("mhd", "MHD Format",
                               "src.losses",
                               "Stokes Q/U magnetohydrodynamic pipeline",
                               reason="FITS header keywords", headers=headers)
        if any(h in blob for h in _OPTICAL_HINTS):
            return FormatRoute("optical", "Optical / Spectroscopic Format",
                               "src.ray_marcher",
                               "Spectral-cube volumetric ray marching",
                               reason="FITS header keywords", headers=headers)
        return FormatRoute("fits", "FITS Format",
                           "src.ray_marcher",
                           "Standard astronomical imaging — inverse-rendering "
                           "ray-marcher",
                           reason="FITS header/extension", headers=headers)

    # ---- 2. extension table ----------------------------------------------
    if suffix in _EXT_MAP:
        fmt = _EXT_MAP[suffix]
        meta = _FORMAT_META[fmt]
        return FormatRoute(fmt, meta[0], meta[1], meta[2],
                           reason=f"file extension {suffix}")

    # ---- 3. filename heuristics ------------------------------------------
    for hint, fmt in (("uvfits", "uvfits"), ("mhd", "mhd"),
                      ("stokes", "mhd"), ("spec", "optical")):
        if hint in name:
            meta = _FORMAT_META[fmt]
            return FormatRoute(fmt, meta[0], meta[1], meta[2],
                               reason=f"filename contains '{hint}'")

    # ---- 4. fallback: treat as generic imaging ---------------------------
    meta = _FORMAT_META["fits"]
    return FormatRoute("fits", meta[0], meta[1], meta[2],
                       reason="no format evidence — defaulting to FITS imaging")


# Human-readable metadata per canonical format id (title, module, blurb).
_FORMAT_META: Dict[str, tuple] = {
    "fits": ("FITS Format", "src.ray_marcher",
             "Standard astronomical imaging — inverse-rendering ray-marcher"),
    "mhd": ("MHD Format", "src.losses",
            "Stokes Q/U magnetohydrodynamic simulation grids"),
    "uvfits": ("Interferometry / UV-FITS Format", "src.uvfits_imaging",
               "EHT / radio baseline visibility reconstruction"),
    "optical": ("Optical / Spectroscopic Format", "src.ray_marcher",
                "Optical frames & spectral cubes — volumetric marcher"),
}


# ---- runtime registry: the modular extension hook --------------------------
_CUSTOM_FORMATS: Dict[str, FormatRoute] = {}


def register_pipeline(fmt_id: str, title: str, pipeline_module: str,
                      description: str,
                      handler: Optional[Callable] = None) -> None:
    """Register a new data format → pipeline mapping at runtime.

    Example
    -------
    >>> register_pipeline("radio_cubes", "Radio Cube Format", "src.ray_marcher",
    ...                   "Multi-channel radio cube renderer", my_loader_fn)
    Future physics models plug in here without touching the dispatcher.
    """
    _CUSTOM_FORMATS[fmt_id] = FormatRoute(
        fmt_id, title, pipeline_module, description, reason="custom registry",
        handler=handler)


def list_formats() -> List[FormatRoute]:
    """All selectable formats: the four built-ins + runtime-registered ones."""
    built = [FormatRoute(k, *_FORMAT_META[k]) for k in
             ("fits", "mhd", "uvfits", "optical")]
    return built + list(_CUSTOM_FORMATS.values())


def route_data(path, fmt_hint: Optional[str] = None) -> FormatRoute:
    """Public hook: detect (or force via *fmt_hint*) a pipeline for *path*."""
    if fmt_hint and fmt_hint in _FORMAT_META:
        meta = _FORMAT_META[fmt_hint]
        return FormatRoute(fmt_hint, meta[0], meta[1], meta[2],
                           reason="user-selected in format modal")
    if fmt_hint and fmt_hint in _CUSTOM_FORMATS:
        return _CUSTOM_FORMATS[fmt_hint]
    return detect_format(path)
