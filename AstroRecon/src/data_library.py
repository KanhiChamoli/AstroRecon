"""Unified Astronomical Data Library for AstroRecon.

A centralized registry and dispatcher that connects AstroRecon's search bar
and "Sources & Credits" box to humanity's major telescope archives via
``astroquery`` and Astropy:

* **NASA SkyView**      - all-sky survey cutouts (DSS, WISE, Fermi, NVSS...)
* **NASA MAST**         - JWST, Hubble (HST), TESS and Kepler observations
* **CDS VizieR / SIMBAD** - catalog queries for stars, galaxies and nebulae
* **HEASARC / Chandra (CXC)** - high-energy X-ray / gamma-ray data
* **AAS WorldWide Telescope** - all-sky survey pyramid & WTML-style metadata

Design goals
------------
1. Everything is cached on disk under ``data/library_cache/`` so repeated
   searches never re-download the same dataset.
2. Downloads land in a structured layout that AstroRecon's FITS loaders,
   ``fits_router`` and PyTorch inverse-rendering pipelines can consume
   directly (paths are returned, never in-memory blobs).
3. An integrated object classifier mirrors ``physics_dispatcher`` so the UI
   can route a target to the right rendering track (GRRT null geodesics vs.
   optically-thin radiative transfer).
4. All archive access is lazy and fault tolerant: a missing/broken
   ``astroquery`` install or an offline machine degrades gracefully instead
   of crashing the GUI.

Typical usage
-------------
>>> from src.data_library import AstroLibraryManager
>>> lib = AstroLibraryManager()
>>> route = lib.get_dispatch_route("Sagittarius A*")   # -> PipelineRoute
>>> lib.get_available_surveys()["skyview"]["surveys"]  # UI dropdown data
>>> path = lib.search_and_fetch("M13", "mast", "HST")  # local FITS path
>>> lib.list_cached_datasets()                          # Sources & Credits

(c) AstroRecon - Y2K Virtual Observatory. Data terms of use follow each
upstream archive (STScI MAST, CDS/SIMBAD, HEASARC, Chandra X-ray Center).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # safe conditional import - module degrades to path-only mode without astropy
    from astropy import units as u
    from astropy.coordinates import SkyCoord
except ImportError:  # pragma: no cover
    u = None
    SkyCoord = None

from .log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Import the physics dispatcher (same package) for the classification hook.
try:
    from .physics_dispatcher import PipelineRoute, classify_query
except ImportError:  # pragma: no cover - running as a top-level script
    from physics_dispatcher import PipelineRoute, classify_query


# --------------------------------------------------------------------------
# Lazy astroquery imports
# --------------------------------------------------------------------------
def _try_import(path: str):
    """Import a dotted object path like ``astroquery.skyview.SkyView``.

    The last component may be either a submodule or a class inside the
    module; returns the object, or ``None`` if unavailable.
    """
    import importlib

    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover
            logger.debug("astroquery import failed for %s: %s", module_name, exc)
            continue
        obj = module
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            logger.debug("astroquery attribute lookup failed for %s: %s", path, exc)
            continue
        return obj
    return None


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class CachedDataset:
    """Metadata for one locally stored archive product.

    This is the record shown in the UI "Sources & Credits" box and handed
    to the FITS loader / PyTorch pipelines.
    """

    path: str
    target: str
    archive: str
    survey: str
    object_class: str
    pipeline: str
    file_size_mb: float
    cached_at: str
    credits: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _CacheIndex:
    """Thread-safe on-disk JSON index of everything in the cache."""

    path: Path
    entries: Dict[str, CachedDataset] = field(default_factory=dict)
    _lock: threading.RLock = threading.RLock()

    # -- persistence -------------------------------------------------
    def load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for key, rec in raw.items():
                    self.entries[key] = CachedDataset(**rec)
                logger.debug("cache index loaded: %d entries", len(self.entries))
            except Exception as exc:
                logger.warning("cache index unreadable (%s); starting fresh", exc)

    def save(self) -> None:
        with self._lock:
            payload = {k: v.to_dict() for k, v in self.entries.items()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    # -- operations --------------------------------------------------
    def key_for(self, target: str, archive: str, survey: str) -> str:
        blob = f"{target.lower().strip()}::{archive.lower()}::{survey.lower()}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[CachedDataset]:
        with self._lock:
            entry = self.entries.get(key)
        if entry is None:
            return None
        if not Path(entry.path).exists():
            with self._lock:
                self.entries.pop(key, None)
                self.save()
            return None
        return entry

    def put(self, entry: CachedDataset) -> None:
        key = self.key_for(entry.target, entry.archive, entry.survey)
        with self._lock:
            self.entries[key] = entry
            self.save()

    def all(self) -> List[CachedDataset]:
        # Drop stale entries whose files were removed externally.
        for key, entry in list(self.entries.items()):
            if not Path(entry.path).exists():
                with self._lock:
                    self.entries.pop(key, None)
        self.save()
        return list(self.entries.values())


# --------------------------------------------------------------------------
# Archive registry
# --------------------------------------------------------------------------
ARCHIVE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "skyview": {
        "label": "NASA SkyView (GSFC)",
        "description": "All-sky survey image cutouts across radio, IR, "
                       "optical, UV and X-ray spectra.",
        "service": "image",
        "credits": "NASA/GSFC SkyView (https://skyview.gsfc.nasa.gov)",
        "surveys": {
            "DSS": {"name": "DSS2 Optical", "band": "optical", "size": 1.0},
            "DSS2 Red": {"name": "DSS2 Red", "band": "optical", "size": 1.0},
            "WISE 12": {"name": "WISE 12 micron", "band": "infrared", "size": 0.5},
            "2MASS": {"name": "2MASS K-band", "band": "infrared", "size": 0.3},
            "NVSS": {"name": "NRAO VLA Sky Survey", "band": "radio", "size": 1.0},
            "VLSS": {"name": "VLA Low-frequency Sky Survey", "band": "radio", "size": 2.0},
            "Fermi 5": {"name": "Fermi LAT 5-year Gamma", "band": "gamma", "size": 5.0},
            "RASS": {"name": "ROSAT All-Sky X-ray", "band": "xray", "size": 3.0},
            "GALEX Near UV": {"name": "GALEX NUV", "band": "ultraviolet", "size": 0.2},
        },
    },
    "mast": {
        "label": "NASA MAST (STScI)",
        "description": "Space-based observations from JWST, Hubble (HST), "
                       "TESS and Kepler.",
        "service": "observations",
        "credits": "STScI / NASA MAST (https://mast.stsci.edu)",
        "missions": {
            "JWST": {"name": "James Webb Space Telescope", "band": "infrared"},
            "HST": {"name": "Hubble Space Telescope", "band": "optical/uv"},
            "TESS": {"name": "Transiting Exoplanet Survey Satellite", "band": "optical"},
            "Kepler": {"name": "Kepler Space Telescope", "band": "optical"},
        },
    },
    "vizier": {
        "label": "CDS VizieR (Strasbourg)",
        "description": "Definitive astronomical catalog repository "
                       "(billions of stellar & galactic records).",
        "service": "catalog",
        "credits": "CDS Strasbourg (https://vizier.u-strasbg.fr)",
        "catalogs": {
            "J/ApJS/199/26": "SDSS quasar catalog",
            "V/154": "HST guide star catalog",
            "VII/259": "Open cluster catalog",
        },
    },
    "simbad": {
        "label": "CDS SIMBAD (Strasbourg)",
        "description": "Object identification, coordinates and cross-IDs.",
        "service": "identifier",
        "credits": "CDS Strasbourg (https://simbad.u-strasbg.fr)",
    },
    "heasarc": {
        "label": "NASA HEASARC / Chandra CXC",
        "description": "High-energy X-ray and gamma-ray observations "
                       "(thermal plasma, accretion-disk emission).",
        "service": "xray",
        "credits": "NASA HEASARC (https://heasarc.gsfc.nasa.gov) · "
                   "Chandra X-ray Center (https://cxc.harvard.edu)",
        "tables": {
            "chanmaster": "Chandra X-ray Center master observation list",
            "xtemaster": "XMM-Newton master source catalog",
            "rospsrc": "ROSAT PSPC pointed observations",
        },
    },
    "wwt": {
        "label": "AAS WorldWide Telescope",
        "description": "Interactive all-sky survey pyramids and WTML-style "
                       "streamable catalog formats.",
        "service": "streaming",
        "credits": "AAS WorldWide Telescope (https://worldwidetelescope.org)",
        "surveys": {
            "DSS": "Digitized Sky Survey (WWT pyramid)",
            "2MASS": "Two Micron All-Sky Survey",
            "WISE": "Wide-field Infrared Survey Explorer",
            "Fermi": "Fermi LAT all-sky gamma",
        },
    },
}

_COORD_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)$|"
    r"(\d{1,2})[h:\s]+(\d{1,2})[m:\s]+([\d.]+)s?\s+"
    r"([+-]?\d{1,2})[d:\s°]+(\d{1,2})[m:\s']+([\d.]+)\"?)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Main manager
# --------------------------------------------------------------------------
class AstroLibraryManager:
    """Unified access layer over the major astronomical archives.

    One instance is cheap to create and safe to reuse for the lifetime of
    the GUI; every public method is thread-safe thanks to an internal lock
    around astroquery's per-module singletons.
    """

    def __init__(self, cache_root: Optional[Path] = None) -> None:
        self.cache_root = Path(cache_root or Path("data") / "library_cache").resolve()
        self.raw_dir = self.cache_root.parent / "raw"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self._index = _CacheIndex(self.cache_root / "index.json")
        self._index.load()
        self._net_lock = threading.RLock()

        # Lazy client handles
        self._skyview = None
        self._mast = None
        self._mast_observations = None
        self._vizier = None
        self._simbad = None
        self._heasarc = None

        logger.info("AstroLibraryManager ready (cache=%s)", self.cache_root)

    # ------------------------------------------------------------------
    # Client accessors (lazy)
    # ------------------------------------------------------------------
    @property
    def skyview(self):
        if self._skyview is None:
            self._skyview = _try_import("astroquery.skyview.SkyView") or False
        return self._skyview or None

    @property
    def mast(self):
        if self._mast is None:
            obs = _try_import("astroquery.mast.Observations")
            self._mast_observations = obs.Observations if obs else False
            self._mast = bool(obs)
        return self._mast_observations or None

    @property
    def vizier(self):
        if self._vizier is None:
            self._vizier = _try_import("astroquery.vizier.Vizier") or False
        return self._vizier or None

    @property
    def simbad(self):
        if self._simbad is None:
            self._simbad = _try_import("astroquery.simbad.Simbad") or False
        return self._simbad or None

    @property
    def heasarc(self):
        if self._heasarc is None:
            self._heasarc = _try_import("astroquery.heasarc.HEASARC") or False
        return self._heasarc or None

    # ------------------------------------------------------------------
    # Object classification / dispatcher hook
    # ------------------------------------------------------------------
    def classify_object(self, target_name: str) -> PipelineRoute:
        """Categorize a search target for UI pipeline routing.

        Delegates keyword heuristics to ``physics_dispatcher.classify_query``
        so GUI dispatch and data-library dispatch always agree.
        """
        return classify_query(target_name)

    def get_dispatch_route(self, target_name: str) -> Dict[str, str]:
        """Human-readable classification summary for the search UI."""
        route = self.classify_object(target_name)
        return {
            "pipeline": route.pipeline,
            "track": route.track,
            "object_class": route.object_class,
            "reason": route.reason,
        }

    # ------------------------------------------------------------------
    # Coordinate resolution
    # ------------------------------------------------------------------
    def resolve_coordinates(self, target_name: str) -> Optional[SkyCoord]:
        """Resolve a target name (or 'ra dec' pair) to a SkyCoord.

        Accepts free-form names, IAU-style designations, and decimal or
        sexagesimal coordinate strings.  Falls back to SIMBAD then NED-style
        name parsing via Astropy.  Returns ``None`` if astropy is missing.
        """
        if SkyCoord is None:  # astropy unavailable
            logger.error("astropy not installed - cannot resolve coordinates")
            return None
        text = (target_name or "").strip()
        if not text:
            return None

        # Plain 'ra dec' decimal pair?
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)$", text)
        if m:
            try:
                return SkyCoord(float(m.group(1)) * u.deg, float(m.group(2)) * u.deg,
                                frame="icrs")
            except ValueError:
                pass

        if self.simbad:
            with self._net_lock:
                try:
                    result = self.simbad.query_object(self._sanitize_name(text))
                    if result is not None and len(result) > 0:
                        # astroquery >= 0.4.8 uses lowercase column names;
                        # older versions used uppercase. Handle both.
                        cols = {c.upper(): c for c in result.colnames}
                        ra_col, dec_col = cols.get("RA"), cols.get("DEC")
                        if not (ra_col and dec_col):
                            return None
                        ra, dec = result[ra_col][0], result[dec_col][0]
                        if isinstance(ra, str) or isinstance(dec, str):
                            return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                        return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
                except Exception as exc:
                    logger.warning("SIMBAD resolve failed for %r: %s", text, exc)

        try:
            return SkyCoord.from_name(text)
        except Exception as exc:
            logger.warning("Coordinate lookup failed for %r: %s", text, exc)
            return None

    @staticmethod
    def _sanitize_name(text: str) -> str:
        """SIMBAD/VizieR friendly name: collapse whitespace, strip accents."""
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # UI registry
    # ------------------------------------------------------------------
    def get_available_surveys(self) -> Dict[str, Dict[str, Any]]:
        """Supported archives + surveys/missions, for UI dropdowns."""
        registry: Dict[str, Dict[str, Any]] = {}
        for key, info in ARCHIVE_REGISTRY.items():
            registry[key] = {
                "label": info["label"],
                "description": info["description"],
                "credits": info["credits"],
                "service": info["service"],
            }
            # Provide a flat "choices" list for simple dropdown population.
            if "surveys" in info:
                registry[key]["choices"] = sorted(info["surveys"])
            elif "missions" in info:
                registry[key]["choices"] = sorted(info["missions"])
            elif "tables" in info:
                registry[key]["choices"] = sorted(info["tables"])
            else:
                registry[key]["choices"] = []
        return registry

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def search_and_fetch(
        self,
        target_name: str,
        archive_name: str,
        survey_or_band: str,
        width_deg: float = 1.0,
        save_as: Optional[str] = None,
    ) -> Optional[CachedDataset]:
        """Search an archive for *target_name* and cache a local data product.

        Parameters
        ----------
        target_name:
            Free-form object name (``"Crab Nebula"``), designation
            (``"M1"``, ``"Sgr A*"``) or ``"ra dec"`` coordinates.
        archive_name:
            One of ``"skyview"``, ``"mast"``, ``"vizier"``, ``"simbad"``,
            ``"heasarc"``, ``"wwt"`` (case-insensitive).
        survey_or_band:
            Survey / mission / table key appropriate for the archive,
            e.g. ``"DSS"``, ``"HST"``, ``"chanmaster"``.
        width_deg:
            Angular cutout size (SkyView only).
        save_as:
            Optional override filename (extension auto-added).

        Returns
        -------
        CachedDataset or None
            Metadata with the local file path (``entry.path``) ready for
            FITS loaders / PyTorch pipelines, or ``None`` on failure.
        """
        archive = (archive_name or "").strip().lower()
        if archive not in ARCHIVE_REGISTRY:
            logger.error("Unknown archive %r (known: %s)",
                         archive_name, ", ".join(ARCHIVE_REGISTRY))
            return None

        key = self._index.key_for(target_name, archive, survey_or_band)
        cached = self._index.get(key)
        if cached:
            logger.info("Cache hit for %s/%s/%s", target_name, archive, survey_or_band)
            return cached

        route = self.classify_object(target_name)
        try:
            if archive == "skyview":
                entry = self._fetch_skyview(target_name, survey_or_band, width_deg, route, save_as)
            elif archive == "mast":
                entry = self._fetch_mast(target_name, survey_or_band, route, save_as)
            elif archive == "vizier":
                entry = self._fetch_vizier(target_name, survey_or_band, route, save_as)
            elif archive == "simbad":
                entry = self._fetch_simbad(target_name, route, save_as)
            elif archive == "heasarc":
                entry = self._fetch_heasarc(target_name, survey_or_band, route, save_as)
            elif archive == "wwt":
                entry = self._fetch_wwt(target_name, survey_or_band, route, save_as)
            else:  # pragma: no cover - guarded above
                entry = None
        except Exception:
            logger.exception("Archive fetch failed: %s / %s", archive, target_name)
            entry = None

        if entry is not None:
            self._index.put(entry)
        return entry

    # ------------------------------------------------------------------
    # Per-archive fetchers (each returns CachedDataset | None)
    # ------------------------------------------------------------------
    def _fetch_skyview(self, target: str, survey: str, width_deg: float,
                       route: PipelineRoute, save_as: Optional[str]) -> Optional[CachedDataset]:
        if not self.skyview:
            logger.warning("astroquery.skyview unavailable - cannot fetch SkyView cutout")
            return None
        survey = survey or "DSS"
        coords = self.resolve_coordinates(target)
        if coords is None:
            logger.error("SkyView: unresolved target %r", target)
            return None
        reg = ARCHIVE_REGISTRY["skyview"]["surveys"].get(survey)
        if reg is None:
            logger.error("SkyView: unknown survey %r (choices: %s)",
                         survey, ", ".join(ARCHIVE_REGISTRY["skyview"]["surveys"]))
            return None

        logger.info("SkyView: fetching %s cutout of %r ...", survey, target)
        with self._net_lock:
            hdus = self.skyview.get_images(
                position=coords, survey=[survey], pixels=int(width_deg * 300))
        if not hdus:
            logger.warning("SkyView returned no data for %r / %s", target, survey)
            return None

        out_dir = self._archive_dir("skyview", survey)
        out_path = out_dir / (save_as or f"{self._slug(target)}_{self._slug(survey)}.fits")
        if not out_path.exists():
            from astropy.io import fits
            fits.PrimaryHDU(hdus[0][0].data, header=hdus[0][0].header).writeto(
                out_path, overwrite=True)
        return self._register(out_path, target, "skyview", survey, route,
                              ARCHIVE_REGISTRY["skyview"]["credits"])

    def _fetch_mast(self, target: str, mission: str, route: PipelineRoute,
                    save_as: Optional[str]) -> Optional[CachedDataset]:
        if not self.mast:
            logger.warning("astroquery.mast unavailable - cannot query MAST")
            return None
        obs_mod = self.mast
        mission = (mission or "HST").upper()
        coords = self.resolve_coordinates(target)
        if coords is None:
            logger.error("MAST: unresolved target %r", target)
            return None

        logger.info("MAST: querying %s observations of %r ...", mission, target)
        with self._net_lock:
            table = obs_mod.query_region(coords, radius=0.2 * u.deg,
                                         obs_collection=mission)
        if table is None or len(table) == 0:
            logger.warning("MAST: no %s observations found for %r", mission, target)
            return None

        # Pick the smallest data products of the first few results so we
        # download preview/quicklook imagery rather than full proposers' sets.
        best = None
        with self._net_lock:
            for row in table[:20]:
                products = obs_mod.get_product_list(row)
                filtered = obs_mod.filter_products(
                    products, productType="SCIENCE", extension="fits")
                if filtered is not None and len(filtered) > 0:
                    filtered.sort("size")
                    best = filtered[0]
                    break
        if best is None:
            logger.warning("MAST: no FITS products for %r", target)
            return None

        out_dir = self._archive_dir("mast", mission)
        local = out_dir / (save_as or f"{self._slug(target)}_{mission}.fits")
        if not local.exists():
            with self._net_lock:
                manifest = obs_mod.download_file(best["productFilename"], local_path=str(local))
            logger.info("MAST download -> %s (%s)", local, manifest)
        entry = self._register(
            local, target, "mast", mission, route,
            ARCHIVE_REGISTRY["mast"]["credits"])
        return entry

    def _fetch_vizier(self, target: str, catalog: str, route: PipelineRoute,
                      save_as: Optional[str]) -> Optional[CachedDataset]:
        if not self.vizier:
            logger.warning("astroquery.vizier unavailable - cannot query VizieR")
            return None
        coords = self.resolve_coordinates(target)
        if coords is None:
            logger.error("VizieR: unresolved target %r", target)
            return None
        catalog = catalog or "V/154"  # HST guide star catalog default
        logger.info("VizieR: querying catalog %s around %r ...", catalog, target)
        with self._net_lock:
            result = self.vizier.query_region(coords, radius=0.25 * u.deg, catalog=catalog)
        if not result:
            logger.warning("VizieR: no records for %r in %s", target, catalog)
            return None
        table = result[0]

        out_dir = self._archive_dir("vizier", catalog)
        out_path = out_dir / (save_as or f"{self._slug(target)}_{self._slug(catalog)}")
        table.write(str(out_path), format="votable", overwrite=True)
        return self._register(out_path, target, "vizier", catalog, route,
                              ARCHIVE_REGISTRY["vizier"]["credits"])

    def _fetch_simbad(self, target: str, route: PipelineRoute,
                      save_as: Optional[str]) -> Optional[CachedDataset]:
        if not self.simbad:
            logger.warning("astroquery.simbad unavailable")
            return None
        logger.info("SIMBAD: querying identifiers for %r ...", target)
        with self._net_lock:
            result = self.simbad.query_object(self._sanitize_name(target))
        if result is None:
            logger.warning("SIMBAD: no match for %r", target)
            return None
        out_dir = self._archive_dir("simbad", "objects")
        out_path = out_dir / (save_as or f"{self._slug(target)}_simbad.vot")
        result.write(str(out_path), format="votable", overwrite=True)
        return self._register(out_path, target, "simbad", "objects", route,
                              ARCHIVE_REGISTRY["simbad"]["credits"])

    def _fetch_heasarc(self, target: str, table: str, route: PipelineRoute,
                       save_as: Optional[str]) -> Optional[CachedDataset]:
        if not self.heasarc:
            logger.warning("astroquery.heasarc unavailable - cannot query HEASARC")
            return None
        table = table or "chanmaster"
        coords = self.resolve_coordinates(target)
        if coords is None:
            logger.error("HEASARC: unresolved target %r", target)
            return None
        logger.info("HEASARC: querying %s around %r ...", table, target)
        with self._net_lock:
            result = self.heasarc.query_region(
                coords, radius=0.5 * u.deg, mission=table, resultmax=100)
        if result is None or len(result) == 0:
            logger.warning("HEASARC: no %s records for %r", table, target)
            return None
        out_dir = self._archive_dir("heasarc", table)
        out_path = out_dir / (save_as or f"{self._slug(target)}_{table}.vot")
        result.write(str(out_path), format="votable", overwrite=True)
        return self._register(out_path, target, "heasarc", table, route,
                              ARCHIVE_REGISTRY["heasarc"]["credits"])

    def _fetch_wwt(self, target: str, survey: str, route: PipelineRoute,
                   save_as: Optional[str]) -> Optional[CachedDataset]:
        """WorldWide Telescope: hand the user a WWT-streamable survey tile URL.

        pywwt is a front-end library (needs a running web view), so we
        materialize a small WTML-style JSON manifest describing how to load
        the requested survey layer - enough for the GUI to open a WWT view
        centered on the target.
        """
        coords = self.resolve_coordinates(target)
        if coords is None:
            logger.error("WWT: unresolved target %r", target)
            return None
        survey = survey or "DSS"
        manifest = {
            "format": "astrorecon-wwt-manifest/1.0",
            "survey": survey,
            "target": target,
            "center": {"ra_deg": float(coords.ra.deg), "dec_deg": float(coords.dec.deg)},
            "credits": ARCHIVE_REGISTRY["wwt"]["credits"],
            "tutorials": "https://docs.worldwidetelescope.org",
        }
        out_dir = self._archive_dir("wwt", survey)
        out_path = out_dir / (save_as or f"{self._slug(target)}_{survey}.wwt.json")
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("WWT manifest -> %s", out_path)
        return self._register(out_path, target, "wwt", survey, route,
                              ARCHIVE_REGISTRY["wwt"]["credits"])

    # ------------------------------------------------------------------
    # Cache introspection (Sources & Credits box)
    # ------------------------------------------------------------------
    def list_cached_datasets(self, object_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """Metadata of all locally cached archive products.

        Pass ``object_class`` (e.g. ``"black_hole"``) to filter for one
        rendering pipeline.  Each item includes a ``credits`` field for the
        UI "Sources & Credits" box.
        """
        entries = self._index.all()
        if object_class:
            entries = [e for e in entries if e.object_class == object_class]
        return sorted((e.to_dict() for e in entries),
                      key=lambda e: e["cached_at"], reverse=True)

    def total_cache_size_mb(self) -> float:
        return sum(e.file_size_mb for e in self._index.all())

    def clear_cache(self, confirm: bool = False) -> int:
        """Delete every cached file (requires ``confirm=True``)."""
        if not confirm:
            logger.info("clear_cache() called without confirm - no-op")
            return 0
        removed = 0
        for entry in self._index.all():
            try:
                Path(entry.path).unlink()
                removed += 1
            except OSError as exc:
                logger.warning("could not delete %s: %s", entry.path, exc)
        self._index.entries.clear()
        self._index.save()
        logger.info("cache cleared: %d files removed", removed)
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _archive_dir(self, archive: str, sub: str) -> Path:
        """Structured cache dir: data/library_cache/<archive>/<survey>."""
        safe = re.sub(r"[^\w\-. ]", "_", sub or "misc").strip() or "misc"
        d = self.cache_root / archive / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _slug(self, text: str) -> str:
        slug = re.sub(r"\s+", "_", (text or "target").strip()).lower()
        return re.sub(r"[^\w\-]", "", slug) or "target"

    def _persist(self, tmp_path: Path, out_dir: Path, target: str, archive: str,
                 survey: str, route: PipelineRoute, save_as: Optional[str],
                 credits: str) -> Optional[CachedDataset]:
        """Move a freshly downloaded temp file into the cache & register it."""
        target_name = save_as or f"{self._slug(target)}_{self._slug(survey)}.fits"
        final = out_dir / target_name
        try:
            if tmp_path != final:
                tmp_path.replace(final)
        except OSError:
            # Cross-device or locked file - fall back to copy
            import shutil
            shutil.copy2(tmp_path, final)
        return self._register(final, target, archive, survey, route, credits)

    def _register(self, path: Path, target: str, archive: str, survey: str,
                  route: PipelineRoute, credits: str) -> Optional[CachedDataset]:
        path = Path(path)
        if not path.exists():
            logger.error("expected file %s missing after fetch", path)
            return None
        size_mb = path.stat().st_size / (1024 * 1024)
        entry = CachedDataset(
            path=str(path),
            target=target,
            archive=archive,
            survey=survey,
            object_class=route.object_class,
            pipeline=route.pipeline,
            file_size_mb=round(size_mb, 3),
            cached_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            credits=credits,
        )
        logger.info("cached: %s (%.2f MB, %s/%s)", path, size_mb, archive, survey)
        return entry


# --------------------------------------------------------------------------
# Module-level convenience singleton
# --------------------------------------------------------------------------
_manager: Optional[AstroLibraryManager] = None
_manager_lock = threading.Lock()


def get_library_manager(cache_root: Optional[Path] = None) -> AstroLibraryManager:
    """Return the shared :class:`AstroLibraryManager` (creating it once)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = AstroLibraryManager(cache_root=cache_root)
        return _manager


def search_and_fetch(target_name: str, archive_name: str,
                     survey_or_band: str) -> Optional[CachedDataset]:
    """One-call convenience wrapper matching the requested API signature."""
    return get_library_manager().search_and_fetch(target_name, archive_name, survey_or_band)


if __name__ == "__main__":  # pragma: no cover - smoke test
    lib = AstroLibraryManager()
    print(json.dumps(lib.get_available_surveys(), indent=2)[:800])
    print("Sgr A* route:", lib.get_dispatch_route("Sagittarius A*"))
    print("Crab Nebula route:", lib.get_dispatch_route("Crab Nebula"))
    print("cached:", lib.list_cached_datasets())
