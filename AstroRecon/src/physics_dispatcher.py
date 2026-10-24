"""Physics dispatcher — classify spatial objects and route to pipelines.

Given a search query / library entry / uploaded data filename, decide which
rendering pipeline to use:

* Track B — **GRRT (General-Relativistic Ray Tracing)** for black holes:
  null geodesics through a Kerr/Schwarzschild metric, Doppler beaming,
  gravitational redshift, photon ring.
* Track A — **Optically thin radiative transfer** for nebulae, star systems,
  clusters: emission-line nebula physics, volumetric ray marching.

Pure-python classification so it can be unit tested without a GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineRoute:
    """Result of the dispatcher: which physics pipeline to run, and why."""
    pipeline: str            # "grrt" | "optically_thin" | "orbit"
    track: str               # human-readable track label
    reason: str              # matched keywords / heuristics
    object_class: str        # "black_hole" | "nebula" | "star_system" | "unknown"
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def is_grrt(self) -> bool:
        return self.pipeline == "grrt"


# ---------------------------------------------------------------- keyword maps
_BLACK_HOLE_KEYWORDS = [
    "black hole", "sgr a*", "sagittarius a*", "m87", "virgo a", "cygnus x-1",
    "cygnus x1", "cyg x-1", "schwarzschild", "kerr", "event horizon",
    "accretion", "photon ring", "eht", "x-ray binary", "relativistic",
]
_NEBULA_KEYWORDS = [
    "nebula", "orion", "m42", "crab", "m1", "h ii", "hii", "hα", "h-alpha",
    "emission", "supernova remnant", "snr", "planetary nebula", "ionized gas",
]
_STAR_SYSTEM_KEYWORDS = [
    "star system", "binary", "orbit", "star cluster", "globular", "m13",
    "open cluster", "stellar", "kepler", "two-body", "pulsar",
]
_PIPELINE_HINTS = {
    "grrt": _BLACK_HOLE_KEYWORDS,
    "optically_thin": _NEBULA_KEYWORDS,
    "orbit": _STAR_SYSTEM_KEYWORDS,
}


def classify_query(query: str) -> PipelineRoute:
    """Classify a free-text search query into a physics pipeline route."""
    q = (query or "").lower()

    scores = {
        pipe: sum(1 for kw in kws if kw in q)
        for pipe, kws in _PIPELINE_HINTS.items()
    }
    best = max(scores, key=scores.get) if scores else "optically_thin"

    if scores[best] == 0:
        # Heuristic fallback: filename / generic words.
        if any(t in q for t in ("hole", "horizon", "sgr")):
            best = "grrt"
        elif any(t in q for t in ("neb", "gas", "dust")):
            best = "optically_thin"
        else:
            best = "orbit"

    object_class = {
        "grrt": "black_hole",
        "optically_thin": "nebula",
        "orbit": "star_system",
    }[best]

    params: Dict[str, float] = {}
    if best == "grrt":
        params = {"spin_a": 0.9, "inclination_deg": 17.0, "m87_mode": 1.0 if "m87" in q else 0.0}
    elif best == "optically_thin":
        params = {"emissivity_j": 1.0, "extinction": 0.0}
    else:
        params = {"gm": 4.3e-3, "dt": 0.001, "steps": 1000}

    return PipelineRoute(
        pipeline=best,
        track={
            "grrt": "Track B — GRRT null-geodesic ray tracing",
            "optically_thin": "Track A — optically thin radiative transfer",
            "orbit": "Track A — N-body / orbital integration",
        }[best],
        reason=f"keyword match (score {scores.get(best, 0)})",
        object_class=object_class,
        params=params,
    )


def classify_entry(entry: Dict) -> PipelineRoute:
    """Classify a library entry dict (name/kind/tags)."""
    blob = " ".join(str(entry.get(k, "")) for k in ("name", "kind", "detail", "tags"))
    return classify_query(blob)


def dispatch(entry: Optional[Dict] = None, query: str = "") -> PipelineRoute:
    """Public hook: route a search / selection to the correct physics pipeline."""
    if entry:
        return classify_entry(entry)
    return classify_query(query)
