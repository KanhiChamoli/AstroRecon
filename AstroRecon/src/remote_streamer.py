"""Remote observation streamer: dynamically streams FITS cutouts and UVFITS data on demand."""

from io import BytesIO
from pathlib import Path
from typing import Tuple, Dict, Any
from urllib.request import urlopen
import logging
import numpy as np
import torch
from astropy.io import fits
import astropy.units as u

from src.io_utils import clean_and_normalize

logger = logging.getLogger(__name__)


class RemoteObservationStreamer:
    """Streams astronomical observations directly into memory without local disk bloat."""

    @staticmethod
    def stream_skyview_survey(
        target: str = "Sagittarius A*",
        survey: str = "NVSS",
        radius_arcmin: float = 10.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Query NASA SkyView and return a normalized 2D observation tensor directly in memory."""
        tensor, _ = RemoteObservationStreamer.stream_skyview_with_metadata(
            target=target,
            survey=survey,
            radius_arcmin=radius_arcmin,
            device=device,
        )
        return tensor

    @staticmethod
    def stream_skyview_with_metadata(
        target: str = "Sagittarius A*",
        survey: str = "NVSS",
        radius_arcmin: float = 10.0,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Query NASA SkyView and return (normalized_tensor, metadata_dict)."""
        try:
            from astroquery.skyview import SkyView
        except ImportError as exc:
            raise RuntimeError("astroquery is required for SkyView streaming. Install via pip.") from exc

        logger.info("Querying NASA SkyView for target '%s' (Survey: '%s', Radius: %s arcmin)...", target, survey, radius_arcmin)
        radius_quantity = radius_arcmin * u.arcmin
        images = SkyView.get_images(
            position=target,
            survey=[survey],
            radius=radius_quantity,
        )
        if not images:
            raise RuntimeError(f"SkyView returned no image data for {target} in survey {survey}")

        primary_data = images[0][0].data.astype(np.float32)
        shape = tuple(primary_data.shape)
        logger.info("Successfully received cutout with shape %s.", shape)
        normalized = clean_and_normalize(primary_data)
        tensor = torch.from_numpy(normalized).to(device=device)

        metadata = {
            "source_name": f"NASA SkyView — {survey} Survey",
            "target": target,
            "survey": survey,
            "dimensions": f"{shape[1]} x {shape[0]} px",
            "angular_radius": f"{radius_arcmin} arcmin",
            "source_url": "https://skyview.gsfc.nasa.gov/",
            "processing": "Robust median subtraction, 0.8σ noise floor threshold, asinh contrast stretch (β=8.0)",
            "observatory": "National Radio Astronomy Observatory (NRAO) / NASA Goddard",
        }
        tensor.metadata = metadata
        return tensor, metadata

    @staticmethod
    def stream_fits_url(url: str, device: str = "cpu", hdu_index: int = 0) -> torch.Tensor:
        """Stream a remote FITS file over HTTP directly into memory as a PyTorch tensor."""
        logger.info("Streaming remote FITS from: %s", url)
        with urlopen(url, timeout=20) as response:
            buffer = BytesIO(response.read())

        with fits.open(buffer, memmap=False) as hdus:
            data = hdus[hdu_index].data
            if data is None:
                raise ValueError(f"HDU index {hdu_index} contains no data.")
            data = np.asarray(data, dtype=np.float32)

        normalized = clean_and_normalize(data)
        tensor = torch.from_numpy(normalized).to(device=device)
        tensor.metadata = {
            "source_name": "Remote HTTP FITS Stream",
            "target": Path(url).stem,
            "source_url": url,
            "dimensions": f"{data.shape[1]} x {data.shape[0]} px",
            "processing": "Robust median subtraction, 0.8σ threshold, asinh stretch",
        }
        return tensor

    @staticmethod
    def stream_or_load_uvfits(source_path_or_bytes, polrep: str = "stokes"):
        """Load UVFITS visibility data into an ehtim Obsdata structure from path or buffer."""
        import ehtim as eh

        if isinstance(source_path_or_bytes, (str, Path)):
            path = Path(source_path_or_bytes)
            if not path.is_file():
                raise FileNotFoundError(f"UVFITS file not found: {path}")
            with fits.open(path, memmap=False) as hdus:
                obs = eh.obsdata.load_uvfits(hdus, polrep="circ")
        elif isinstance(source_path_or_bytes, (bytes, BytesIO)):
            buffer = BytesIO(source_path_or_bytes) if isinstance(source_path_or_bytes, bytes) else source_path_or_bytes
            with fits.open(buffer, memmap=False) as hdus:
                obs = eh.obsdata.load_uvfits(hdus, polrep="circ")
        else:
            raise TypeError(f"Unsupported source type for UVFITS: {type(source_path_or_bytes)}")

        if polrep.lower() == "stokes":
            obs = obs.switch_polrep("stokes", allow_singlepol=True)
        return obs
