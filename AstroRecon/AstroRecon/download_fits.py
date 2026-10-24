from pathlib import Path
import shutil
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FITS_URL = "https://data.astropy.org/example_images/galaxies.fits"


def validate_fits(path):
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            if hdu.data is not None and hdu.data.ndim in (2, 3):
                return tuple(hdu.data.shape)
    raise ValueError("The downloaded file contains no 2D image or 3D FITS cube")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / "galaxies_sample.fits"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=RAW_DIR, suffix=".fits", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with urlopen(FITS_URL, timeout=10) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        shape = validate_fits(temporary_path)
        temporary_path.replace(destination)
        print(f"Downloaded and validated FITS image with shape {shape}: {destination}")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        fallback = RAW_DIR / "M13_sample.fits"
        if not fallback.is_file():
            raise RuntimeError(
                f"Download failed ({error}) and no local M13_sample.fits fallback exists in {RAW_DIR}"
            ) from error
        shape = validate_fits(fallback)
        print(f"Remote URL unavailable ({error}); using existing M13 FITS sample, shape {shape}: {fallback}")


if __name__ == "__main__":
	main()