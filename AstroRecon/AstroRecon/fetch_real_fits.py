from pathlib import Path
from astroquery.skyview import SkyView

def download_sgr_a_fits():
    # 1. Ensure data/raw directory exists
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "sgr_a_real.fits"

    print("Querying NASA SkyView for real Sagittarius A* data (VGPS survey)...")
    try:
        # VGPS (VLA Galactic Plane Survey) provides high-resolution radio observations of the galactic center
        images = SkyView.get_images(position='Sagittarius A*', survey=['VGPS'])
        
        if images and len(images) > 0:
            images[0].writeto(output_path, overwrite=True)
            print(f"SUCCESS: Real FITS file successfully saved to: {output_path.resolve()}")
        else:
            print("WARNING: Query completed, but no image data was returned.")
    except Exception as e:
        print(f"ERROR during download: {e}")

if __name__ == "__main__":
    download_sgr_a_fits()