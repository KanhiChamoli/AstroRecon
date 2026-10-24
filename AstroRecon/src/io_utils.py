from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def clean_and_normalize(data, noise_sigma=0.8, stretch=8.0, denoise_sigma=1.0):
	"""Remove background, gently clip noise, and apply an astronomy-friendly asinh stretch."""
	array = np.asarray(data, dtype=np.float32).squeeze()
	if array.ndim != 2:
		raise ValueError(f"Expected a 2D observation, got shape {array.shape}")
	finite = np.isfinite(array)
	if not finite.any():
		raise ValueError("Observation contains no finite pixel values")

	background = float(np.median(array[finite]))
	array = np.where(finite, array, background)
	noise_level = 1.4826 * float(np.median(np.abs(array[finite] - background)))
	signal = np.maximum(array - background - noise_sigma * noise_level, 0.0)
	valid_signal = signal[finite]
	peak = float(np.percentile(valid_signal, 99.8)) if valid_signal.size else 0.0
	if peak > 0.0:
		signal = np.clip(signal / peak, 0.0, 1.0)
		if stretch > 0:
			signal = np.arcsinh(stretch * signal) / np.arcsinh(stretch)
	if denoise_sigma > 0:
		signal = gaussian_filter(signal, sigma=denoise_sigma, mode="nearest")
	return signal.astype(np.float32, copy=False)


def load_fits(path, device="cpu", denoise_sigma=1.0, hdu_index=0):
	"""Load the primary FITS image, clean it, and return a float32 tensor."""
	fits_path = Path(path)
	if not fits_path.is_file():
		raise FileNotFoundError(f"FITS file does not exist: {fits_path}")
	image = fits.getdata(fits_path, ext=hdu_index, memmap=False)
	return torch.from_numpy(clean_and_normalize(image, denoise_sigma=denoise_sigma)).to(device=device)


def load_fits_cube(path, device="cpu", denoise_sigma=1.0, hdu_index=0):
	"""Load a 3D FITS cube in NumPy order (axis-3, y, x) and denoise it."""
	fits_path = Path(path)
	if not fits_path.is_file():
		raise FileNotFoundError(f"FITS cube does not exist: {fits_path}")
	with fits.open(fits_path, memmap=False) as hdus:
		data = hdus[hdu_index].data
		if data is None or data.ndim != 3:
			shape = None if data is None else data.shape
			raise ValueError(f"Expected a 3D primary FITS cube, got shape {shape}")
		cube = np.asarray(data, dtype=np.float32)
		finite = np.isfinite(cube)
		if not finite.any():
			raise ValueError("FITS cube contains no finite values")
		cube = np.where(finite, cube, np.nanmedian(cube[finite]))
		background = float(np.median(cube))
		noise = 1.4826 * float(np.median(np.abs(cube - background)))
		signal = np.maximum(cube - background - 0.8 * noise, 0.0)
		peak = float(np.percentile(signal, 99.8))
		if peak > 0:
			signal = np.clip(signal / peak, 0.0, 1.0)
			signal = np.arcsinh(8.0 * signal) / np.arcsinh(8.0)
		header = hdus[hdu_index].header.copy()
	if denoise_sigma > 0:
		signal = gaussian_filter(
			signal,
			sigma=(max(0.25, denoise_sigma * 0.5), denoise_sigma, denoise_sigma),
			mode="nearest",
		)
	return torch.from_numpy(signal.astype(np.float32, copy=False)).to(device=device), header


def save_fits_cube(cube, header, path):
	"""Save a processed 3D cube while preserving the original FITS WCS header."""
	values = torch.as_tensor(cube).detach().cpu().numpy().astype(np.float32)
	if values.ndim != 3:
		raise ValueError(f"Expected a 3D cube, got shape {values.shape}")
	output = Path(path)
	output.parent.mkdir(parents=True, exist_ok=True)
	fits.PrimaryHDU(data=values, header=header).writeto(output, overwrite=True)
	return output


def make_synthetic_observation(size=64, device="cpu"):
	"""Create a normalized nebula-like target for a no-data smoke run."""
	if size < 2:
		raise ValueError("size must be at least 2")
	axis = torch.linspace(-1.0, 1.0, size, device=device)
	y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
	image = (
		torch.exp(-((x_grid + 0.28) ** 2 / 0.12 + (y_grid - 0.12) ** 2 / 0.3))
		+ 0.65 * torch.exp(-((x_grid - 0.32) ** 2 / 0.08 + (y_grid + 0.25) ** 2 / 0.12))
		+ 0.25 * torch.exp(-((x_grid) ** 2 / 0.45 + (y_grid + 0.5) ** 2 / 0.06))
	)
	return image / image.max()


def save_vtk_volume(volume, path):
	"""Write a 3D tensor as an ASCII VTK STRUCTURED_POINTS volume."""
	values = torch.as_tensor(volume).detach().cpu().numpy().astype(np.float32)
	if values.ndim != 3:
		raise ValueError(f"Expected a 3D volume, got shape {values.shape}")
	depth, height, width = values.shape
	vtk_path = Path(path)
	vtk_path.parent.mkdir(parents=True, exist_ok=True)
	with vtk_path.open("w", encoding="ascii", newline="\n") as output:
		output.write("# vtk DataFile Version 3.0\nAstroRecon density volume\nASCII\n")
		output.write("DATASET STRUCTURED_POINTS\n")
		output.write(f"DIMENSIONS {width} {height} {depth}\n")
		output.write("ORIGIN 0 0 0\nSPACING 1 1 1\n")
		output.write(f"POINT_DATA {values.size}\nSCALARS density float 1\nLOOKUP_TABLE default\n")
		np.savetxt(output, values.reshape(-1), fmt="%.7g")
