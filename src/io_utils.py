from pathlib import Path

import numpy as np
import torch
from astropy.io import fits


def clean_and_normalize(data, noise_sigma=3.0):
	"""Subtract a robust background, suppress low-level noise, and scale to [0, 1]."""
	array = np.asarray(data, dtype=np.float32).squeeze()
	if array.ndim != 2:
		raise ValueError(f"Expected a 2D observation, got shape {array.shape}")

	finite = np.isfinite(array)
	if not finite.any():
		raise ValueError("Observation contains no finite pixel values")

	background = float(np.median(array[finite]))
	array = np.where(finite, array, background)
	deviations = np.abs(array - background)
	noise_level = 1.4826 * float(np.median(deviations))
	signal = np.maximum(array - background - noise_sigma * noise_level, 0.0)
	peak = float(signal.max())
	if peak > 0.0:
		signal /= peak
	return signal.astype(np.float32, copy=False)


def load_fits(path, device="cpu"):
	"""Load the primary FITS image, clean it, and return a float32 tensor."""
	fits_path = Path(path)
	if not fits_path.is_file():
		raise FileNotFoundError(f"FITS file does not exist: {fits_path}")
	image = fits.getdata(fits_path, memmap=False)
	normalized = clean_and_normalize(image)
	return torch.from_numpy(normalized).to(device=device)


def make_synthetic_observation(size=64, device="cpu"):
	"""Create a normalized nebula-like 2D target for a no-data smoke run."""
	if size < 2:
		raise ValueError("size must be at least 2")

	axis = torch.linspace(-1.0, 1.0, size, device=device)
	y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
	components = (
		1.0 * torch.exp(-((x_grid + 0.28) ** 2 / 0.12 + (y_grid - 0.12) ** 2 / 0.3))
		+ 0.65 * torch.exp(-((x_grid - 0.32) ** 2 / 0.08 + (y_grid + 0.25) ** 2 / 0.12))
		+ 0.25 * torch.exp(-((x_grid) ** 2 / 0.45 + (y_grid + 0.5) ** 2 / 0.06))
	)
	return components / components.max()


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
