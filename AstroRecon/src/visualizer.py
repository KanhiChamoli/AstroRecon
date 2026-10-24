from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.render_metadata import write_render_metadata


def save_comparison(observed, filtered, rendered, output_path, metadata=None):
	"""Save native-size input, denoised target, reconstruction, and residual."""
	arrays = [torch.as_tensor(value).detach().cpu().numpy() for value in (observed, filtered, rendered)]
	if any(array.ndim != 2 for array in arrays) or len({array.shape for array in arrays}) != 1:
		raise ValueError("observed, filtered, and rendered must be matching 2D arrays or tensors")
	observed_array, filtered_array, rendered_array = arrays
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	figure, axes = plt.subplots(1, 4, figsize=(17, 4.7), constrained_layout=True)
	panels = (
		(observed_array, "Input FITS", "magma"),
		(filtered_array, "Denoised target", "magma"),
		(rendered_array, "Reconstruction", "magma"),
		(abs(filtered_array - rendered_array), "Absolute residual", "viridis"),
	)
	for axis, (image, title, color_map) in zip(axes[:3], panels[:3]):
		axis.imshow(image, origin="lower", cmap=color_map, vmin=0.0, vmax=1.0)
		axis.set_title(title)
		axis.set_axis_off()
	residual = panels[3][0]
	residual_limit = max(float(np.percentile(residual, 99.5)), 1e-6)
	axes[3].imshow(residual, origin="lower", cmap="viridis", vmin=0.0, vmax=residual_limit)
	axes[3].set_title(f"Absolute residual (0 to {residual_limit:.3g})")
	axes[3].set_axis_off()
	figure.savefig(output, dpi=180)
	plt.close(figure)
	write_render_metadata(output, {"render_type": "comparison", **(metadata or {})})
	return output


def save_image(image, output_path, title="Filtered FITS image", metadata=None):
	"""Save a single high-resolution grayscale astronomical image."""
	array = torch.as_tensor(image).detach().cpu().numpy()
	if array.ndim != 2:
		raise ValueError(f"image must be 2D, got shape {array.shape}")
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
	axis.imshow(array, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
	axis.set_title(title)
	axis.set_axis_off()
	figure.savefig(output, dpi=200)
	plt.close(figure)
	write_render_metadata(output, {"render_type": "image", "title": title, **(metadata or {})})
	return output


def save_cube_comparison(original_cube, filtered_cube, output_path, metadata=None):
	"""Compare raw and denoised max projections and central cube slices."""
	original = torch.as_tensor(original_cube).detach().cpu().numpy()
	filtered = torch.as_tensor(filtered_cube).detach().cpu().numpy()
	if original.ndim != 3 or filtered.shape != original.shape:
		raise ValueError("original_cube and filtered_cube must be matching 3D arrays or tensors")
	middle = original.shape[0] // 2
	panels = (
		(original.max(axis=0), "Raw max-intensity projection"),
		(filtered.max(axis=0), "Denoised max-intensity projection"),
		(original[middle], f"Raw slice {middle}"),
		(filtered[middle], f"Denoised slice {middle}"),
	)
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	figure, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
	for axis, (image, title) in zip(axes, panels):
		axis.imshow(image, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
		axis.set_title(title)
		axis.set_axis_off()
	figure.savefig(output, dpi=180)
	plt.close(figure)
	write_render_metadata(output, {"render_type": "3D FITS cube denoising comparison", **(metadata or {})})
	return output
