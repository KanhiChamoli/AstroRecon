from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_comparison(observed, rendered, output_path):
	"""Save observed, rendered, and absolute-error images side by side."""
	observed_array = torch.as_tensor(observed).detach().cpu().numpy()
	rendered_array = torch.as_tensor(rendered).detach().cpu().numpy()
	if observed_array.shape != rendered_array.shape or observed_array.ndim != 2:
		raise ValueError("observed and rendered must be matching 2D arrays or tensors")

	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	error = abs(observed_array - rendered_array)
	figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
	for axis, image, title, color_map in zip(
		axes,
		(observed_array, rendered_array, error),
		("Observed", "Reconstruction", "Absolute error"),
		("magma", "magma", "viridis"),
	):
		axis.imshow(image, origin="lower", cmap=color_map)
		axis.set_title(title)
		axis.set_axis_off()
	figure.savefig(output, dpi=160)
	plt.close(figure)
	return output
