import torch.nn.functional as F


def total_variation_3d(volume):
	"""Anisotropic 3D total variation, encouraging locally smooth density."""
	if volume.ndim != 3:
		raise ValueError(f"volume must be 3D, got {volume.ndim} dimensions")
	differences = [
		(volume[1:, :, :] - volume[:-1, :, :]).abs().mean(),
		(volume[:, 1:, :] - volume[:, :-1, :]).abs().mean(),
		(volume[:, :, 1:] - volume[:, :, :-1]).abs().mean(),
	]
	return sum(differences) / len(differences)


def compute_total_loss(
	rendered_img,
	observed_fits,
	voxel_grid,
	tv_weight=1e-4,
	density_weight=1e-5,
):
	"""Combine reprojection MSE, 3D smoothness, and a weak density penalty."""
	if rendered_img.shape != observed_fits.shape:
		raise ValueError(
			"rendered_img and observed_fits must have the same shape; "
			f"got {tuple(rendered_img.shape)} and {tuple(observed_fits.shape)}"
		)
	if voxel_grid.ndim != 3:
		raise ValueError(f"voxel_grid must be 3D, got {voxel_grid.ndim} dimensions")
	if tv_weight < 0 or density_weight < 0:
		raise ValueError("Regularization weights must be nonnegative")

	reprojection = F.mse_loss(rendered_img, observed_fits)
	smoothness = total_variation_3d(voxel_grid)
	density_penalty = voxel_grid.abs().mean()
	return reprojection + tv_weight * smoothness + density_weight * density_penalty
