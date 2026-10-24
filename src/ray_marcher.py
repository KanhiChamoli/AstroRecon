import torch


def continuous_ray_march(voxel_grid, step_size=1.0):
	"""Integrate a ``(depth, height, width)`` density grid along its depth axis."""
	if voxel_grid.ndim != 3:
		raise ValueError(f"voxel_grid must be 3D, got {voxel_grid.ndim} dimensions")
	if step_size <= 0:
		raise ValueError("step_size must be positive")
	return torch.sum(voxel_grid, dim=0) * step_size
