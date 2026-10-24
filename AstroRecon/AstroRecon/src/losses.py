import torch
import torch.nn.functional as F


def compute_total_loss(rendered_img, observed_fits, voxel_grid):
    """Combine projection matching with a small density regularizer."""
    if rendered_img.shape != observed_fits.shape:
        raise ValueError(
            "rendered_img and observed_fits must have the same shape; "
            f"got {tuple(rendered_img.shape)} and {tuple(observed_fits.shape)}"
        )

    projection_loss = F.mse_loss(rendered_img, observed_fits)
    density_regularization = voxel_grid.abs().mean()
    return projection_loss + 1e-4 * density_regularization