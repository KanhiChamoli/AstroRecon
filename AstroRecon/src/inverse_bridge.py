"""Modular Inverse Rendering Bridge connecting volumetric representations to observation losses."""

from typing import Callable, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import total_variation_3d, total_variation_2d
from src.ray_marcher import render_volume


class VolumetricInverseRenderer(nn.Module):
    """Modular inverse rendering engine for 3D density optimization from 2D observations."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (64, 64, 64),
        ray_marcher_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            grid_shape: (Depth, Height, Width) resolution of the 3D voxel field.
            ray_marcher_fn: Differentiable function taking (D, H, W) and returning (H, W).
            device: 'cpu' or 'cuda'.
        """
        super().__init__()
        self.device = torch.device(device)
        self.depth, self.height, self.width = grid_shape

        # Learnable voxel parameter representation (initialized positive with softplus)
        init_tensor = torch.randn(grid_shape, device=self.device) * 0.01 + 0.1
        self.raw_density = nn.Parameter(init_tensor)

        # Pluggable ray marcher; default is the true 3D volumetric renderer
        # (trilinear grid_sample ray integration, align_corners=True). The
        # former depth-axis ``torch.sum(grid, dim=0)`` extrusion is banned.
        if ray_marcher_fn is not None:
            self.ray_marcher = ray_marcher_fn
        else:
            self.ray_marcher = lambda grid: render_volume(
                grid, (self.height, self.width)
            )

    @property
    def density_field(self) -> torch.Tensor:
        """Physical nonnegative 3D density field enforced by Softplus."""
        return F.softplus(self.raw_density)

    def forward(self) -> torch.Tensor:
        """Forward projection of 3D volume into 2D image plane."""
        return self.ray_marcher(self.density_field)

    def train_step(
        self,
        observed_target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        tv_weight: float = 1e-3,
        density_l1_weight: float = 1e-4,
        interferometer_loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Perform one differentiable optimization step and return loss telemetry."""
        optimizer.zero_grad()
        predicted_projection = self.forward()

        # Normalize predicted projection to [0, 1] range for photometric alignment
        peak = torch.max(predicted_projection)
        norm_projection = predicted_projection / (peak + 1e-8) if peak > 0 else predicted_projection

        # 1. Primary Data Fit Loss
        if interferometer_loss_fn is not None:
            data_loss = interferometer_loss_fn(norm_projection)
        else:
            data_loss = F.mse_loss(norm_projection, observed_target.to(self.device))

        # 2. Physics Regularization Priors
        tv_loss = total_variation_3d(self.density_field)
        l1_loss = torch.mean(self.density_field)

        total_loss = data_loss + tv_weight * tv_loss + density_l1_weight * l1_loss
        total_loss.backward()
        optimizer.step()

        metrics = {
            "total_loss": float(total_loss.item()),
            "data_loss": float(data_loss.item()),
            "tv_loss": float(tv_loss.item()),
            "l1_density": float(l1_loss.item()),
        }
        return float(total_loss.item()), metrics

