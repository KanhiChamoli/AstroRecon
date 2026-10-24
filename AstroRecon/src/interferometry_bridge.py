"""Differentiable interferometry bridge connecting UVFITS data and MPoL to PyTorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableInterferometer(nn.Module):
    """Bridges radio interferometric visibilities (u, v) directly into differentiable PyTorch loss."""

    def __init__(self, u_lambdas, v_lambdas, vis_complex, sigma_vis, fov_rad=None, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.u = torch.as_tensor(u_lambdas, dtype=torch.float32, device=self.device)
        self.v = torch.as_tensor(v_lambdas, dtype=torch.float32, device=self.device)
        
        # Real and imaginary components of observed visibilities
        vis_complex = np.asarray(vis_complex)
        self.vis_real = torch.as_tensor(np.real(vis_complex), dtype=torch.float32, device=self.device)
        self.vis_imag = torch.as_tensor(np.imag(vis_complex), dtype=torch.float32, device=self.device)
        self.sigma = torch.as_tensor(sigma_vis, dtype=torch.float32, device=self.device)
        self.fov_rad = fov_rad or (150.0 * 1e-6 / 206265.0)  # default: 150 uas in radians

    @classmethod
    def from_ehtim_obs(cls, obs, max_samples=4000, device="cpu"):
        """Extract baseline visibilities directly from an ehtim Obsdata structure."""
        data = obs.data
        if len(data) > max_samples:
            # Subsample for efficient GPU/CPU gradient iterations
            indices = np.random.choice(len(data), size=max_samples, replace=False)
            data = data[indices]

        u = data["u"]  # In spatial frequencies (lambda)
        v = data["v"]
        vis = data["vis"]
        sigma = data["sigma"]
        fov_rad = obs.fov() if hasattr(obs, "fov") else None
        return cls(u, v, vis, sigma, fov_rad=fov_rad, device=device)

    def forward_dft(self, image_plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable 2D Discrete Fourier Transform evaluated at ungridded (u, v) coordinates.
        
        Evaluates:
            V(u, v) = sum_{x, y} I(x, y) * exp(-2*pi*i * (u*x + v*y))
        """
        if image_plane.ndim != 2:
            raise ValueError(f"Expected 2D image plane, got shape {image_plane.shape}")
        
        height, width = image_plane.shape
        # Angular grid coordinates normalized to radian field of view
        x_coords = torch.linspace(-0.5 * self.fov_rad, 0.5 * self.fov_rad, width, device=self.device)
        y_coords = torch.linspace(-0.5 * self.fov_rad, 0.5 * self.fov_rad, height, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        flat_i = image_plane.reshape(-1)
        flat_x = grid_x.reshape(-1)
        flat_y = grid_y.reshape(-1)

        # Vectorized phase matrix: 2 * pi * (u * x + v * y)
        # Shape: (N_visibilities, N_pixels)
        two_pi = 2.0 * np.pi
        # Chunk computation if visibilities x pixels is large to conserve memory
        phase = two_pi * (self.u.unsqueeze(1) * flat_x.unsqueeze(0) + self.v.unsqueeze(1) * flat_y.unsqueeze(0))
        
        pred_real = torch.matmul(torch.cos(phase), flat_i)
        pred_imag = -torch.matmul(torch.sin(phase), flat_i)
        return pred_real, pred_imag

    def visibility_loss(self, image_plane: torch.Tensor, loss_mode: str = "amplitude") -> torch.Tensor:
        """Compute differentiable chi-squared loss between predicted and observed radio visibilities."""
        pred_real, pred_imag = self.forward_dft(image_plane)

        if loss_mode == "complex":
            # Direct complex visibility chi-squared
            residual_sq = (pred_real - self.vis_real) ** 2 + (pred_imag - self.vis_imag) ** 2
            loss = torch.mean(residual_sq / (self.sigma ** 2 + 1e-8))
        elif loss_mode == "amplitude":
            # Visibility amplitude loss (robust to atmospheric antenna phase jitter)
            obs_amp = torch.sqrt(self.vis_real ** 2 + self.vis_imag ** 2 + 1e-12)
            pred_amp = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + 1e-12)
            loss = torch.mean(((pred_amp - obs_amp) / (self.sigma + 1e-8)) ** 2)
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        return loss

