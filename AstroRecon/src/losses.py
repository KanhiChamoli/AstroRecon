import torch.nn.functional as F


# ==========================================================================
# Step 5 — MHD divergence-free constraint, Stokes Q/U, composite loss
# ==========================================================================
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def total_variation_3d(volume):
	"""Anisotropic 3D total variation, encouraging locally smooth density."""
	if volume.ndim != 3:
		raise ValueError(f"volume must be 3D, got {volume.ndim} dimensions")
	differences = (
		(volume[1:, :, :] - volume[:-1, :, :]).abs().mean(),
		(volume[:, 1:, :] - volume[:, :-1, :]).abs().mean(),
		(volume[:, :, 1:] - volume[:, :, :-1]).abs().mean(),
	)
	return sum(differences) / len(differences)


def total_variation_2d(image):
	"""Anisotropic total variation for an image-plane reconstruction."""
	if image.ndim != 2:
		raise ValueError(f"image must be 2D, got {image.ndim} dimensions")
	vertical = (image[1:, :] - image[:-1, :]).abs().mean()
	horizontal = (image[:, 1:] - image[:, :-1]).abs().mean()
	return vertical + horizontal


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


# --------------------------------------------------------------------------
# 3. MHD divergence-free constraint  (Gauss's law: ∇·B = 0)
# --------------------------------------------------------------------------
def divergence_B(
	B: torch.Tensor,
	dx: float = 1.0,
	dy: float = 1.0,
	dz: float = 1.0,
) -> torch.Tensor:
	"""Discrete divergence ∇·B of a 3D magnetic field via central differences.

	Implements the discrete differential operator

		(∇·B)_ijk = (B_x[i+1] − B_x[i−1])/(2Δx)
				  + (B_y[j+1] − B_y[j−1])/(2Δy)
				  + (B_z[k+1] − B_z[k−1])/(2Δz)

	using separable depthwise 3D convolutions with the (−½, 0, +½) stencil
	along each component's own axis and unit smoothing on the other two.

	Parameters
	----------
	B : (…, 3, D, H, W) tensor — magnetic field components (B_x, B_y, B_z).
		A leading batch dim is optional.
	dx, dy, dz : float — voxel spacings in the same unit as B.

	Returns
	-------
	(..., D', H', W') tensor — the scalar divergence field (interior
	voxels; borders are zero-padded, i.e. B → 0 far from the hole).
	Fully differentiable: autograd flows back into B.
	"""
	if B.dim() == 4:
		B = B.unsqueeze(0)
	if B.dim() != 5 or B.shape[1] != 3:
		raise ValueError(f"B must be (…, 3, D, H, W); got {tuple(B.shape)}")

	device, dtype = B.device, B.dtype

	# --- separable central-difference kernels -------------------------------
	# torch conv3d computes cross-correlation: out[i] = sum_j k[j]*in[i+j].
	# For the central difference d/dx = (f[i+1] - f[i-1])/(2*delta) the
	# kernel must be stored as (-1/2, 0, +1/2) - no extra flip needed.
	# conv3d dim order = (depth=D(z), height=H(y), width=W(x)); kernels are
	# separable: the (-1/2, 0, +1/2) tap sits on the derivative axis, and
	# [1,1,1] unit taps sit on the other two axes.
	def _axis_kernel(spacing: float, axis: int) -> torch.Tensor:
		"""(1,1,3,3,3) kernel; axis: 0=z(D), 1=y(H), 2=x(W).

		The derivative tap runs on `axis`; the other two axes carry
		NORMALIZED unit taps (ones/3, summing to 1) so the conv output is
		exactly the partial derivative - unnormalized ones would scale the
		result by 3 per smoothing axis (9x total in 3D).
		"""
		tap = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype) / spacing
		smooth = torch.ones(3, device=device, dtype=dtype) / 3.0
		axes = [smooth, smooth, smooth]
		axes[axis] = tap
		return (axes[0].view(1, 1, 3, 1, 1)
				 * axes[1].view(1, 1, 1, 3, 1)
				 * axes[2].view(1, 1, 1, 1, 3))

	kx = _axis_kernel(dx, 2)   # ∂/∂x  (width dim)

	kz = _axis_kernel(dz, 0)   # ∂/∂z  (depth dim)
	ky = _axis_kernel(dy, 1)   # ∂/∂y  (height dim)
	kx = _axis_kernel(dx, 2)   # ∂/∂x  (width dim)

	# depthwise conv (groups=1 per component); each derivative applied to
	# its own component of B.
	dBx = F.conv3d(B[:, 0:1], kx, padding=0)   # ∂B_x/∂x
	dBy = F.conv3d(B[:, 1:2], ky, padding=0)   # ∂B_y/∂y
	dBz = F.conv3d(B[:, 2:3], kz, padding=0)

	# All three outputs share the same (D-2, H-2, W-2) interior shape here
	# (3-tap kernels shrink every axis by 2), so a plain sum suffices.
	return dBx + dBy + dBz


def mhd_divergence_loss(
	B: torch.Tensor,
	dx: float = 1.0,
	dy: float = 1.0,
	dz: float = 1.0,
) -> torch.Tensor:
	"""Scalar MHD penalty  L_MHD = mean((∇·B)²)  — Gauss's law for magnetism.

	Zero for a perfectly solenoidal field; the mean (rather than the sum)
	keeps the penalty resolution-independent.
	"""
	div = divergence_B(B, dx, dy, dz)
	return (div ** 2).mean()


# --------------------------------------------------------------------------
# 2. Stokes Q & U linear polarization (synchrotron EVPA mapping)
# --------------------------------------------------------------------------
def stokes_qu_from_vectors(
	intensity: torch.Tensor,
	pol_angle: torch.Tensor,
	pol_fraction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Synchrotron Stokes Q/U from per-ray polarization data.

		Q = m · I · cos(2χ)
		U = m · I · sin(2χ)

	with χ the electric-vector position angle (EVPA, radians) — the 2χ
	doubling is why EVPAs wrap every 180° and Stokes space is a circle.
	"""
	two_chi = 2.0 * pol_angle
	q = pol_fraction * intensity * torch.cos(two_chi)
	u = pol_fraction * intensity * torch.sin(two_chi)
	return q, u


def evpa_from_qu(Q: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
	"""EVPA χ = ½ atan2(U, Q) (radians), the inverse Stokes mapping."""
	return 0.5 * torch.atan2(U, Q)


def polarized_intensity(Q: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
	"""Linearly polarized intensity P = √(Q² + U²)."""
	return torch.sqrt(Q * Q + U * U + 1e-24)


# --------------------------------------------------------------------------
# 1. Composite loss  L_total = L_reproj + λ_TV·L_TV + λ_MHD·L_MHD
# --------------------------------------------------------------------------
@dataclass
class CompositeLoss:
	"""Modular composite objective for GRRT inverse rendering.

	Weights default to weak regularization so data dominates; raise λ_MHD
	if monopoles appear, λ_TV if the volume turns grainy.
	"""

	lambda_tv: float = 1e-3
	lambda_mhd: float = 1e-2
	lambda_pol: float = 1.0
	voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
	normalize_reproj: bool = True

	def __call__(
		self,
		rendered: Dict[str, torch.Tensor],
		target: torch.Tensor,
		voxel: Optional[torch.Tensor] = None,
		magnetic_field: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""Evaluate the composite loss.

		parameters: rendered dict (``image``, optional ``Q``/``U``);
		target as (H, W) intensity or (3, H, W) Stokes cube; ``voxel``
		enables TV; ``magnetic_field`` enables the MHD term.

		Returns (loss, parts) — parts is a plain-float breakdown for logs.
		"""
		parts: Dict[str, float] = {}

		# --- split target into Stokes components (if provided) -------------
		# A full Stokes cube carries a Stokes axis of length 3.  Accept both
		# (3, H, W) and (B, 3, H, W) layouts; detected by shape, with the
		# spatial dims required to match the rendered image.
		img = rendered["image"]
		H, W = img.shape[-2:]
		if target.shape[-2:] == (H, W) and target.numel() == 3 * H * W:
			tI, tQ, tU = target[0], target[1], target[2]
		elif target.dim() == 4 and target.shape[1] == 3 \
				and target.shape[-2:] == (H, W):
			tI, tQ, tU = target[:, 0], target[:, 1], target[:, 2]
		else:
			tI, tQ, tU = target, None, None

		if self.normalize_reproj:
			with torch.no_grad():
				peak = float(tI.abs().max())
			if peak > 0:
				tI = tI / peak
				tQ = tQ / peak if tQ is not None else None
				tU = tU / peak if tU is not None else None

		# --- reprojection (Stokes I, plus Q/U when both sides have them) ---
		l_reproj = F.mse_loss(rendered["image"], tI)
		count = 1
		if tQ is not None and rendered.get("Q") is not None:
			l_reproj = l_reproj + F.mse_loss(rendered["Q"], tQ)
			count += 1
		if tU is not None and rendered.get("U") is not None:
			l_reproj = l_reproj + F.mse_loss(rendered["U"], tU)
			count += 1
		l_reproj = l_reproj / count
		parts["reproj"] = float(l_reproj.detach())
		total = l_reproj

		# --- polarization EVPA consistency (rendered side only, soft) ------
		if rendered.get("Q") is not None and rendered.get("U") is not None \
				and tQ is not None and tU is not None:
			# match linear polarization magnitude (EVPA angles compare mod π;
			# a full EVPA loss needs per-pixel reference frames, so the
			# default composite keeps the Q/U MSE above and adds P matching)
			p_pred = polarized_intensity(rendered["Q"], rendered["U"])
			p_true = polarized_intensity(tQ, tU)
			l_pol = F.mse_loss(p_pred, p_true)
			total = total + self.lambda_pol * l_pol
			parts["pol"] = float(l_pol.detach())

		# --- TV smoothness --------------------------------------------------
		if voxel is not None:
			l_tv = total_variation_3d(voxel)
			total = total + self.lambda_tv * l_tv
			parts["tv"] = float(l_tv.detach())

		# --- MHD divergence -------------------------------------------------
		if magnetic_field is not None:
			dx, dy, dz = self.voxel_spacing
			l_mhd = mhd_divergence_loss(magnetic_field, dx, dy, dz)
			total = total + self.lambda_mhd * l_mhd
			parts["mhd"] = float(l_mhd.detach())

		parts["total"] = float(total.detach())
		return total, parts
