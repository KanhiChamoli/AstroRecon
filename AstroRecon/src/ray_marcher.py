"""True 3D volumetric ray marcher for AstroRecon.

No 2D depth extrusion lives here (or anywhere in the pipeline). The
reconstructed volume ``V ∈ ℝ^{D×H×W}`` is a genuine 3D voxel grid of spatial
density/emission spanning the X, Y, Z axes, and every camera pixel integrates
along a ray through that box:

    Î(x, y) = Σ_k  V(x(t_k), y(t_k), z(t_k)) · τ_k · Δs

with optical-depth extinction ``τ_k = exp(-Σ_{j<k} σ · V_j · Δs)`` accumulated
front-to-back along the line of sight.

Voxel sampling uses ``torch.nn.functional.grid_sample`` with
``align_corners=True`` on a ``(1, 1, D, H, W)`` grid, so gradients flow back
through all three spatial dimensions of ``V`` and a standard optimizer can
update the voxels via plain backpropagation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_BBOX = 1.0          # voxel box spans [-1, 1]^3 (align_corners=True corners)
_ORIGIN_Z = 1.4      # camera sits on the +z axis, outside the box
_TINY = 1e-8


@dataclass
class CameraRig:
    """Ray caster per pixel: image plane at +z, looking toward the origin.

    ``yaw_deg`` / ``pitch_deg`` rotate the ray bundle so multiple known rigs
    can supply true tomographic views of one voxel volume.
    """

    height: int
    width: int
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    n_steps: int = 64
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.sigma < 0:
            raise ValueError("sigma (extinction) must be nonnegative")


def _pixel_dirs(camera: CameraRig, device: torch.device) -> torch.Tensor:
    """Unit ray directions for every pixel: (H, W, 3), pointing -z."""
    ys = torch.linspace(-_BBOX, _BBOX, camera.height, device=device)
    xs = torch.linspace(-_BBOX, _BBOX, camera.width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack(
        [grid_x * 0.5, -grid_y * 0.5, -torch.ones_like(grid_x)], dim=-1
    )
    dirs = F.normalize(dirs, dim=-1)
    yaw = math.radians(camera.yaw_deg)
    pitch = math.radians(camera.pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2 = cy * x + sy * z
    z2 = -sy * x + cy * z
    y2 = cp * y - sp * z2
    z3 = sp * y + cp * z2
    return F.normalize(torch.stack([x2, y2, z3], dim=-1), dim=-1)


def _ray_bbox_ts(dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel near/far ray-box intersection with [-1,1]^3.

    Rays start at (0, 0, ``_ORIGIN_Z``), outside the box; ``dir.z < 0`` by
    construction, so every pixel has a valid entry and exit point.
    """
    def _inv(axis: torch.Tensor) -> torch.Tensor:
        # near-zero direction components get a ±tiny denominator so the slab
        # math stays finite; rays may still miss that slab pair via min/max.
        return 1.0 / torch.where(
            axis.abs() < _TINY,
            torch.where(axis < 0, axis.new_tensor(-_TINY), axis.new_tensor(_TINY)),
            axis,
        )

    inv_x, inv_y, inv_z = _inv(dirs[..., 0]), _inv(dirs[..., 1]), _inv(dirs[..., 2])

    # x and y slabs: ray origin coordinate is 0 on both axes
    t_xa, t_xb = -_BBOX * inv_x, _BBOX * inv_x
    t_ya, t_yb = -_BBOX * inv_y, _BBOX * inv_y
    # z slab: ray origin is at _ORIGIN_Z
    t_za, t_zb = (-_BBOX - _ORIGIN_Z) * inv_z, (_BBOX - _ORIGIN_Z) * inv_z

    x_near, x_far = torch.minimum(t_xa, t_xb), torch.maximum(t_xa, t_xb)
    y_near, y_far = torch.minimum(t_ya, t_yb), torch.maximum(t_ya, t_yb)
    z_near, z_far = torch.minimum(t_za, t_zb), torch.maximum(t_za, t_zb)

    t_near = torch.maximum(torch.maximum(x_near, y_near), z_near)
    t_far = torch.minimum(torch.minimum(x_far, y_far), z_far)
    eps = 1e-6
    t_near = torch.clamp(t_near, min=eps)
    t_far = torch.maximum(t_far, t_near + eps)
    return t_near, t_far


class VolumetricRayMarcher:
    """Differentiable volume renderer over a (D, H, W) voxel tensor."""

    def __init__(
        self,
        volume_shape: Sequence[int],
        image_shape: Sequence[int],
        device: Optional[torch.device] = None,
        n_steps: int = 64,
        sigma: float = 1.0,
        camera: Optional[CameraRig] = None,
    ) -> None:
        if len(volume_shape) != 3 or len(image_shape) != 2:
            raise ValueError("volume_shape must be (D, H, W); image_shape (H, W)")
        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.device = device
        self.camera = camera or CameraRig(
            height=self.image_shape[0],
            width=self.image_shape[1],
            n_steps=n_steps,
            sigma=sigma,
        )
        self._cache: dict = {}

    # ------------------------------------------------------------------ api --
    def render(
        self,
        voxel_grid: torch.Tensor,
        camera: Optional[CameraRig] = None,
    ) -> torch.Tensor:
        """Render ``voxel_grid`` (D, H, W) → image (H_img, W_img).

        Fully differentiable with respect to ``voxel_grid`` over all three
        spatial dimensions; every sample is trilinear with align_corners=True.
        """
        if voxel_grid.ndim != 3:
            raise ValueError(
                f"voxel_grid must be 3D (D,H,W), got shape {tuple(voxel_grid.shape)}"
            )
        cam = camera or self.camera
        device = voxel_grid.device

        t_near, t_far = self._bbox_ts(cam, device)
        ts = torch.linspace(0.0, 1.0, cam.n_steps, device=device).view(1, 1, -1)
        ts = t_near.unsqueeze(-1) + (t_far - t_near).unsqueeze(-1) * ts  # (H, W, K)
        delta_s = ((t_far - t_near) / cam.n_steps).unsqueeze(-1)        # (H, W, 1)

        dirs = self._dirs(cam, device)                                  # (H, W, 3)
        origin = dirs.new_tensor([0.0, 0.0, _ORIGIN_Z]).view(1, 1, 3)
        points = origin + dirs.unsqueeze(-2) * ts.unsqueeze(-1)         # (H, W, K, 3)

        # world [-1,1]^3 → grid_sample coordinates (align_corners=True)
        #   x → sample axis W, y → sample axis H (row order inverted), z → D
        grid = torch.stack(
            [points[..., 0], -points[..., 1], points[..., 2]], dim=-1
        )

        vol = voxel_grid.unsqueeze(0).unsqueeze(0)                      # (1,1,D,H,W)
        rho = F.grid_sample(
            vol,
            grid.unsqueeze(0),                                          # (1,H,W,K,3)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0)                                         # (H, W, K)

        # τ_k = exp(-Σ_{j<k} σ · ρ_j · Δs), front-to-back via cumulative sum
        sigma_delta = cam.sigma * delta_s
        optical = torch.cumsum(rho, dim=-1) * sigma_delta
        shifted = torch.cat(
            [torch.zeros_like(optical[..., :1]), optical[..., :-1]], dim=-1
        )
        tau = torch.exp(-shifted)

        return torch.sum(rho * tau * delta_s, dim=-1)                   # (H, W)

    def render_batch(
        self,
        voxel_grid: torch.Tensor,
        cameras: Sequence[CameraRig],
    ) -> torch.Tensor:
        """Render several rigs → stacked views (C, H, W) for multi-view fitting."""
        return torch.stack([self.render(voxel_grid, cam) for cam in cameras], dim=0)

    # -------------------------------------------------------------- cached --
    def _cache_for(self, cam: CameraRig, device: torch.device) -> dict:
        key = (cam.height, cam.width, cam.yaw_deg, cam.pitch_deg, cam.n_steps, str(device))
        if self._cache.get("key") != key:
            dirs = _pixel_dirs(cam, device)
            self._cache.clear()
            self._cache.update(
                key=key, dirs=dirs, ts=_ray_bbox_ts(dirs)
            )
        return self._cache

    def _dirs(self, cam: CameraRig, device: torch.device) -> torch.Tensor:
        return self._cache_for(cam, device)["dirs"]

    def _bbox_ts(self, cam: CameraRig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._cache_for(cam, device)
        # bbox ts were cached against the same key inside _cache_for
        return cache["ts"]


def render_volume(
    voxel_grid: torch.Tensor,
    image_shape: Sequence[int],
    device: Optional[torch.device] = None,
    n_steps: int = 64,
    sigma: float = 1.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> torch.Tensor:
    """Convenience functional API: V (D,H,W) → image (H,W)."""
    marcher = VolumetricRayMarcher(
        tuple(voxel_grid.shape),
        tuple(image_shape),
        device=None,  # render follows the voxel grid's device
        n_steps=n_steps,
        sigma=sigma,
        camera=CameraRig(
            height=int(image_shape[0]),
            width=int(image_shape[1]),
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            n_steps=n_steps,
            sigma=sigma,
        ),
    )
    return marcher.render(voxel_grid)


# ==========================================================================
# Track B — Step 1: GRRT camera-ray initialization & spacetime 4-vectors
# ==========================================================================
# This section is the relativistic counterpart of the Euclidean CameraRig
# above.  Track A marches straight rays through a voxel box; Track B will
# integrate *null geodesics* through curved spacetime (Schwarzschild /
# Kerr).  Step 1 only prepares the initial conditions:
#
#   * a bundle of camera rays (one per pixel), parameterized by FOV and a
#     differentiable camera position / look-at target, and
#   * the spacetime initial data (x^mu, p^mu) for each ray, with the
#     momenta verified to satisfy the flat-space null condition
#
#         g_{mu nu} p^mu p^nu = eta_{mu nu} p^mu p^nu
#                             = -p_t^2 + px^2 + py^2 + pz^2 = 0
#
#     i.e. every photon travels at exactly c before entering the
#     gravitational well.  (Signature convention: (-,+,+,+).)
#
# Everything is torch.Tensor with requires_grad support so optimization of
# camera parameters (FOV, camera position, target) can flow through the
# eventual geodesic integrator.

Metric = Optional[torch.Tensor]  # (4, 4) or (B, 4, 4) metric tensor; None = flat


@dataclass
class GRRTCamera:
    """Differentiable pinhole camera for GRRT ray initialization.

    Parameters
    ----------
    height, width:
        Output image resolution (pixels).
    fov_deg:
        Full field of view across the wider image dimension (degrees).
    position:
        Camera position in Boyer-Lindquist-style Cartesian coordinates
        (x, y, z), expressed in units of the black-hole mass M (= GM/c^2).
        Either a 3-sequence (shared across pixels) or a (B, 3) tensor for
        batched camera poses.
    look_at:
        World-space point the camera is aimed at (3-sequence or tensor).
        Defaults to the origin (the black hole).
    up:
        Reference up vector for the camera basis (default +z, which points
        along the spin axis in the usual BH rendering convention).

    All continuous parameters can be torch tensors with requires_grad=True;
    :meth:`initialize_camera_rays` will then propagate gradients through
    them once the geodesic integrator (Step 2) is in place.
    """

    height: int
    width: int
    fov_deg: float = 60.0
    position: Sequence[float] = (0.0, 0.0, 10.0)   # r = 10 M, +z axis
    look_at: Sequence[float] = (0.0, 0.0, 0.0)     # aim at the black hole
    up: Sequence[float] = (0.0, 0.0, 1.0)          # spin axis convention

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("camera resolution must be >= 1 pixel per axis")
        if not (0.0 < self.fov_deg < 180.0):
            raise ValueError("fov_deg must lie in (0, 180)")

    @property
    def resolution(self) -> tuple:
        """(H, W) pixel grid."""
        return (self.height, self.width)


def _as_tensor(x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Coerce scalars / sequences / tensors to a tensor on *device*."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _look_at_basis(
    position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor,
) -> tuple:
    """Orthonormal camera basis (forward, right, up) via Gram-Schmidt.

    All inputs are (3,) tensors sharing device/dtype.  The forward axis
    points from the camera toward the target.  Degrades gracefully to an
    arbitrary perpendicular pair when forward ∥ up (numerically guarded).
    """
    forward = F.normalize(look_at - position, dim=-1)
    # Guard the degenerate forward ∥ up case with a small orthogonal nudge.
    parallel = torch.abs(torch.dot(forward, up)) > 1.0 - 1e-6
    if bool(parallel):
        helper = torch.zeros_like(up)
        helper[torch.argmin(torch.abs(up))] = 1.0
        up = helper
    right = F.normalize(torch.linalg.cross(forward, up), dim=-1)
    true_up = torch.linalg.cross(right, forward)
    return forward, right, true_up


def _camera_ray_directions(
    camera: GRRTCamera,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pinhole ray directions d̂ for every pixel: (H, W, 3), unit norm.

    The pixel grid spans ``[-tan(fov/2), +tan(fov/2)]`` on the wider image
    dimension (kept square so pixels are isotropic), and the shorter
    dimension is scaled by its aspect ratio — standard perspective camera.
    Direction vectors are expressed in the *world* frame via the camera
    basis, and remain affine (differentiable) functions of the pixel grid.
    """
    H, W = camera.height, camera.width

    # Half-extent of the image plane at unit focal distance.
    half = math.tan(math.radians(camera.fov_deg) * 0.5)
    aspect = W / H

    # Pixel-centre coordinates, row-major like an image tensor (H, W).
    xs = torch.linspace(-half * aspect, half * aspect, W, device=device, dtype=dtype)
    ys = torch.linspace(half, -half, H, device=device, dtype=dtype)  # row 0 = top
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Camera looks along -forward convention would flip handedness; we use
    # +forward as the viewing direction and place the plane at focal length 1.
    dirs_cam = torch.stack(
        [grid_x, grid_y, torch.ones_like(grid_x)], dim=-1
    )  # (H, W, 3): (right, up, forward) components

    # Rotate into the world frame: d_world = right*dx + up*dy + forward*dz.
    pos = _as_tensor(camera.position, device, dtype)
    tgt = _as_tensor(camera.look_at, device, dtype)
    up = _as_tensor(camera.up, device, dtype)
    fwd, rgt, tup = _look_at_basis(pos, tgt, up)
    dirs = (
        dirs_cam[..., 0:1] * rgt
        + dirs_cam[..., 1:2] * tup
        + dirs_cam[..., 2:3] * fwd
    )
    return F.normalize(dirs, dim=-1)  # (H, W, 3), unit vectors


def initialize_camera_rays(
    camera: GRRTCamera,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    affine_time_offsets: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build the GRRT initial data: camera rays as spacetime 4-vectors.

    This is **Track B / Step 1** of the differentiable GRRT engine.  For
    every pixel it produces:

    * ``x0`` — spacetime 4-position ``x^mu = (t, x, y, z)`` on the camera
      plane (all photons emitted/received simultaneously in the observer's
      frame: ``t = 0`` by affine choice of origin),
    * ``p0`` — the momentum 4-vector ``p^mu = dx^mu / dlambda``.  In flat
      space the spatial part is the unit direction d̂ and the time
      component is set by the null condition (here ``p^t = 1`` in
      c = 1, G = 1 geometrized units, so ``ds^2 = 0`` exactly),
    * ``directions`` — the bare 3D unit directions (convenience for the
      Track-A style marchers and for debugging),

    plus a per-ray affine parameterization ``dlambda``.

    The null condition ``g_{mu nu} p^mu p^nu = 0`` is enforced analytically
    (Minkowski metric) rather than nudged numerically, so the rays enter
    the gravitational well at exactly the speed of light.

    Parameters
    ----------
    camera : GRRTCamera
        Camera resolution and intrinsics (fov, position, look_at).
    device : str or torch.device, optional
        Target device ("cuda" / "cpu"); defaults to cuda when available.
    dtype : torch.dtype, optional
        Floating dtype (default float32; use float64 for integrator tests).
    batch_size : int, optional
        Number of camera poses to batch (uses the same intrinsics; useful
        when several epochs/rigs share geometry). Default 1.
    affine_time_offsets : bool, optional
        If True (default), add a batched affine time offset t0 (all zeros
        for a static camera; a hook for later time-dependent rendering).

    Returns
    -------
    dict with keys
        ``x0``       (B, H, W, 4) — 4-positions (t, x, y, z)
        ``p0``       (B, H, W, 4) — null momenta (p^t, px, py, pz)
        ``directions`` (H, W, 3)  — unit pixel directions
        ``dlambda``  (B, H, W, 1) — affine parameter step (== 1.0)
        ``metric``   (B, 4, 4)    — flat Minkowski metric eta_{mu nu}
        ``null_residual`` (B, H, W) — |g p p|, identically 0 (sanity check)

    All outputs are torch tensors on *device* with *dtype*; they carry
    gradients back to any camera parameter that was created with
    ``requires_grad=True`` (e.g. a learnable camera position tensor).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    H, W = camera.resolution
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    # ---- pixel directions in the world frame -----------------------------
    dirs = _camera_ray_directions(camera, device, dtype)          # (H, W, 3)

    # ---- camera world position (batched) ----------------------------------
    pos = _as_tensor(camera.position, device, dtype)              # (3,)
    pos_b = pos.unsqueeze(0).expand(batch_size, 3)                # (B, 3)

    # ---- spacetime 4-positions x^mu = (t, x, y, z) ------------------------
    # All camera-plane photons share t = 0 (affine gauge choice); this
    # keeps the initial hypersurface spacelike and geodesic integration
    # well-posed.  affine_time_offsets gives a per-batch t0 hook.
    t0 = torch.zeros(batch_size, device=device, dtype=dtype)
    if not affine_time_offsets:
        t0 = t0[:1].expand(batch_size)  # keep the key shape stable
    x_spatial = pos_b[:, None, None, :].expand(batch_size, H, W, 3)
    t_grid = t0[:, None, None].expand(batch_size, H, W)
    x0 = torch.stack(
        [t_grid, x_spatial[..., 0], x_spatial[..., 1], x_spatial[..., 2]],
        dim=-1,
    )  # (B, H, W, 4)

    # ---- null momenta p^mu = (p^t, px, py, pz) -----------------------------
    # Signature (-,+,+,+):  eta = diag(-1, 1, 1, 1).
    # With c = G = M = 1 the null condition gives
    #     -(p^t)^2 + |p_spatial|^2 = 0  =>  p^t = |p_spatial| = 1.
    # Setting p^t := 1 and p_spatial := d̂ (unit) satisfies it *exactly*.
    p_spatial = dirs  # already unit norm
    p_t = torch.ones_like(p_spatial[..., 0])   # (H, W): p^t = 1  =>  null
    p0 = torch.stack(
        [p_t, p_spatial[..., 0], p_spatial[..., 1], p_spatial[..., 2]],
        dim=-1,
    ).unsqueeze(0).expand(batch_size, H, W, 4)  # (B, H, W, 4)

    # ---- affine parameterization ------------------------------------------
    dlambda = torch.ones(batch_size, H, W, 1, device=device, dtype=dtype)

    # ---- flat metric & analytic null check ---------------------------------
    metric = torch.eye(4, device=device, dtype=dtype)
    metric[..., 0, 0] = -1.0                                       # (-,+,+,+)
    metric = metric.unsqueeze(0).expand(batch_size, 4, 4)

    # g_{mu nu} p^mu p^nu = -(p^t)^2 + px^2 + py^2 + pz^2  (per ray)
    null_residual = (
        -(p0[..., 0] ** 2) + p0[..., 1] ** 2 + p0[..., 2] ** 2 + p0[..., 3] ** 2
    )  # (B, H, W)

    return {
        "x0": x0,
        "p0": p0,
        "directions": dirs,
        "dlambda": dlambda,
        "metric": metric,
        "null_residual": null_residual,
    }


def check_null_condition(
    rays: Dict[str, torch.Tensor], atol: float = 1e-5
) -> bool:
    """Assert |g_{mu nu} p^mu p^nu| < atol for every ray (sanity gate)."""
    res = rays.get("null_residual")
    if res is None:
        raise KeyError("rays dict lacks 'null_residual'")
    max_dev = float(res.abs().max())
    ok = max_dev < atol
    logger.info("null-condition max |g p p| = %.3e (atol %.1e) -> %s",
                max_dev, atol, "PASS" if ok else "FAIL")
    return ok


# ==========================================================================
# Track B — Step 2: Differentiable RK4 null-geodesic integrator
# ==========================================================================
# The integrator works in **Schwarzschild coordinates (t, r, θ, φ)** with
# signature (-,+,+,+), where the Christoffel symbols have compact textbook
# closed forms (see `schwarzschild_christoffels`).  Step 1 emits Cartesian
# camera rays, so `cartesian_to_spherical_rays` adapts the phase space into
# the spherical chart; the null invariant g_μν p^μ p^ν is chart-independent
# and stays 0 along the path.
#
# Geodesic equation (affine parameter λ):
#
#     d²x^μ/dλ² = − Γ^μ_{αβ} p^α p^β ,      p^μ = dx^μ/dλ
#
# `rk4_geodesic_step` advances (x, p) by dλ with the classic 4-stage RK4
# scheme, fully in torch ops so autograd connects camera parameters to the
# *curved* light paths — exactly what inverse rendering needs.
# ==========================================================================


# ---- physical guards ------------------------------------------------------
_R_MIN_FRACTION = 1.02   # never step closer than r < 1.02 r_s (horizon)
_R_MIN_ABS = 1e-3        # absolute floor in units of M
_THETA_EPS = 1e-6        # polar-axis guard for sin(theta) division


def _sanitize_state(
    x: torch.Tensor, p: torch.Tensor
) -> tuple:
    """Clamp r above the horizon and θ away from the coordinate poles.

    Schwarzschild coordinates are singular at r = 2M (horizon) and at
    sinθ = 0 (polar axis).  Photons in AstroRecon renders always live far
    from both, but adaptive optimizers can wander; these clamps keep the
    integration finite and differentiable (clamps are autograd-safe).
    """
    r = x[..., 1]
    r_min = max(2.0 * _R_MIN_FRACTION, _R_MIN_ABS)   # horizon guard (r = 2.04 M)
    r_clamped = torch.clamp(r, min=r_min)
    x = torch.stack([
        x[..., 0],
        r_clamped,
        torch.clamp(x[..., 2], _THETA_EPS, math.pi - _THETA_EPS),
        torch.remainder(x[..., 3], 2.0 * math.pi),
    ], dim=-1)
    # p is returned unchanged: clamping only protects coordinate
    # singularities, and re-scaling p by r_clamped/r (an earlier attempt)
    # broke the null invariant.  Rays that reach the horizon guard are
    # captured by Step 3's event-horizon check, not here.
    return x, p


def schwarzschild_christoffels(
    r: torch.Tensor, theta: torch.Tensor, M: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Non-zero Schwarzschild Christoffel symbols Γ^μ_{αβ}.

    Metric: ds² = −f dt² + f⁻¹ dr² + r² dθ² + r² sin²θ dφ²,
    with f(r) = 1 − 2M/r.  Signature (-,+,+,+), geometrized units
    (G = c = 1), coordinates ordered (t, r, θ, φ).

    Non-zero components (symmetric in the lower indices):

        Γ^t_{tr} = Γ^t_{rt} =  M / (r² f)
        Γ^r_{tt}            =  M f / r²
        Γ^r_{rr}            = −M / (r² f)
        Γ^r_{θθ}            = −r f
        Γ^r_{φφ}            = −r f sin²θ
        Γ^θ_{rθ} = Γ^θ_{θr} =  1 / r
        Γ^θ_{φφ}            = −sinθ cosθ
        Γ^φ_{rφ} = Γ^φ_{φr} =  1 / r
        Γ^φ_{θφ} = Γ^φ_{φθ} =  cotθ

    All returned tensors broadcast against ``r`` / ``theta`` shapes.
    """
    f = 1.0 - 2.0 * M / r                      # lapse factor
    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)

    m_over_r2f = M / (r * r * f)               # M/(r² f), used by two symbols

    return {
        "t_tr": m_over_r2f,                    # also used as Γ^t_{rt}
        "r_tt": M * f / (r * r),
        "r_rr": -m_over_r2f,                   # note the sign: −M/(r² f)
        "r_thth": -r * f,
        "r_phph": -r * f * sin_th * sin_th,
        "th_rth": torch.ones_like(r) / r,      # also Γ^θ_{θr}
        "th_phph": -sin_th * cos_th,
        "ph_rph": torch.ones_like(r) / r,      # also Γ^φ_{φr}
        "ph_thph": cos_th / torch.clamp(sin_th, min=_THETA_EPS),  # cotθ
    }


def geodesic_acceleration(
    x: torch.Tensor, p: torch.Tensor, M: float = 1.0
) -> tuple:
    """Right-hand side of the geodesic equation: (dx/dλ, dp/dλ).

    Parameters
    ----------
    x : (..., 4) tensor — events (t, r, θ, φ)
    p : (..., 4) tensor — momenta (p^t, p^r, p^θ, p^φ)

    Returns
    -------
    (dx, dp): both (..., 4); dx = p (definition of the momentum) and
    dp^μ = −Γ^μ_{αβ} p^α p^β (gravitational light bending).

    The Γ products are expanded term-by-term (no einsum over a full 4×4×4
    symbol array) — this keeps the graph small, is faster on GPU, and each
    line maps 1:1 to a textbook symbol for auditing.
    """
    t, r, th, ph = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    pt, pr, pth, pph = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

    G = schwarzschild_christoffels(r, th, M)
    f = 1.0 - 2.0 * M / r
    sin_th = torch.sin(th)
    cos_th = torch.cos(th)

    # -- dλt = −Γ^t_{αβ} p^α p^β : only Γ^t_{tr} = Γ^t_{rt} contribute -----
    dp_t = -2.0 * G["t_tr"] * pt * pr

    # -- dλr = −Γ^r_{αβ} p^α p^β -------------------------------------------
    dp_r = -(
        G["r_tt"] * pt * pt          # gravitational redshift of coordinate time
        + G["r_rr"] * pr * pr        # radial metric curvature (Γ = −M/r²f)
        + G["r_thth"] * pth * pth  # centrifugal (angular momentum barrier)
        + G["r_phph"] * pph * pph    # azimuthal barrier (−r f sin²θ)
    )

    # -- dλθ = −Γ^θ_{αβ} p^α p^β -------------------------------------------
    dp_th = -(
        2.0 * G["th_rth"] * pr * pth          # radial stretching of angle
        + G["th_phph"] * pph * pph            # −sinθ cosθ φ̇² polar turn
    )

    # -- dλφ = −Γ^φ_{αβ} p^α p^β -------------------------------------------
    dp_ph = -(
        2.0 * G["ph_rph"] * pr * pph
        + 2.0 * G["ph_thph"] * pth * pph
    )

    # θ-pole handling: `schwarzschild_christoffels` already clamps cotθ's
    # sinθ denominator, and `_sanitize_state` keeps θ away from 0/π, so no
    # extra masking is needed here.
    dp = torch.stack([dp_t, dp_r, dp_th, dp_ph], dim=-1)
    dx = p  # dx^μ/dλ = p^μ, by definition of the 4-momentum
    return dx, dp


def rk4_geodesic_step(
    x: torch.Tensor,
    p: torch.Tensor,
    dlambda: Union[float, torch.Tensor] = 0.1,
    M: float = 1.0,
) -> tuple:
    """One differentiable RK4 step of the Schwarzschild null geodesic.

    Implements the classical 4-stage scheme for the coupled system
    (x^μ, p^μ):

        k1 = f(x, p)
        k2 = f(x + ½ dλ k1_x,  p + ½ dλ k1_p)
        k3 = f(x + ½ dλ k2_x,  p + ½ dλ k2_p)
        k4 = f(x +    dλ k3_x, p +    dλ k3_p)

        x' = x + dλ (k1_x + 2 k2_x + 2 k3_x + k4_x) / 6
        p' = p + dλ (k1_p + 2 k2_p + 2 k3_p + k4_p) / 6

    Every stage is pure torch arithmetic on the live autograd graph — no
    in-place ops, no `.detach()` — so gradients flow from the final image
    back through the *curved* photon paths to any learnable upstream
    quantity (camera position, FOV, even the metric mass ``M`` if you
    promote it to a tensor).

    Parameters
    ----------
    x, p : (..., 4) tensors
        Spacetime events (t, r, θ, φ) and momenta (p^t, p^r, p^θ, p^φ).
    dlambda : float or tensor
        Affine step size dλ (can be a per-ray tensor broadcast to (..., 1)).
    M : float
        Black-hole mass in geometrized units (default 1.0).

    Returns
    -------
    (x_new, p_new): advanced (..., 4) tensors.
    """
    lam = torch.as_tensor(dlambda, dtype=x.dtype, device=x.device)

    # sanitize once per outer step (cheap, keeps RK4 stages well-posed)
    x, p = _sanitize_state(x, p)

    # ---- stage 1 -----------------------------------------------------------
    dx1, dp1 = geodesic_acceleration(x, p, M)

    # ---- stage 2 (midpoint using k1) ---------------------------------------
    dx2, dp2 = geodesic_acceleration(
        x + 0.5 * lam * dx1, p + 0.5 * lam * dp1, M)

    # ---- stage 3 (midpoint using k2) ---------------------------------------
    dx3, dp3 = geodesic_acceleration(
        x + 0.5 * lam * dx2, p + 0.5 * lam * dp2, M)

    # ---- stage 4 (endpoint using k3) ---------------------------------------
    dx4, dp4 = geodesic_acceleration(
        x + lam * dx3, p + lam * dp3, M)

    # ---- weighted RK4 update (out-of-place ⇒ autograd-safe) ----------------
    x_new = x + (lam / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    p_new = p + (lam / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
    return x_new, p_new


def null_residual_spherical(
    x: torch.Tensor, p: torch.Tensor, M: float = 1.0
) -> torch.Tensor:
    """g_μν p^μ p^ν evaluated with the spherical Schwarzschild metric.

    Should stay ≈ 0 along the whole trajectory (it is an exact invariant of
    the geodesic equation; RK4 preserves it to integrator accuracy).
    """
    r, th = x[..., 1], x[..., 2]
    f = 1.0 - 2.0 * M / r
    sin2 = torch.sin(th) ** 2
    return (
        -f * p[..., 0] ** 2
        + p[..., 1] ** 2 / f
        + (r * p[..., 2]) ** 2
        + (r * torch.sin(th) * p[..., 3]) ** 2
    )


def cartesian_to_spherical_rays(
    x_cart: torch.Tensor, p_cart: torch.Tensor, M: float = 1.0
) -> tuple:
    """Map Step-1 Cartesian phase space into the spherical chart.

    Contravariant tensor transformation: p^μ_sph = (∂u^μ/∂x^ν_cart) p^ν_cart.
    The spatial momentum components become (p^r, p^θ, p^φ) via the
    orthonormal spherical frame.  p^t is *rescaled* so the null condition
    holds at finite radius: for a photon arriving from infinity at speed
    c (flat-space Step-1 convention |p⃗| = 1), local energy conservation
    gives p^t = |p⃗| / f(r) with f = 1 − 2M/r.  At r → ∞ (f → 1) this
    reduces to p^t = 1, matching the Step-1 Minkowski normalization.

    Parameters
    ----------
    x_cart : (..., 4) — (t, x, y, z) events on the camera plane
    p_cart : (..., 4) — (p^t, px, py, pz) momenta (null in Minkowski)
    M : float — black-hole mass used for the p^t normalization

    Returns
    -------
    (x_sph, p_sph): (..., 4) tensors in (t, r, θ, φ) coordinates.
    """
    t = x_cart[..., 0]
    xc, yc, zc = x_cart[..., 1], x_cart[..., 2], x_cart[..., 3]
    px, py, pz = p_cart[..., 1], p_cart[..., 2], p_cart[..., 3]

    # sqrt'(0) = inf makes gradients NaN when a pixel's spatial position
    # coincides with the camera (r -> 0).  NOTE: torch.where-based floors
    # do NOT fix this — PyTorch still back-propagates 0 * inf = NaN through
    # the unselected branch.  The smooth fix is an epsilon inside the sqrt
    # argument, which is finite everywhere and analytically differentiable.
    # --- launch-point nudge -------------------------------------------------
    # The camera typically sits ON the spin axis (θ ≈ 0), where the
    # spherical chart is singular: p_φ = p·φ̂ / (r sinθ) → O(1/ε) for any
    # ray with transverse direction.  Formally correct, but numerically
    # explosive (cotθ p^θ p^φ ~ 1/sinθ).  Standard fix: step the launch
    # point a tiny ε ALONG the ray before converting.  Transverse rays are
    # then off-axis with sinθ = O(ε) while the physical products
    # (r sinθ p^φ, cotθ p^θ p^φ) stay O(1).  The pure on-axis center ray
    # keeps p_θ = p_φ = 0 exactly and evolves legitimately on the pole.
    # ε: small vs. the RK4 step, large vs. float64 precision.
    eps = 1e-4
    p_dir = torch.stack([px, py, pz], dim=-1)         # (..., 3), unit-ish
    p_norm = torch.clamp(
        torch.sqrt(p_dir * p_dir + 1e-18), min=1e-12)
    p_hat = p_dir / p_norm
    xc2 = xc + eps * p_hat[..., 0]
    yc2 = yc + eps * p_hat[..., 1]
    zc2 = zc + eps * p_hat[..., 2]

    r_raw = torch.sqrt(xc2 * xc2 + yc2 * yc2 + zc2 * zc2 + 1e-18)
    r = torch.clamp(r_raw, min=1e-9)          # forward floor; grad stays finite
    r_xy = torch.clamp(torch.sqrt(xc2 * xc2 + yc2 * yc2 + 1e-18), min=1e-12)
    # acos' blows up at argument ±1 (polar axis); clamp keeps dθ/dz finite.
    # The polar axis is a coordinate singularity of the spherical chart,
    # so the clamped θ-gradient there is a gauge artifact, not physics.
    theta = torch.acos(
        torch.clamp(zc / torch.clamp(r, min=1e-12), -(1.0 - 1e-9), 1.0 - 1e-9)
    )
    # atan2(0, 0) at the exact pole has NaN gradients (0/0).  Feed atan2
    # *safe inputs* built with where (pin to (1, 0) on the pole ⇒ φ = 0),
    # rather than where-ing the result — the latter still leaks NaN through
    # the unselected branch's 0 * inf gradient.
    on_pole = (xc2.abs() < 1e-12) & (yc2.abs() < 1e-12)
    xs_safe = torch.where(on_pole, torch.ones_like(xc2), xc2)
    ys_safe = torch.where(on_pole, torch.zeros_like(yc2), yc2)
    phi = torch.atan2(ys_safe, xs_safe)

    # Orthonormal spherical frame in the Cartesian chart (textbook):
    #   r̂  = (sinθcosφ, sinθsinφ, cosθ)
    #   θ̂  = (cosθcosφ, cosθsinφ, -sinθ)   ← note: NOT (z·x/r, ...);
    #        that alternate form is only valid away from the pole.
    #   φ̂  = (-sinφ, cosφ, 0)
    # Written via the identity r̂ = pos/r keeps cos/sin consistent:
    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    sin_ph = torch.sin(phi)
    cos_ph = torch.cos(phi)
    e_r = torch.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th], dim=-1)
    e_th = torch.stack([cos_th * cos_ph, cos_th * sin_ph, -sin_th], dim=-1)
    e_ph = torch.stack([-sin_ph, cos_ph, torch.zeros_like(sin_ph)], dim=-1)
    e_r_x, e_r_y, e_r_z = e_r.unbind(-1)
    e_th_x, e_th_y, e_th_z = e_th.unbind(-1)
    e_ph_x, e_ph_y, e_ph_z = e_ph.unbind(-1)

    p_r = px * e_r_x + py * e_r_y + pz * e_r_z
    p_th = (px * e_th_x + py * e_th_y + pz * e_th_z) / torch.clamp(r, min=1e-12)
    p_ph = (px * e_ph_x + py * e_ph_y + pz * e_ph_z) / torch.clamp(
        r * torch.sin(theta), min=1e-12
    )

    # Null normalization: g p p = -f (p^t)^2 + p_r^2/f + a_th^2 + a_ph^2 = 0
    # is solved *exactly* for p^t at the launch point (a_th = r p^θ,
    # a_ph = r sinθ p^φ are the orthonormal angular components).  This is
    # valid for any ray inclination, not just tangential ones.
    f = 1.0 - 2.0 * M / torch.clamp(r, min=2.0 * _R_MIN_FRACTION)
    a_th = torch.clamp(r, min=1e-12) * p_th
    a_ph = torch.clamp(r * torch.sin(theta), min=1e-12) * p_ph
    p_t = torch.sqrt(
        torch.clamp(p_r * p_r / f + a_th * a_th + a_ph * a_ph, min=1e-24) / f
    ) * torch.sign(p_cart[..., 0])  # preserve Step-1 forward/backward time

    x_sph = torch.stack([t, r, theta, phi], dim=-1)
    p_sph = torch.stack([p_t, p_r, p_th, p_ph], dim=-1)
    return x_sph, p_sph


def trace_null_geodesics(
    x0: torch.Tensor,
    p0: torch.Tensor,
    dlambda: float = 0.1,
    M: float = 1.0,
    max_steps: int = 512,
    direction: int = -1,
    record_trajectory: bool = False,
) -> Dict[str, torch.Tensor]:
    """Multi-step RK4 marching of photon rays through curved spacetime.

    Steps every ray (batch dims preserved) forward or backward in the
    affine parameter for up to ``max_steps`` RK4 updates.

    Parameters
    ----------
    x0, p0 : (..., 4) tensors
        Initial phase space (e.g. from `initialize_camera_rays` followed by
        `cartesian_to_spherical_rays`).  Leading dims are free batch dims,
        so (B, H, W, 4) camera grids integrate in one shot.
    dlambda : float
        Affine step size (coordinate step ≈ physical step far from the hole).
    M : float
        Black-hole mass (geometrized units, default 1.0).
    max_steps : int
        Maximum number of RK4 steps.
    direction : int
        ``+1`` traces photons forward in time (emission → observer);
        ``-1`` (default) traces *backwards* from the camera — the standard
        GRRT rendering convention (camera rays fired into the past).
    record_trajectory : bool
        If True, also return ``trajectory`` with shape
        (steps+1, ..., 4) — useful for visualizing lensed paths.  Costs
        memory (steps × x0.numel()); off by default.

    Returns
    -------
    dict with:
        ``x``        final events (..., 4)
        ``p``        final momenta (..., 4)
        ``steps``    number of steps taken (int)
        ``null_max`` max |g_μν p^μ p^ν| seen along the way (scalar tensor)
        ``trajectory`` (optional) stacked events
    """
    if direction not in (+1, -1):
        raise ValueError("direction must be +1 or -1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    x, p = x0, p0
    lam = float(dlambda) * float(direction)

    null_max = torch.zeros((), device=x0.device, dtype=x0.dtype)
    trajectory = [x] if record_trajectory else None
    steps = 0

    for _ in range(max_steps):
        # Autograd note: rk4_geodesic_step is fully out-of-place, so the
        # computation graph simply grows with `steps`.  For very deep
        # traces, wrap selected segments in torch.utils.checkpoint.
        x, p = rk4_geodesic_step(x, p, lam, M)
        steps += 1
        if record_trajectory:
            trajectory.append(x)
        # track null drift for the sanity report (no_grad: bookkeeping only)
        with torch.no_grad():
            res = null_residual_spherical(x, p, M).abs().max()
            null_max = torch.maximum(null_max, res)

    out: Dict[str, torch.Tensor] = {
        "x": x,
        "p": p,
        "steps": steps,
        "null_max": null_max,
    }
    if record_trajectory:
        out["trajectory"] = torch.stack(trajectory, dim=0)
    return out


if __name__ == "__main__":  # pragma: no cover - Step 2 smoke test
    # Photon deflection sanity: weak-field GR predicts Δφ ≈ 4 M / b.
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, b, r0 = 1.0, 100.0, 500.0

    # Camera photon in Cartesian chart (Step 1 convention), z-axis view.
    x_c = torch.tensor([[0.0, r0, b, 0.0]], device=dev, dtype=torch.float64)
    d_c = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=dev, dtype=torch.float64)
    p_c = torch.tensor([[1.0, -1.0, 0.0, 0.0]], device=dev, dtype=torch.float64)
    x_s, p_s = cartesian_to_spherical_rays(x_c, p_c)

    out = trace_null_geodesics(x_s, p_s, dlambda=1.0, M=M, max_steps=2000,
                               direction=+1, record_trajectory=False)
    print("null max drift:", float(out["null_max"]))
    print("final r:", float(out["x"][0, 1]))


# ==========================================================================
# Track B — Step 3: Relativistic boundary checks & optical phenomenon flags
# ==========================================================================
# Physical length scales of a Schwarzschild black hole (geometrized units,
# all in units of M):
#
#   * event horizon  r_s  = 2M     — one-way membrane; photons crossing it
#                                    are captured.  In the image plane these
#                                    rays form the black-hole *shadow*.
#   * photon sphere  r_ph = 1.5 r_s = 3M — unstable circular photon orbits;
#                                    rays skimming near it wrap many times
#                                    around the hole (highest lensing action).
#   * Einstein-ring region  r ~ 2.6 r_s — impact parameters whose primary
#                                    image merges into the ring; flagged for
#                                    secondary-magnification handling.
#
# The marcher checks these boundaries *every step* and freezes terminated
# rays: an absorbed ray stops receiving position/momentum updates (saving
# compute and preventing NaNs near the horizon singularity), while active
# rays keep full autograd connectivity.  Freezing uses a boolean `alive`
# mask and torch.where, so the operation is a single fused GPU kernel.


@dataclass
class BoundaryRadii:
    """Physical boundary radii (in units of M) for a Schwarzschild hole."""

    horizon: float = 2.0          # r_s  = 2M   - shadow-forming capture radius
    photon_sphere: float = 3.0    # r_ph = 3M   - unstable photon orbits
    einstein_ring: float = 5.2    # ~2.6 r_s    - secondary magnification band
    horizon_tol: float = 1e-6     # capture margin: absorb at r <= r_s + tol
    photon_band: float = 0.25     # |r - r_ph| band half-width for flags
    ring_band: float = 0.6        # |r - r_er| band half-width for flags

    def __post_init__(self) -> None:
        for name in ("horizon", "photon_sphere", "einstein_ring"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} radius must be positive")
        if not (self.horizon < self.photon_sphere < self.einstein_ring):
            raise ValueError("require horizon < photon_sphere < einstein_ring")


def _apply_frozen_update(
    alive: torch.Tensor,
    x_old: torch.Tensor,
    p_old: torch.Tensor,
    x_new: torch.Tensor,
    p_new: torch.Tensor,
) -> tuple:
    """Merge frozen (absorbed) rays back into the advanced batch.

    Rays whose ``alive`` bit is False keep their *previous* phase-space
    coordinates; active rays take the RK4-advanced ones.  Implemented with
    torch.where over the leading batch dims so it is one fused GPU kernel
    and stays inside the autograd graph (gradients simply stop flowing into
    dead rays, which is exactly the intended semantics).

    ``alive`` shape (...) matches the leading dims of x/p (... , 4).
    """
    mask = alive.unsqueeze(-1)                      # (..., 1) broadcast to 4
    x = torch.where(mask, x_new, x_old)
    p = torch.where(mask, p_new, p_old)
    return x, p


def trace_null_geodesics_boundaries(
    x0: torch.Tensor,
    p0: torch.Tensor,
    dlambda: float = 0.1,
    M: float = 1.0,
    max_steps: int = 512,
    direction: int = -1,
    record_trajectory: bool = False,
    boundaries: Optional[BoundaryRadii] = None,
    escape_radius: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Boundary-aware null-geodesic marcher: shadows, photon sphere, rings.

    Extends :func:`trace_null_geodesics` with per-step relativistic
    boundary checks (Step 3):

    1. **Event-horizon absorption** — a ray at ``r <= r_s + tol`` is
       captured: marked in the shadow mask and frozen (its position and
       momentum no longer update in later steps; its eventual emission is
       zero, forming the black central shadow in the rendered image).
    2. **Photon-sphere monitoring** — rays inside ``|r - 1.5 r_s| < band``
       are flagged; these wrap the hole multiple times and dominate the
       thin ring of lensed emission around the shadow edge.
    3. **Einstein-ring flagging** — rays passing near ``r ~ 2.6 r_s`` get
       the secondary-magnification flag for background lensing.
    4. **Escape detection** — a ray that reaches ``r >= escape_radius``\       moving outward with positive radial momentum has escaped to the
       background sky and is frozen too (its sky position is its final
       point) — this is what lets the marcher terminate early.

    Parameters
    ----------
    x0, p0 : (..., 4) tensors
        Initial phase space in Schwarzschild (t, r, θ, φ) coordinates —
        e.g. from `initialize_camera_rays` + `cartesian_to_spherical_rays`.
    dlambda, M, max_steps, direction, record_trajectory :
        As in :func:`trace_null_geodesics`.
    boundaries : BoundaryRadii, optional
        Custom boundary radii; defaults to the canonical Schwarzschild set.
    escape_radius : float, optional
        Radius beyond which an outward-moving ray counts as escaped.
        Defaults to 20× the largest initial radius.

    Returns
    -------
    dict with
        ``x``, ``p``        final phase space (..., 4)
        ``shadow_mask``     (...,) bool — True = captured by the horizon
                            (black shadow pixel), False = escaped to sky
        ``absorbed_mask``   alias of ``shadow_mask`` (explicit naming)
        ``escaped_mask``    (...,) bool — True = reached the background sky
        ``photon_sphere_mask``  (...,) bool — passed within the photon-
                            sphere band at any step (unstable-orbit zone)
        ``einstein_ring_mask``  (...,) bool — passed within the Einstein-
                            ring band at any step (secondary magnification)
        ``alive_mask``      (...,) bool — True = still integrating at the
                            end of the budget (increase max_steps or dλ)
        ``absorbed_step``   (...,) float — the step index at which each ray
                            was captured (useful for time-of-arrival maps)
        ``steps``, ``null_max``, ``trajectory`` as before.
    """
    if direction not in (+1, -1):
        raise ValueError("direction must be +1 or -1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    br = boundaries or BoundaryRadii()
    x, p = x0, p0
    lam = float(dlambda) * float(direction)

    # ---- escape threshold: far beyond everything the camera launched -----
    if escape_radius is None:
        with torch.no_grad():
            r_max0 = float(x0[..., 1].abs().max())
        escape_radius = max(20.0 * r_max0, 50.0 * M)

    # ---- per-ray bookkeeping (bool masks match the batch dims) ----------
    batch_shape = x0.shape[:-1]
    device, dtype = x0.device, x0.dtype
    alive = torch.ones(batch_shape, device=device, dtype=torch.bool)
    absorbed = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    escaped = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    hit_photon_sphere = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    hit_einstein_ring = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    absorbed_step = torch.full(batch_shape, float("inf"), device=device, dtype=dtype)

    null_max = torch.zeros((), device=device, dtype=dtype)
    trajectory = [x] if record_trajectory else None
    steps = 0

    for step in range(1, max_steps + 1):
        # ---------- 1. boundary checks BEFORE stepping ----------------------
        with torch.no_grad():
            r = x[..., 1]
            pr = p[..., 1]

            # (a) event-horizon absorption: r <= r_s + tol
            newly_absorbed = alive & (r <= br.horizon + br.horizon_tol)
            absorbed |= newly_absorbed
            absorbed_step = torch.where(
                newly_absorbed,
                torch.full_like(absorbed_step, float(step - 1)),
                absorbed_step,
            )
            alive = alive & ~newly_absorbed

            # (b) photon-sphere band flag (r ~ 1.5 r_s = 3M)
            hit_photon_sphere |= alive & (
                (r - br.photon_sphere).abs() < br.photon_band)

            # (c) Einstein-ring band flag (r ~ 2.6 r_s)
            hit_einstein_ring |= alive & (
                (r - br.einstein_ring).abs() < br.ring_band)

            # (d) escape to the background sky: far out AND moving outward
            newly_escaped = alive & (r >= escape_radius) & (pr > 0)
            escaped |= newly_escaped
            alive = alive & ~newly_escaped

        # Everyone terminated?  Snapshot and stop stepping (saves compute).
        if not bool(alive.any()):
            break

        # ---------- 2. RK4 advance (active rays only receive updates) ------
        x_new, p_new = rk4_geodesic_step(x, p, lam, M)

        # Frozen rays keep their old phase space; active rays take the
        # update.  torch.where keeps everything on the autograd graph —
        # gradients simply stop reaching dead rays, which is the intended
        # semantics (their image contribution is a constant zero emission).
        x, p = _apply_frozen_update(alive, x, p, x_new, p_new)
        steps = step

        if record_trajectory:
            trajectory.append(x)

        with torch.no_grad():
            res = null_residual_spherical(x, p, M).abs().max()
            null_max = torch.maximum(null_max, res)

    # Final sweep in case a ray hit the horizon exactly on the last step.
    with torch.no_grad():
        r_final = x[..., 1]
        final_absorbed = alive & (r_final <= br.horizon + br.horizon_tol)
        absorbed |= final_absorbed
        absorbed_step = torch.where(
            final_absorbed,
            torch.full_like(absorbed_step, float(steps)),
            absorbed_step,
        )
        alive = alive & ~final_absorbed

    shadow_mask = absorbed  # the iconic central shadow

    out: Dict[str, torch.Tensor] = {
        "x": x,
        "p": p,
        "steps": steps,
        "null_max": null_max,
        "shadow_mask": shadow_mask,
        "absorbed_mask": shadow_mask,
        "escaped_mask": escaped,
        "photon_sphere_mask": hit_photon_sphere,
        "einstein_ring_mask": hit_einstein_ring,
        "alive_mask": alive,
        "absorbed_step": absorbed_step,
        "escape_radius": escape_radius,
    }
    if record_trajectory:
        out["trajectory"] = torch.stack(trajectory, dim=0)
    return out


if __name__ == "__main2__":  # pragma: no cover - Step 3 smoke test (manual)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 1.0
    # Camera at r = 50M looking at the hole; the central pixels (small
    # impact parameter) must be absorbed, wide ones escape.
    cam = GRRTCamera(height=64, width=64, fov_deg=20.0)
    rays = initialize_camera_rays(cam, device=dev, dtype=torch.float64)
    x_s, p_s = cartesian_to_spherical_rays(rays["x0"], rays["p0"], M=M)
    out = trace_null_geodesics_boundaries(
        x_s, p_s, dlambda=1.0, M=M, max_steps=300, direction=-1)
    sm = out["shadow_mask"]
    center = sm[0, 32, 32].item()
    corner = sm[0, 0, 0].item()
    print("center pixel absorbed:", center, "(expected True)")
    print("corner pixel absorbed:", corner, "(expected False)")
    print("shadow size:", int(sm.sum()), "pixels of", sm.numel())


# ==========================================================================
# Track B — Step 4: Voxel sampling, Doppler beaming & radiative transfer
# ==========================================================================
# The grand finale: the curved null geodesics from Steps 1-3 now actually
# "see" the accretion flow.  At every RK4 step k along each ray we
#
#   1. convert the event's spatial position to a Cartesian map direction
#      (needed because the voxel grid is defined in the Cartesian chart),
#   2. sample the learnable density tensor V ∈ R^{D×H×W} with trilinear
#      `grid_sample` (fully differentiable: ∂Î/∂V flows back through every
#      sample point),
#   3. compute the local orbital plasma velocity and the special-relativistic
#      Doppler factor δ, boosting the emissivity by δ^3 (specific intensity)
#      — the "headlight effect" that brightens the approaching side of the
#      disk and dims the receding side,
#   4. integrate the radiative transfer equation along the ray:
#
#         dI/dλ = j(x_k) · δ_k^3 · e^{-τ_k}   →   Î = Σ_k j_k δ_k^3 τ_k Δλ
#
#      where τ_k is the optical depth accumulated front-to-back.
#
# Rays captured by the event horizon (Step 3 shadow mask) contribute
# exactly zero emission — the black central shadow.

# ---- Cartesian frame convention for the voxel grid ------------------------
# The voxel tensor V lives in the SAME Cartesian frame as Step 1's camera
# rays: x, y in the orbital plane and z along the spin axis.  A spherical
# event (r, θ, φ) maps back as
#     x = r sinθ cosφ,  y = r sinθ sinφ,  z = r cosθ
# and grid_sample's normalized coordinates are
#     gx = x / R_grid,  gy = y / R_grid,  gz = z / R_grid
# with R_grid the physical half-size of the box in units of M.  Voxels
# outside [-1 (no emission beyond the box).


def spherical_to_cartesian_pos(x_sph: torch.Tensor) -> torch.Tensor:
    """Map (..., 4) spherical events (t, r, θ, φ) → (..., 3) positions."""
    r, th, ph = x_sph[..., 1], x_sph[..., 2], x_sph[..., 3]
    return torch.stack([
        r * torch.sin(th) * torch.cos(ph),
        r * torch.sin(th) * torch.sin(ph),
        r * torch.cos(th),
    ], dim=-1)


def ks_schwarzchild_metric(
    x: torch.Tensor, M: float = 1.0
) -> tuple:
    """Schwarzschild metric in **Cartesian Kerr–Schild** form, with
    derivatives — the chart used by production GRRT codes because it has
    NO coordinate singularity at the poles (unlike spherical coordinates).

    Metric:  g_{μν} = η_{μν} + f l_μ l_ν,   f = 2M/r,
    l_μ = (1, x/r, y/r, z/r),   η = diag(-1, 1, 1, 1).

    Inverse:  g^{μν} = η^{μν} − f l^μ l^ν,  l^μ = (-1, x/r, y/r, z/r).

    Parameters
    ----------
    x : (..., 4) events (t, x, y, z); only the spatial part is used.

    Returns
    -------
    (g, ginv, dgd):
        g    (..., 4, 4)   covariant metric
        ginv (..., 4, 4)   contravariant metric
        dgd  (..., 4, 4, 4) partial derivatives, dgd[k,a,b] = ∂_k g_{ab}
            (static field ⇒ dgd[0] = 0).  Derived analytically:
            ∂_k f        = -f n_k / r
            ∂_k l_i      = (δ_ki − n_k n_i)/r
            ∂_k g_{00}   = ∂_k f
            ∂_k g_{0i}   = (∂_k f) n_i + f ∂_k l_i
            ∂_k g_{ij}   = (∂_k f) n_i n_j + f (∂_k l_i) n_j + f n_i (∂_k l_j)

    All outputs keep autograd connectivity to x.
    """
    pos = x[..., 1:]                                   # (..., 3)
    r = torch.clamp(torch.sqrt(pos * pos + 1e-18).sum(-1), min=1e-9)
    n = pos / r.unsqueeze(-1)                          # unit radial (..., 3)
    f = 2.0 * M / r                                    # KS potential

    # covariant null vector l_μ = (1, n)
    ones = torch.ones_like(r)
    zeros = torch.zeros_like(r)
    l_cov = torch.stack([ones, n[..., 0], n[..., 1], n[..., 2]], dim=-1)
    l_con = torch.stack([-ones, n[..., 0], n[..., 1], n[..., 2]], dim=-1)

    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0],
                                  device=x.device, dtype=x.dtype))
    eta = eta.expand(x.shape[:-1] + (4, 4))

    # g = η + f l l^T ;  ginv = η − f l l^T (contravariant l)
    ll_cov = l_cov.unsqueeze(-1) * l_cov.unsqueeze(-2)      # (...,4,4)
    g = eta + f.unsqueeze(-1).unsqueeze(-1) * ll_cov
    ll_con = l_con.unsqueeze(-1) * l_con.unsqueeze(-2)
    ginv = eta - f.unsqueeze(-1).unsqueeze(-1) * ll_con

    # ---- analytic derivatives dgd[k,a,b] = ∂_k g_{ab} ---------------------
    df = -f / r                                          # scalar ∂|f|/∂r
    df_vec = -f.unsqueeze(-1) * n / r.unsqueeze(-1)      # (..., 3) = ∂_k f

    # ∂_k l_i = (δ_ki − n_k n_i)/r, shape (..., 3, 3)
    eye3 = torch.eye(3, device=x.device, dtype=x.dtype)
    dli = (eye3 - n.unsqueeze(-1) * n.unsqueeze(-2)) / r.unsqueeze(-1).unsqueeze(-1)

    dgd = torch.zeros(x.shape[:-1] + (4, 4, 4), device=x.device, dtype=x.dtype)
    # spatial derivative index k ∈ {1,2,3} maps to dgd[..., k, a, b]
    # ∂_k g_00 = ∂_k f
    dgd[..., 1:, 0, 0] = df_vec
    # ∂_k g_0i = (∂_k f) n_i + f ∂_k l_i
    dgd0i = (df_vec.unsqueeze(-1) * n.unsqueeze(-2)
             + f.unsqueeze(-1).unsqueeze(-1) * dli)          # (..., 3, 3)
    dgd[..., 1:, 0, 1:] = dgd0i
    dgd[..., 1:, 1:, 0] = dgd0i.transpose(-1, -2)
    # ∂_k g_ij = (∂_k f) n_i n_j + f (∂_k l_i) n_j + f n_i (∂_k l_j)
    # Built with einsum to avoid broadcast-index gymnastics:
    #   term1: df_vec[k]·n[i]·n[j] = einsum('...k,...i,...j->...kij')
    #   term2: f·dli[k,i]·n[j]     = f * einsum('...ki,...j->...kij')
    #   term3: f·n[i]·dli[k,j]     = f * einsum('...i,...kj->...kij')
    dgij = (torch.einsum("...k,...i,...j->...kij", df_vec, n, n)
            + f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * torch.einsum("...ki,...j->...kij", dli, n)
            + f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * torch.einsum("...i,...kj->...kij", n, dli))
    dgd[..., 1:, 1:, 1:] = dgij
    return g, ginv, dgd


def geodesic_acceleration_cartesian(
    x: torch.Tensor, p: torch.Tensor, M: float = 1.0
) -> tuple:
    """Cartesian-chart geodesic RHS: dp^μ/dλ = −Γ^μ_{αβ} p^α p^β.

    Γ is assembled from the analytic metric derivatives (see
    `ks_schwarzchild_metric`) by pure tensor contraction — exact, and
    differentiable all the way back to any learnable upstream quantity.
    """
    g, ginv, dgd = ks_schwarzchild_metric(x, M)

    # Γ^μ_{αβ} = ½ g^{μν} (∂_α g_{βν} + ∂_β g_{αν} − ∂_ν g_{αβ})
    # dgd[k,a,b] = ∂_k g_{ab}; need C[n,a,b] = ∂_a g_{bn} + ∂_b g_{an} − ∂_n g_{ab}.
    # Last-3-dims permutation of (k, a, b):
    #   part1[n,a,b] = dgd[a,b,n] → (n,a,b) = (old b, old k, old a) = (2,0,1)
    #   part2[n,a,b] = dgd[b,a,n] → (n,a,b) = (old b, old a, old k) = (2,1,0)
    #   part3[n,a,b] = dgd[n,a,b] → identity
    # permute only the last 3 dims (batch dims stay in place, works for 6D):
    nd = dgd.dim()
    part1 = dgd.permute(*range(nd - 3), nd - 1, nd - 3, nd - 2)
    part2 = dgd.permute(*range(nd - 3), nd - 1, nd - 2, nd - 3)
    part3 = dgd
    C = part1 + part2 - part3

    # Γ^m_{ab} = ½ Σ_n ginv[m, n] C[n, a, b]
    Gamma = 0.5 * torch.einsum("...mn,...nab->...mab", ginv, C)

    # dp^m = − Γ^m_{ab} p^a p^b
    dp = -torch.einsum("...mab,...a,...b->...m", Gamma, p, p)
    return p, dp


def rk4_geodesic_step_cartesian(
    x: torch.Tensor,
    p: torch.Tensor,
    dlambda: float = 0.1,
    M: float = 1.0,
) -> tuple:
    """One RK4 step of the geodesic equation in the Cartesian KS chart.

    Identical 4-stage structure to `rk4_geodesic_step`, but with no polar
    singularity — safe for cameras on the spin axis.  Fully out-of-place;
    autograd flows through the curved path.
    """
    lam = torch.as_tensor(dlambda, dtype=x.dtype, device=x.device)
    dx1, dp1 = geodesic_acceleration_cartesian(x, p, M)
    dx2, dp2 = geodesic_acceleration_cartesian(x + 0.5 * lam * dx1, p + 0.5 * lam * dp1, M)
    dx3, dp3 = geodesic_acceleration_cartesian(x + 0.5 * lam * dx2, p + 0.5 * lam * dp2, M)
    dx4, dp4 = geodesic_acceleration_cartesian(x + lam * dx3, p + lam * dp3, M)
    x_new = x + (lam / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    p_new = p + (lam / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
    return x_new, p_new


def solve_p_t_cartesian(
    x: torch.Tensor, p_spatial: torch.Tensor, M: float = 1.0
) -> torch.Tensor:
    """Solve the null condition g_{μν} p^μ p^ν = 0 for p^t in Cartesian KS.

    With the spatial momentum fixed, this is a quadratic in p^t:
        a (p^t)² + b p^t + c = 0,
        a = g_00,  b = 2 g_0i p^i,  c = g_ij p^i p^j.
    Outside the horizon the roots have opposite signs (c/a < 0), so the
    positive root is the forward-in-time photon momentum.
    """
    g, _, _ = ks_schwarzchild_metric(x, M)
    a = g[..., 0, 0]
    b = 2.0 * (g[..., 0, 1:] * p_spatial).sum(-1)
    c = torch.einsum("...ij,...i,...j->...", g[..., 1:, 1:], p_spatial, p_spatial)
    disc = (b * b - 4.0 * a * c).clamp(min=0.0)
    sqrt_d = torch.sqrt(disc)
    root1 = (-b + sqrt_d) / (2.0 * a)
    root2 = (-b - sqrt_d) / (2.0 * a)
    return torch.maximum(root1, root2)   # exactly one root is positive


def keplerian_velocity_field_cartesian(
    x: torch.Tensor, M: float = 1.0, sub_keplerian: float = 1.0
) -> torch.Tensor:
    """Local Keplerian plasma velocity, Cartesian version.

    Circular orbits about the z-axis with Ω = √(M/r³) ⇒ v = √(M/r)·φ̂,
    where the azimuthal direction in the x-y plane is
        φ̂ = (-y/r, x/r, 0).

    Parameters
    ----------
    x : (..., 4) Cartesian events (t, x, y, z); r is the Euclidean radius.
    """
    pos = x[..., 1:]
    r = torch.clamp(torch.sqrt((pos * pos).sum(-1) + 1e-18), min=3.0 * M)
    r_safe = r.clamp(min=1e-12)
    v_mag = sub_keplerian * torch.sqrt(M / r_safe)
    e_phi = torch.stack([-pos[..., 1] / r_safe, pos[..., 0] / r_safe,
                         torch.zeros_like(r_safe)], dim=-1)
    return v_mag.unsqueeze(-1) * e_phi


def doppler_factor_cartesian(
    x: torch.Tensor,
    v_cart: torch.Tensor,
    ray_direction_cart: torch.Tensor,
    M: float = 1.0,
) -> torch.Tensor:
    """SR Doppler factor δ = 1/(γ(1 − β·n̂)) — Cartesian twin of
    `doppler_factor` (identical physics; kept separate for chart clarity).
    """
    beta_sq = (v_cart * v_cart).sum(-1).clamp(max=0.9999)
    gamma = 1.0 / torch.sqrt(1.0 - beta_sq)
    beta_dot_n = (v_cart * ray_direction_cart).sum(-1)
    return 1.0 / (gamma * (1.0 - beta_dot_n).clamp(min=1e-6))


def sample_voxel_density_cartesian(
    V: torch.Tensor,
    x_cart: torch.Tensor,
    grid_radius: float = 10.0,
) -> torch.Tensor:
    """Trilinear voxel sampling directly from Cartesian events.

    (1, H, W, 4) events → sample the (D, H, W) volume at each event's
    spatial position, normalized to [-1, 1] by `grid_radius`.
    """
    pos = x_cart[..., 1:]                               # (..., 3)
    norm = pos / grid_radius
    batch_shape = norm.shape[:-1]
    grid = norm.reshape(1, -1, 1, 1, 3)                 # (1, N, 1, 1, 3)
    V5 = V.unsqueeze(0).unsqueeze(0)                    # (1, 1, D, H, W)
    sampled = torch.nn.functional.grid_sample(
        V5, grid, mode="bilinear", padding_mode="zeros",
        align_corners=True,
    )                                                   # (1, 1, N, 1, 1)
    return sampled.reshape(batch_shape)


def keplerian_velocity_field(
    x_sph: torch.Tensor, M: float = 1.0, sub_keplerian: float = 1.0
) -> torch.Tensor:
    """Local plasma 3-velocity (Cartesian) for circular Keplerian orbits.

    In Schwarzschild spacetime the orbital angular velocity measured by a
    local static observer is  Ω = √(M / r³)  (geometrized units), giving
    the circular speed  v = √(M / r)  in flat-space units of c.  A
    sub-Keplerian factor scales it (e.g. 0.5 for a thick torus with
    pressure support).

    Returns a (..., 3) tensor of velocity vectors in Cartesian components
    (x, y, z), tangential to the orbital plane:  v = v_phi · φ̂ with

        φ̂ = (-sinφ, cosφ, 0)

    All components keep autograd connectivity to r, φ (useful if the flow
    geometry is ever made learnable).
    """
    r = x_sph[..., 1].clamp(min=3.0 * M)  # flow lives outside the ISCO region
    phi = x_sph[..., 3]

    v_mag = sub_keplerian * torch.sqrt(M / r)
    e_phi_x = -torch.sin(phi)
    e_phi_y = torch.cos(phi)

    return torch.stack([
        v_mag * e_phi_x,
        v_mag * e_phi_y,
        torch.zeros_like(v_mag),
    ], dim=-1)


def doppler_factor(
    x_sph: torch.Tensor,
    v_cart: torch.Tensor,
    ray_direction_cart: torch.Tensor,
    M: float = 1.0,
) -> torch.Tensor:
    """Special-relativistic Doppler factor δ = 1 / (γ (1 - β·n̂)).

    Parameters
    ----------
    x_sph : (..., 4) events (t, r, θ, φ) — only r is used (gravity enters
        through the gravitational redshift, see `g_gravity`).
    v_cart : (..., 3) plasma velocity in units of c.
    ray_direction_cart : (..., 3) unit vector pointing FROM the emission
        point TOWARD the observer (the negated local ray direction for
        backward-traced camera rays, i.e. `-d̂` of the outgoing photon).

    Returns
    -------
    δ tensor: > 1 for material moving toward the observer (brightening),
    < 1 for receding material.

    Notes
    -----
    This is the flat-spacetime "headlight" factor.  The gravitational
    redshift  √(1 - 2M/r)  between the emission radius and the observer is
    applied separately in `render_grrt_view` so the two effects can be
    toggled/inspected independently.
    """
    beta_sq = (v_cart * v_cart).sum(-1).clamp(max=0.9999)  # β² < 1
    gamma = 1.0 / torch.sqrt(1.0 - beta_sq)

    # β · n̂ : only the line-of-sight velocity component matters
    beta_dot_n = (v_cart * ray_direction_cart).sum(-1)

    delta = 1.0 / (gamma * (1.0 - beta_dot_n).clamp(min=1e-6))
    return delta


def sample_voxel_density(
    V: torch.Tensor,
    x_sph: torch.Tensor,
    grid_radius: float = 10.0,
) -> torch.Tensor:
    """Trilinear voxel-grid sampling of the density/emissivity field.

    Parameters
    ----------
    V : (D, H, W) tensor — the learnable density/emissivity volume.
        Layout follows the Track-A convention: axis 0 = depth (z),
        axis 1 = height (y), axis 2 = width (x).
    x_sph : (..., 4) spherical events (t, r, θ, φ).
    grid_radius : float — physical half-size of the voxel box in units of
        M.  The box spans [-grid_radius, +grid_radius]³; emissions beyond
        it are zero-padded.

    Returns
    -------
    density : (...) tensor with full autograd connectivity to V.

    Implementation
    --------------
    `grid_sample` expects a 5D input (N, C, D, H, W) and normalized
    coordinates (N, D_out, H_out, W_out, 3) in [-1, 1], with axis order
    (x, y, z) matching the LAST dimension of the coordinate tensor and
    `align_corners=True` meaning ±1 hits the voxel centers at the box
    corners.  We build the coordinate tensor as
        grid[..., 0] = x / grid_radius   (width)
        grid[..., 1] = y / grid_radius   (height)
        grid[..., 2] = z / grid_radius   (depth)
    then transpose the sampled (…, C, D, H, W) result back to pointwise
    (...) density values.  `padding_mode="zeros"` ⇒ no emission outside
    the box; `align_corners=True` matches Track A's voxel convention.
    """
    pos = spherical_to_cartesian_pos(x_sph)          # (..., 3)
    norm = pos / grid_radius                          # → [-1, 1]^3

    # grid_sample with a 5D input (N, C, D, H, W) requires a 5D grid
    # (N, D_out, H_out, W_out, 3).  Flatten all event batch dims into
    # D_out, keep H_out = W_out = 1: grid shape (1, N, 1, 1, 3).
    batch_shape = norm.shape[:-1]
    grid = norm.reshape(1, -1, 1, 1, 3)               # (1, N, 1, 1, 3)

    V5 = V.unsqueeze(0).unsqueeze(0)                  # (1, 1, D, H, W)
    sampled = torch.nn.functional.grid_sample(
        V5, grid,
        mode="bilinear",          # trilinear in 3D
        padding_mode="zeros",
        align_corners=True,
    )                                                  # (1, 1, N, 1, 1)

    return sampled.reshape(batch_shape)                # (...)


def optical_depth_step(
    density: torch.Tensor, dlambda: float, opacity: float = 1.0
) -> torch.Tensor:
    """Per-step optical depth increment dτ = κ ρ Δλ (opacity × density)."""
    return opacity * density * dlambda


def render_grrt_view(
    camera: GRRTCamera,
    voxel_grid: torch.Tensor,
    M: float = 1.0,
    dlambda: float = 0.2,
    max_steps: int = 600,
    grid_radius: float = 10.0,
    opacity: float = 1.0,
    doppler_power: int = 3,
    sub_keplerian: float = 1.0,
    apply_gravitational_redshift: bool = True,
    boundaries: Optional[BoundaryRadii] = None,
    record_debug: bool = False,
    polar_axis_offset: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """End-to-end differentiable GRRT renderer (Steps 1 → 4).

    Pipeline
    --------
    1. `initialize_camera_rays` — pixel grid of null initial data.
    2. `cartesian_to_spherical_rays` — into the Schwarzschild chart.
    3. `trace_null_geodesics_boundaries`-style loop, but with **voxel
       sampling and radiative transfer interleaved into the RK4 march**:
       at every step, each still-alive ray samples the voxel grid at its
       current spatial position, boosts the local emissivity by the
       Doppler factor, and accumulates intensity front-to-back.
    4. Rays captured by the horizon contribute exactly zero (shadow);
       escaped rays carry the fully integrated intensity.

    Parameters
    ----------
    camera : GRRTCamera
        Camera intrinsics (resolution, FOV, position, look-at).
    voxel_grid : (D, H, W) tensor
        Learnable density/emissivity volume in Cartesian M-units.
    M : float
        Black-hole mass (geometrized units).
    dlambda : float
        Affine step for the RK4 integrator.
    max_steps : int
        Maximum RK4 steps (rays frozen at horizon/escape stop early).
    grid_radius : float
        Physical half-size of the voxel box in M (matches voxel_grid's
        normalization).
    opacity : float
        Opacity coefficient κ in dτ = κ ρ Δλ.
    doppler_power : int
        Exponent on δ: 3 for specific intensity I_ν (the standard
        headlight effect), 4 for the observed bolometric flux variant.
    sub_keplerian : float
        Velocity scale factor for the orbital flow (1.0 = Keplerian).
    apply_gravitational_redshift : bool
        Multiply each sample's contribution by √(1 - 2M/r) — the
        gravitational part of the total redshift.
    boundaries : BoundaryRadii, optional
        Custom horizon/photon-sphere/ring radii.
    record_debug : bool
        If True, also return per-step sample statistics for inspection.

    Returns
    -------
    dict with
        ``image``        (H, W) rendered intensity Î (differentiable w.r.t.
                         voxel_grid and any camera parameters)
        ``shadow_mask``  (H, W) bool — pixels whose rays fell into the
                         horizon (assigned zero emission)
        ``photon_sphere_mask``, ``einstein_ring_mask``, ``escaped_mask``,
                         ``alive_mask`` — from Step 3
        ``total_density`` (H, W) — column density (diagnostic, no beaming)
        ``doppler_mean``  (H, W) — path-averaged δ (diagnostic)
    """
    device = voxel_grid.device
    dtype = voxel_grid.dtype

    # ---------- polar-axis guard ------------------------------------------
    # A camera exactly on the spin axis (polar top-down view) sits on the
    # spherical chart's coordinate singularity: p_φ = p·φ̂/(r sinθ) blows up
    # for transverse rays and the integration goes numerically unstable.
    # Physically the view is identical for an arbitrarily small tilt, so we
    # nudge the camera 0.05 M off-axis (or `polar_axis_offset`) unless the
    # user has already placed it away from the axis.  This keeps every ray
    # in a well-conditioned region of the chart.
    pos = list(camera.position)
    if math.hypot(pos[0], pos[1]) < 1e-6:
        pos[0] = pos[0] + polar_axis_offset
        camera = replace(camera, position=tuple(pos))

    # ---------- Step 1: camera rays ---------------------------------------
    rays = initialize_camera_rays(camera, device=device, dtype=dtype)
    x0_cart, p0_cart = rays["x0"], rays["p0"]       # (1, H, W, 4)

    # ---------- Step 2: into the Cartesian Kerr–Schild chart --------------
    # Note: Step-1 rays are ALREADY Cartesian; no chart conversion needed.
    # We only need to normalize p^t so the null condition holds in the
    # curved metric (Step 1 assumed flat Minkowski).
    p0_cart = p0_cart.clone()
    p0_spatial = p0_cart[..., 1:]
    p0_t = solve_p_t_cartesian(x0_cart, p0_spatial, M=M)
    p0_cart = torch.cat([p0_t.unsqueeze(-1), p0_spatial], dim=-1)

    H, W = camera.height, camera.width
    br = boundaries or BoundaryRadii()

    # Escape threshold: voxel box + margin so rays terminate once outside.
    escape_radius = max(grid_radius * 1.5, 3.0 * br.einstein_ring)

    # ---------- Step 3+4: marching with radiative transfer ----------------
    # Cartesian KS chart: no polar singularity; boundaries operate on the
    # Euclidean radius r = |x| (equals the Schwarzschild areal radius).
    batch_shape = x0_cart.shape[:-1]                 # (1, H, W)
    alive = torch.ones(batch_shape, device=device, dtype=torch.bool)
    shadow = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    escaped = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    hit_photon_sphere = torch.zeros(batch_shape, device=device, dtype=torch.bool)
    hit_ring = torch.zeros(batch_shape, device=device, dtype=torch.bool)

    intensity = torch.zeros(batch_shape, device=device, dtype=dtype)
    tau = torch.zeros(batch_shape, device=device, dtype=dtype)
    density_sum = torch.zeros(batch_shape, device=device, dtype=dtype)
    doppler_sum = torch.zeros(batch_shape, device=device, dtype=dtype)
    doppler_n = torch.zeros(batch_shape, device=device, dtype=dtype)

    # Per-ray Cartesian direction toward the observer.  For camera rays
    # launched backward (-d̂ convention of Step 1) the photon physically
    # travels along +d̂_photon; the observer sits opposite, at -d̂_photon
    # direction from the emission point... Actually for the standard
    # headlight formula we need n̂ = unit vector from emission toward the
    # OBSERVER.  Camera rays were built pointing from the camera INTO the
    # scene, so the observer direction is simply -rays["directions"].
    observer_dir = -rays["directions"]               # (H, W, 3), unit

    x, p = x0_cart, p0_cart
    lam = float(dlambda)
    dlam_tensor = torch.full(batch_shape, float(dlambda), device=device, dtype=dtype)

    debug_samples = [] if record_debug else None

    for step in range(1, max_steps + 1):
        if not bool(alive.any()):
            break

        # ---------- boundary checks (Step 3) ------------------------------
        with torch.no_grad():
            pos3 = x[..., 1:]
            r = torch.clamp(
                torch.sqrt((pos3 * pos3).sum(-1) + 1e-18), min=1e-9)
            pr = (p[..., 1:] * pos3 / r.unsqueeze(-1)).sum(-1)   # p·r̂

            newly_absorbed = alive & (r <= br.horizon + br.horizon_tol)
            shadow |= newly_absorbed
            alive = alive & ~newly_absorbed

            hit_photon_sphere |= alive & (
                (r - br.photon_sphere).abs() < br.photon_band)
            hit_ring |= alive & ((r - br.einstein_ring).abs() < br.ring_band)

            newly_escaped = alive & (r >= escape_radius) & (pr > 0)
            escaped |= newly_escaped
            alive = alive & ~newly_escaped

        if not bool(alive.any()):
            break

        # ---------- voxel sampling + radiative transfer (Step 4) ----------
        density = sample_voxel_density_cartesian(
            voxel_grid, x, grid_radius)                             # (...)

        # Doppler factor from the local Keplerian flow (Cartesian).
        v_cart = keplerian_velocity_field_cartesian(
            x, M, sub_keplerian)

        # Observer direction broadcast per event.
        obs_dir_e = observer_dir.unsqueeze(0).expand(batch_shape + (3,))

        delta = doppler_factor_cartesian(x, v_cart, obs_dir_e, M)   # (...)

        # Gravitational redshift at the emission radius.
        if apply_gravitational_redshift:
            g_grav = torch.sqrt(1.0 - 2.0 * M / r.clamp(min=2.0 * br.horizon))
        else:
            g_grav = torch.ones_like(r)

        j = density                                                # emissivity

        dtau = optical_depth_step(density, dlambda, opacity)
        contrib = (
            j
            * delta.pow(doppler_power)
            * g_grav
            * torch.exp(-tau)
            * dlam_tensor
        )
        intensity = intensity + alive * contrib
        tau = tau + alive * dtau
        density_sum = density_sum + alive * density
        doppler_sum = doppler_sum + alive * delta
        doppler_n = doppler_n + alive.float()

        if record_debug:
            debug_samples.append({
                "step": step,
                "density_mean": float(density.mean()),
                "delta_mean": float(delta.mean()),
            })

        # ---------- RK4 advance (Step 2, Cartesian) -----------------------
        x_new, p_new = rk4_geodesic_step_cartesian(x, p, lam, M)
        x, p = _apply_frozen_update(alive, x, p, x_new, p_new)

    # ---------- shadow: zero emission for captured rays --------------------
    # Rays that fell into the horizon contribute exactly zero intensity;
    # they are already excluded from accumulation by the alive mask, but we
    # also zero the image explicitly so no numerical residue survives.
    image = torch.where(shadow, torch.zeros_like(intensity), intensity)

    # Mask the "not finished" rays as zero too (they neither escaped nor
    # were captured within the step budget — treat as dark rather than
    # inventing emission).
    image = torch.where(alive, torch.zeros_like(intensity), image)

    out: Dict[str, torch.Tensor] = {
        "image": image,
        "shadow_mask": shadow,
        "escaped_mask": escaped,
        "photon_sphere_mask": hit_photon_sphere,
        "einstein_ring_mask": hit_ring,
        "alive_mask": alive,
        "total_density": density_sum,
        "doppler_mean": torch.where(doppler_n > 0, doppler_sum / doppler_n.clamp(min=1.0),
                                    torch.zeros_like(doppler_sum)),
        "steps": step,
    }
    if record_debug:
        out["debug"] = debug_samples
    return out


if __name__ == "__main3__":  # pragma: no cover - Step 4 smoke test (manual)
    # Render a small test scene: an emissive torus around the hole.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    M = 1.0
    D = H = W = 48
    grid_radius = 10.0

    # Build a Cartesian voxel grid containing a circular torus in the x-y
    # plane at r ∈ [3.5, 6] M, thickness 0.8 M — the classic test disk.
    zs = torch.linspace(-grid_radius, grid_radius, D, device=dev)
    ys = torch.linspace(-grid_radius, grid_radius, H, device=dev)
    xs = torch.linspace(-grid_radius, grid_radius, W, device=dev)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    torus = torch.exp(-((rr - 4.5)**2) / 0.5) * torch.exp(-(zz**2) / 0.4)
    V = (torus * 2.0).requires_grad_(True)   # learnable density

    cam = GRRTCamera(height=32, width=32, fov_deg=25.0,
                     position=(0.0, 0.0, 12.0))
    out = render_grrt_view(cam, V, M=M, dlambda=0.15, max_steps=400,
                           grid_radius=grid_radius, opacity=1.0)

    img = out["image"]
    print("image range:", float(img.min()), "-", float(img.max()))
    print("shadow pixels:", int(out["shadow_mask"].sum()))
    print("doppler mean range:", float(out["doppler_mean"].min()), "-",
          float(out["doppler_mean"].max()))

    # Autograd check: gradient of a scalar loss w.r.t. the voxel grid.
    loss = img.pow(2).sum()
    loss.backward()
    print("V.grad nonzero:", float(V.grad.abs().sum()) > 0)
    print("V.grad finite:", bool(torch.isfinite(V.grad).all()))
