import torch

from src.ray_marcher import VolumetricRayMarcher, CameraRig, render_volume
from src.hardware import get_manager, HardwareManager, gpu_present

# --- ray marcher: forward shape, differantiability through X,Y,Z ----------
torch.manual_seed(0)
m = VolumetricRayMarcher((8, 16, 20), (16, 20), n_steps=24)
vol = torch.rand(8, 16, 20, requires_grad=True)
img = m.render(vol)
assert img.shape == (16, 20), img.shape
assert torch.isfinite(img).all(), "nonfinite render"

img.sum().backward()
assert vol.grad is not None and torch.isfinite(vol.grad).all()
for dim, size in enumerate(vol.shape):
    grad_norm = vol.grad.pow(1 if dim != 0 else 1).sum(dim=tuple(i for i in range(3) if i != dim))
    assert torch.count_nonzero(torch.abs(grad_norm) > 0) >= max(1, size // 4), (
        f"dim {dim} shows too few unmasked voxels with gradient"
    )
print("backward pass: gradients reach all three spatial dims OK")

# --- different yaw rigs must produce different projections ----------------
v = torch.rand(8, 16, 20)
img0 = render_volume(v, (16, 20), yaw_deg=0.0)
img40 = render_volume(v, (16, 20), yaw_deg=40.0)
img40b = render_volume(v, (16, 20), yaw_deg=40.0)
assert torch.allclose(img40b, img40, atol=1e-6), "caching must not leak between rigs"
assert (img0 - img40).abs().max() > 1e-4, "yaw rotation had zero effect"
print("rig rotation changes the projection OK")

# --- extinction sanity: heavier medium attenuates the far side ------------
slab = torch.zeros(8, 16, 20)
slab[6, 8, 10] = 1.0  # a dense blob deep in Z
img_near = render_volume(slab, (16, 20), sigma=0.1)
img_far = render_volume(slab, (16, 20), sigma=4.0)  # strong extinction
assert img_far.abs().max() <= img_near.abs().max() + 1e-6
print("extinction attenuates deep emission OK")

# --- hardware manager strictness ------------------------------------------
mgr = HardwareManager(strict_gpu=True)
dev = mgr.device()
dev_names = [s.identifier for s in mgr.devices]
req = None if gpu_present() else "cpu"
d = mgr.resolve_requested(req)
assert d == mgr.device()
if gpu_present():
    assert mgr.selected.startswith("cuda"), mgr.selected
    # auto-selection must never silently land on CPU with a GPU present:
    mgr.set_selected("cuda:0")
print(f"hardware manager OK — active device: {mgr.device()} (cuda_available={gpu_present()})")
print(f"inventory: {[(s.identifier, s.label) for s in mgr.devices]}")

# rotated-rig optimizer step sanity: 3D voxel loss decreases
from src.losses import compute_total_loss
target = torch.rand(16, 20)
logits = torch.randn(8, 16, 20, requires_grad=True)
opt = torch.optim.Adam([logits], lr=0.05)
first = None
for _ in range(12):
    opt.zero_grad()
    volume = torch.sigmoid(logits)
    rho = render_volume(volume, (16, 20), n_steps=16)
    loss = compute_total_loss(rho, target, volume)
    loss.backward(); opt.step()
    first = float(loss) if first is None else first
assert float(loss) < first, "3D tomography loss did not decrease"
print(f"3D optimization loop reduces loss OK ({first:.4f} -> {float(loss):.4f})")
print("ALL CHECKS PASSED")
