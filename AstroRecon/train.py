import torch
from src.ray_marcher import continuous_ray_march
from src.losses import compute_total_loss

# 1. Create continuous 3D density grid (64x64x64 voxels)
voxel_grid = torch.rand((64, 64, 64), requires_grad=True)

# 2. Simulate observed target FITS projection
observed_fits = torch.ones((64, 64))

# 3. Render 2D continuous line integral
rendered_img = continuous_ray_march(voxel_grid)

# 4. Compute multi-term reprojection loss
loss = compute_total_loss(rendered_img, observed_fits, voxel_grid)

# 5. Backpropagate gradients
loss.backward()

print("Autograd Verification Success!")
print("Grid Gradient Tensor Shape:", voxel_grid.grad.shape)