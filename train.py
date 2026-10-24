import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.io_utils import load_fits, make_synthetic_observation, save_vtk_volume
from src.losses import compute_total_loss
from src.ray_marcher import continuous_ray_march
from src.visualizer import save_comparison


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
	parser = argparse.ArgumentParser(
		description="Reconstruct a 3D density volume from a 2D FITS observation."
	)
	parser.add_argument("--input", type=Path, help="Optional 2D FITS observation")
	parser.add_argument("--size", type=int, default=64, help="Volume edge length")
	parser.add_argument("--epochs", type=int, default=50, help="Optimization steps")
	parser.add_argument("--learning-rate", type=float, default=0.05)
	parser.add_argument("--tv-weight", type=float, default=1e-3)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	return parser.parse_args()


def main():
	args = parse_args()
	if args.size < 2 or args.epochs < 1 or args.learning_rate <= 0:
		raise ValueError("size must be >= 2, epochs >= 1, and learning-rate > 0")

	device = torch.device(args.device)
	torch.manual_seed(42)

	if args.input:
		observed = load_fits(args.input, device=device)
	else:
		observed = make_synthetic_observation(args.size, device=device)
		synthetic_path = PROJECT_ROOT / "data" / "synthetic" / "baseline.npy"
		synthetic_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(synthetic_path, observed.detach().cpu().numpy())
		print(f"Using synthetic observation: {synthetic_path}")

	observed = F.interpolate(
		observed[None, None],
		size=(args.size, args.size),
		mode="bilinear",
		align_corners=False,
	)[0, 0]

	raw_density = torch.nn.Parameter(
		torch.full((args.size, args.size, args.size), -2.5, device=device)
		+ 0.05 * torch.randn((args.size, args.size, args.size), device=device)
	)
	optimizer = torch.optim.Adam([raw_density], lr=args.learning_rate)
	initial_loss = None

	for epoch in range(1, args.epochs + 1):
		optimizer.zero_grad(set_to_none=True)
		density = F.softplus(raw_density)
		rendered = continuous_ray_march(density)
		loss = compute_total_loss(
			rendered,
			observed,
			density,
			tv_weight=args.tv_weight,
		)
		loss.backward()
		optimizer.step()
		if initial_loss is None:
			initial_loss = float(loss.detach())

	with torch.no_grad():
		density = F.softplus(raw_density)
		rendered = continuous_ray_march(density)
		final_loss = float(
			compute_total_loss(
				rendered,
				observed,
				density,
				tv_weight=args.tv_weight,
			)
		)

	render_path = PROJECT_ROOT / "outputs" / "renders" / "reconstruction.png"
	volume_path = PROJECT_ROOT / "outputs" / "volumes" / "density.vtk"
	save_comparison(observed, rendered, render_path)
	save_vtk_volume(density, volume_path)

	print(f"Training complete on {device}: loss {initial_loss:.6f} -> {final_loss:.6f}")
	print(f"Render: {render_path}")
	print(f"Volume: {volume_path}")


if __name__ == "__main__":
	main()
