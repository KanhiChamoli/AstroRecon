import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Your existing imports below:
import torch
import numpy as np
# ... rest of your train.py code ...
import argparse
import json
import logging
from pathlib import Path
import time
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from src.log_config import setup_logging
setup_logging()
logger = logging.getLogger("astrorecon.train")

import numpy as np
import torch

from src.fits_router import classify_fits
from src.hardware import get_manager
from src.io_utils import (
	load_fits,
	load_fits_cube,
	make_synthetic_observation,
	save_fits_cube,
	save_vtk_volume,
)
from src.losses import compute_total_loss
from src.ray_marcher import VolumetricRayMarcher, CameraRig
from src.visualizer import save_comparison, save_cube_comparison, save_image
from src.render_viewer import show_render_gallery
from src.uvfits_imaging import image_uvfits
from src.remote_streamer import RemoteObservationStreamer
from src.simulation_engine import launch_engine_app

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
	parser = argparse.ArgumentParser(
		description="Identify FITS product types and route to the matching processing pipeline."
	)
	input_group = parser.add_mutually_exclusive_group()
	input_group.add_argument("--input", type=Path, help="FITS image, cube, table, or UVFITS visibility file (type detected automatically)")
	input_group.add_argument("--input-cube", type=Path, help="Alias for supplying a 3D FITS cube")
	input_group.add_argument("--stream", type=str, help="Stream target from NASA SkyView directly into memory (e.g. 'Sagittarius A*')")
	input_group.add_argument("--stream-url", type=str, help="Stream remote FITS file over HTTP directly into memory")
	parser.add_argument("--stream-survey", type=str, default="NVSS", help="NASA SkyView survey name (e.g. 'NVSS', 'INTEGRAL/SPI GC', '2MASS-K')")
	parser.add_argument("--input2", type=Path, help="Optional second UVFITS band from the same source and observing day")
	parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
	parser.add_argument("--size", type=int, default=64, help="Synthetic image size / maximum UVFITS reconstruction size")
	parser.add_argument("--epochs", type=int, default=800, help="TV-regularized image optimization steps")
	parser.add_argument("--depth", type=int, default=64, help="Z resolution of the true 3D voxel volume")
	parser.add_argument("--n-steps", type=int, default=64, help="Ray integration samples per pixel")
	parser.add_argument("--sigma", type=float, default=1.0, help="Dust extinction coefficient along rays")
	parser.add_argument("--density-weight", type=float, default=0.00001, help="L1 density penalty weight")
	parser.add_argument("--yaw-deg", type=float, default=0.0, help="Camera rig yaw for the known viewing geometry")
	parser.add_argument("--pitch-deg", type=float, default=0.0, help="Camera rig pitch for the known viewing geometry")
	parser.add_argument("--learning-rate", type=float, default=0.03)
	parser.add_argument("--tv-weight", type=float, default=0.015)
	parser.add_argument("--denoise-sigma", type=float, default=1.25, help="Gaussian denoise radius in pixels")
	parser.add_argument("--averaging-seconds", type=float, default=60.0, help="UVFITS coherent visibility averaging interval")
	parser.add_argument("--parameter-set", type=int, default=0, help="Row index in the official day-specific EHT parameter survey")
	parser.add_argument("--major-iterations", type=int, default=5, help="EHT blur-and-refine passes")
	parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Open the generated renders in a separate window (use --no-show for batch runs)")
	parser.add_argument("--live-engine", action=argparse.BooleanOptionalAction, default=True, help="Display live optimization in an interactive windowed simulation engine application")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()
	if args.epochs < 1 or args.depth < 1 or args.learning_rate <= 0 or args.denoise_sigma < 0:
		raise ValueError("epochs, depth, and learning-rate must be positive; denoise-sigma cannot be negative")
	if args.tv_weight < 0 or args.sigma < 0:
		raise ValueError("tv-weight and sigma must be nonnegative")

	# Strict GPU enforcement: automatic selection locks onto the primary CUDA
	# GPU; CPU runs require explicit opt-in and are logged loudly.
	hardware = get_manager(strict_gpu=True)
	device = hardware.resolve_requested(args.device)
	if args.device and args.device.startswith("cuda") and device.type == "cpu":
		raise ValueError("CUDA device requested but execution resolved to CPU — refusing silent fallback")
	input_path = args.input or args.input_cube
	if args.input2 and not input_path:
		raise ValueError("--input2 requires a primary --input UVFITS file")
	input_product = classify_fits(input_path) if input_path else None
	input_kind = input_product["kind"] if input_product else "synthetic"
	if args.input2 and input_kind != "uvfits_visibility":
		raise ValueError("--input2 is only supported when --input is UVFITS visibility data")
	if input_kind == "uvfits_visibility":
		if args.input2 and classify_fits(args.input2)["kind"] != "uvfits_visibility":
			raise ValueError("--input2 must also be a UVFITS visibility file")
		result = image_uvfits(
			input_path,
			args.output_dir,
			second_input_path=args.input2,
			image_size=min(args.size, 80),
			averaging_seconds=args.averaging_seconds,
			max_iterations=args.epochs,
			major_iterations=args.major_iterations,
			parameter_set=args.parameter_set,
			show=args.show,
		)
		logger.info("EHT image: %s", result['fits_path'])
		logger.info("Render: %s", result['render_path'])
		logger.info("Metadata: %s", result['metadata_path'])
		if args.show:
			show_render_gallery([result["render_path"]])
		return
	if input_kind == "image_cube":
		cube_input_path = input_path
		original_cube, header = load_fits_cube(
			cube_input_path,
			device=device,
			denoise_sigma=0.0,
			hdu_index=input_product["hdu_index"],
		)
		filtered_cube, _ = load_fits_cube(
			cube_input_path,
			device=device,
			denoise_sigma=args.denoise_sigma,
			hdu_index=input_product["hdu_index"],
		)
		filtered_projection = filtered_cube.max(dim=0).values
		output_dir = args.output_dir
		projection_path = output_dir / "renders" / "cube_comparison.png"
		filtered_path = output_dir / "renders" / "filtered_input.png"
		middle_slice_path = output_dir / "renders" / "cube_middle_slice.png"
		cube_path = output_dir / "volumes" / "denoised_cube.fits"
		axis_metadata = {
			"fits_axis_3_type": header.get("CTYPE3"),
			"fits_axis_3_unit": header.get("CUNIT3"),
			"fits_axis_3_length": int(filtered_cube.shape[0]),
			"input_data_shape": list(filtered_cube.shape),
			"processing": ["global robust background subtraction", "asinh stretch", "3D Gaussian denoising"],
			"denoise_sigma_spatial_pixels": args.denoise_sigma,
			"denoise_sigma_axis_3_samples": max(0.25, args.denoise_sigma * 0.5),
			"wcs_note": "Original FITS header is preserved; interpret axis 3 from CTYPE3/CUNIT3.",
		}
		save_cube_comparison(
			original_cube,
			filtered_cube,
			projection_path,
			metadata={
				"data_source": str(cube_input_path),
				"source_kind": "3D FITS cube",
				"processing": axis_metadata["processing"],
				"cube_shape": axis_metadata["input_data_shape"],
				"fits_axis_3_type": axis_metadata["fits_axis_3_type"],
				"fits_axis_3_unit": axis_metadata["fits_axis_3_unit"],
			},
		)
		save_image(
			filtered_projection,
			filtered_path,
			title="Denoised cube max-intensity projection",
			metadata={"data_source": str(cube_input_path), "source_kind": "3D FITS cube max-intensity projection", **axis_metadata},
		)
		save_image(
			filtered_cube[filtered_cube.shape[0] // 2],
			middle_slice_path,
			title="Denoised middle FITS cube slice",
			metadata={"data_source": str(cube_input_path), "source_kind": "3D FITS cube slice", "slice_index_axis_3": filtered_cube.shape[0] // 2, **axis_metadata},
		)
		save_fits_cube(filtered_cube, header, cube_path)
		metadata_path = output_dir / "reconstruction.json"
		metadata_path.write_text(
			json.dumps({
				"input": str(cube_input_path),
				"input_data_shape": list(filtered_cube.shape),
				"fits_axis_3_type": axis_metadata["fits_axis_3_type"],
				"fits_axis_3_unit": axis_metadata["fits_axis_3_unit"],
				"denoise_sigma_spatial_pixels": args.denoise_sigma,
				"denoise_sigma_axis_3_samples": axis_metadata["denoise_sigma_axis_3_samples"],
				"volume_output": str(cube_path),
				"interpretation_note": "A third FITS axis may be wavelength or velocity, not geometric distance. Use its WCS and a physical model before interpreting it as depth.",
			}, indent=2),
			encoding="utf-8",
		)
		logger.info("Denoised measured FITS cube: %s", tuple(filtered_cube.shape))
		logger.info("Cube (WCS preserved): %s", cube_path)
		logger.info("Projection: %s", projection_path)
		logger.info("Middle slice: %s", middle_slice_path)
		if args.show:
			show_render_gallery([projection_path, filtered_path, middle_slice_path])
		return

	if input_kind in {"image_hypercube", "fits_table", "unknown_fits"}:
		raise ValueError(
			f"Detected FITS product type '{input_kind}' in {input_path}. "
			"This pipeline supports 2D image FITS, 3D image cubes, and UVFITS visibility files. "
			"Tables/hypercubes require a product-specific calibration and axis adapter."
		)
	dataset_metadata = {}
	if args.stream:
		logger.info("Streaming target '%s' directly into memory via NASA SkyView...", args.stream)
		original, stream_meta = RemoteObservationStreamer.stream_skyview_with_metadata(
			target=args.stream, survey=args.stream_survey, device=device
		)
		filtered = original.clone()
		dataset_metadata = stream_meta
	elif args.stream_url:
		logger.info("Streaming remote observation from '%s' directly into memory...", args.stream_url)
		original = RemoteObservationStreamer.stream_fits_url(args.stream_url, device=device)
		filtered = original.clone()
		dataset_metadata = getattr(original, "metadata", {})
	elif input_kind == "image2d":
		original = load_fits(
			input_path,
			device=device,
			denoise_sigma=0.0,
			hdu_index=input_product["hdu_index"],
		)
		filtered = load_fits(
			input_path,
			device=device,
			denoise_sigma=args.denoise_sigma,
			hdu_index=input_product["hdu_index"],
		)
		dataset_metadata = {
			"source_name": f"Local FITS: {input_path.name}",
			"target": input_path.stem,
			"source_url": "https://data.astropy.org/",
			"dimensions": f"{filtered.shape[1]} x {filtered.shape[0]} px",
			"processing": f"Gaussian denoise (σ={args.denoise_sigma}), asinh stretch, 2D TV optimization",
		}
	else:
		original = make_synthetic_observation(128, device=device)
		filtered = original.clone()
		baseline_path = PROJECT_ROOT / "data" / "synthetic" / "baseline.npy"
		baseline_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(baseline_path, original.detach().cpu().numpy())
		logger.info("Using synthetic observation: %s", baseline_path)
		dataset_metadata = {
			"source_name": "Synthetic Nebular Observation",
			"target": "Simulated Optically Thin Nebula",
			"source_url": "https://github.com",
			"dimensions": "128 x 128 px",
			"processing": "Multi-component Gaussian density profile with TV regularization",
		}

	# Initialize Live Windowed Simulation Engine if requested
	engine_app = None
	engine_window = None
	if args.live_engine and args.show:
		try:
			engine_app, engine_window = launch_engine_app(metadata=dataset_metadata)
			engine_window.set_target(filtered.detach().cpu().numpy())
			logger.info("Live viewport window opened.")
		except Exception as err:
			logger.warning("Could not initialize live engine GUI: %s", err)

	# True 3D volumetric reconstruction: optimize voxel logits V ∈ ℝ^{D×H×W}
	# directly against the observation through the differentiable ray marcher.
	# A single view leaves depth weakly constrained; the 3D-ness comes from the
	# forward model (rays march through X, Y, Z), never from extrusion.
	height, width = filtered.shape
	depth = args.depth
	initial = filtered.clamp(1e-5, 1.0 - 1e-5)
	layer_noise = torch.randn(depth, height, width, device=device) * 0.01
	voxel_logits = torch.nn.Parameter(
		torch.logit(initial).unsqueeze(0).repeat(depth, 1, 1) + layer_noise
	)
	camera = CameraRig(height=height, width=width, yaw_deg=args.yaw_deg, pitch_deg=args.pitch_deg, n_steps=args.n_steps, sigma=args.sigma)
	marcher = VolumetricRayMarcher((depth, height, width), (height, width), device=device, camera=camera)
	optimizer = torch.optim.Adam([voxel_logits], lr=args.learning_rate)
	hardware.assert_on_gpu(voxel_logits, filtered)
	initial_loss = None
	t0 = time.time()
	update_interval = max(1, args.epochs // 40)

	for epoch in range(1, args.epochs + 1):
		optimizer.zero_grad(set_to_none=True)
		volume = torch.sigmoid(voxel_logits)
		reconstruction = marcher.render(volume)
		loss = compute_total_loss(
			reconstruction,
			filtered,
			volume,
			tv_weight=args.tv_weight,
			density_weight=args.density_weight,
		)
		loss.backward()
		optimizer.step()
		if initial_loss is None:
			initial_loss = float(loss.detach())

		if engine_window is not None and (epoch % update_interval == 0 or epoch == args.epochs):
			elapsed = max(time.time() - t0, 1e-4)
			fps = epoch / elapsed
			engine_window.update_simulation_frame(
				epoch,
				args.epochs,
				reconstruction.detach().cpu().numpy(),
				float(loss.item()),
				fps,
			)
			# Cooperative pause: sleep in the OUTER loop (never inside the Qt
			# update call — that re-entered the event loop and froze the UI).
			while engine_window.is_paused:
				time.sleep(0.05)

	with torch.no_grad():
		volume = torch.sigmoid(voxel_logits)
		reconstruction = marcher.render(volume)
		final_data_loss = float(torch.mean((reconstruction - filtered) ** 2))
		final_smoothness_loss = float(compute_total_loss(
			reconstruction, filtered, volume,
			tv_weight=1.0, density_weight=0.0,
		) - torch.mean((reconstruction - filtered) ** 2))

	output_dir = args.output_dir
	render_path = output_dir / "renders" / "reconstruction.png"
	filtered_path = output_dir / "renders" / "filtered_input.png"
	volume_path = output_dir / "volumes" / "density.vtk"
	metadata_path = output_dir / "reconstruction.json"
	image_metadata = {
		"data_source": str(input_path) if input_path else "synthetic baseline generated by src.io_utils.make_synthetic_observation",
		"source_kind": "2D FITS image" if input_path else "synthetic 2D image",
		"processing": [
			"robust median background subtraction",
			"0.8-sigma noise threshold",
			"99.8th-percentile normalization",
			"asinh contrast stretch",
			"Gaussian denoise",
			"true 3D volumetric ray-marched refinement",
		],
		"image_shape": list(reconstruction.shape),
		"optimization_steps": args.epochs,
		"denoise_sigma_pixels": args.denoise_sigma,
		"tv_weight": args.tv_weight,
		"final_data_mse": final_data_loss,
		"depth_note": "True 3D tracer fit; a single known rig weakly constrains depth — supply multiple calibrated rigs for full tomography.",
		"code_modules": ["train.py", "src/io_utils.py", "src/losses.py", "src/ray_marcher.py", "src/visualizer.py"],
	}
	save_comparison(original, filtered, reconstruction, render_path, metadata=image_metadata)
	save_image(
		reconstruction,
		filtered_path,
		title="Denoised reconstruction",
		metadata=image_metadata,
	)
	save_vtk_volume(volume, volume_path)
	metadata = {
		"input": str(input_path) if input_path else "synthetic",
		"image_shape": list(reconstruction.shape),
		"volume_shape": list(volume.shape),
		"epochs": args.epochs,
		"denoise_sigma": args.denoise_sigma,
		"tv_weight": args.tv_weight,
		"initial_loss": initial_loss,
		"final_data_mse": final_data_loss,
		"final_total_variation": final_smoothness_loss,
		"device": str(device),
		"ray_integration": {
			"n_steps": args.n_steps,
			"sigma": args.sigma,
			"yaw_deg": args.yaw_deg,
			"pitch_deg": args.pitch_deg,
			"sampler": "grid_sample bilinear align_corners=True",
		},
		"volume_note": "Genuine D×H×W voxel fit via volumetric ray integration; no 2D depth extrusion.",
	}
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	logger.info("Reconstruction complete on %s: %d steps, %s voxels", device, args.epochs, tuple(volume.shape))
	logger.info("Image: %s", render_path)
	logger.info("Filtered image: %s", filtered_path)
	logger.info("Volume: %s", volume_path)
	logger.info("Metrics and filter settings: %s", metadata_path)
	logger.info("Data MSE: %.8f; total variation: %.8f", final_data_loss, final_smoothness_loss)
	# Debug: confirm tensors are on GPU if available
	logger.debug("Reconstruction tensor device: %s", reconstruction.device)
	# Keep the GUI alive after training if live engine was launched
	if engine_app is not None:
		logger.info("Training complete – GUI will remain open until closed.")
		engine_app.exec_()
	if args.show:
		show_render_gallery([render_path, filtered_path])


if __name__ == "__main__":
	main()