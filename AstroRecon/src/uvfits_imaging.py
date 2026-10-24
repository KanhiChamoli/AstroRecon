"""EHT UVFITS visibility imaging through the eht-imaging package."""

import json
import importlib.util
from datetime import datetime, timedelta
from importlib.metadata import version
from itertools import groupby
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from src.fits_router import classify_fits
from src.render_metadata import write_render_metadata
from src.render_viewer import show_render_gallery


def _enable_numpy_2_ragged_compatibility(ehtim):
	"""Restore ehtim's documented list-of-arrays tlist behavior on NumPy 1.24+."""
	def time_list(self, conj=False, t_gather=0.0, scan_gather=False):
		data = self.data_conj() if conj else self.data
		groups = []
		if t_gather <= 0.0 and not scan_gather:
			key_function = lambda row: row["time"]
		elif t_gather > 0.0 and not scan_gather:
			width_hours = t_gather / 3600.0
			key_function = lambda row: int(row["time"] / width_hours)
		else:
			if self.scans is None or len(self.scans) == 0:
				self.add_scans()
			key_function = lambda row: np.searchsorted(self.scans[:, 0], row["time"])
		for _, rows in groupby(data, key_function):
			groups.append(np.asarray(list(rows), dtype=data.dtype))
		return groups

	# ehtim 1.2.4 documents a list return; its np.array conversion breaks on NumPy 1.24+.
	ehtim.obsdata.Obsdata.tlist = time_list


def _load_uvfits(ehtim, path):
	with fits.open(path, memmap=False) as hdus:
		return ehtim.obsdata.load_uvfits(hdus, polrep="circ")


def image_uvfits(
	input_path,
	output_dir,
	image_size=80,
	fov_uas=150.0,
	averaging_seconds=60.0,
	max_iterations=500,
	major_iterations=5,
	parameter_set=0,
	second_input_path=None,
	show=False,
):
	"""Reconstruct Stokes-I from calibrated EHT UVFITS using eht-imaging."""
	product = classify_fits(input_path)
	if product["kind"] != "uvfits_visibility":
		raise ValueError(f"Expected UVFITS visibility data, detected {product['kind']}")
	if image_size < 16 or averaging_seconds <= 0 or max_iterations < 1 or major_iterations < 1:
		raise ValueError("image_size must be >= 16 and averaging_seconds/max_iterations/major_iterations must be positive")

	try:
		import ehtim as eh
	except ImportError as error:
		raise RuntimeError(
			"UVFITS imaging requires ehtim==1.2.4; install requirements.txt "
			"in the active environment."
		) from error

	if version("ehtim") != "1.2.4":
		raise RuntimeError(
		f"The validated Sgr A* UVFITS workflow requires ehtim==1.2.4, found {version('ehtim')}"
		)
	_enable_numpy_2_ragged_compatibility(eh)
	first_raw_obs = _load_uvfits(eh, input_path)
	day_of_year = (datetime(1858, 11, 17) + timedelta(days=int(first_raw_obs.mjd))).timetuple().tm_yday
	reference_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "EHTC_FirstSgrAResults_May2022" / "pipeline_reference"
	parameter_path = reference_dir / f"eht-imaging_{day_of_year:03d}_params.csv"
	preimcal_path = reference_dir / "preimcal.py"
	if not parameter_path.is_file() or not preimcal_path.is_file():
		raise FileNotFoundError(
			"The official EHT pre-imaging helper and parameter table are required. "
			f"Expected {preimcal_path} and {parameter_path}."
		)
	parameter_table = pd.read_csv(parameter_path)
	if parameter_set < 0 or parameter_set >= len(parameter_table):
		raise ValueError(f"Parameter combination must be in [0, {len(parameter_table) - 1}] for day {day_of_year:03d}")
	parameters = parameter_table.iloc[parameter_set]
	preimcal_spec = importlib.util.spec_from_file_location("astrorecon_eht_preimcal", preimcal_path)
	if preimcal_spec is None or preimcal_spec.loader is None:
		raise ImportError(f"Could not load official EHT pre-imaging helper at {preimcal_path}")
	preimcal = importlib.util.module_from_spec(preimcal_spec)
	preimcal_spec.loader.exec_module(preimcal)

	def prepare_observation(path):
		loaded = _load_uvfits(eh, path)
		loaded = loaded.switch_polrep("stokes", allow_singlepol=True)
		is_normalized = "_norm" in str(path).lower()
		original_cwd = Path.cwd()
		try:
			os.chdir(reference_dir)
			prepared = preimcal.preim_pipeline(
				loaded,
				is_normalized=is_normalized,
				is_deblurred=False,
				lcarr=None,
				nproc=1,
				do_LMTcal=False,
				do_JCMTcal=False,
				tint=averaging_seconds,
				syserr=float(parameters.syserr),
				ref_optype=None if str(parameters.ref_type).lower() == "false" else str(parameters.ref_type),
				ref_scale=float(parameters.ref_scale),
				do_deblurr=bool(parameters.deblurr),
				do_psd_noise=bool(parameters.psd_noise),
				a=float(parameters.psd_a),
				u0=float(parameters.psd_u0),
				b=float(parameters.psd_b),
				c=float(parameters.psd_c),
			)
		finally:
			os.chdir(original_cwd)
		return prepared.switch_polrep("stokes")

	obs = prepare_observation(input_path)
	input_paths = [str(Path(input_path).resolve())]
	if second_input_path is not None:
		second_product = classify_fits(second_input_path)
		if second_product["kind"] != "uvfits_visibility":
			raise ValueError("The second input must also be UVFITS visibility data")
		second_obs = prepare_observation(second_input_path)
		if second_obs.source != obs.source or second_obs.mjd != obs.mjd:
			raise ValueError("Combining UVFITS bands requires the same target and observing day")
		if not np.array_equal(second_obs.tarr["site"], obs.tarr["site"]):
			raise ValueError("Combining UVFITS bands requires matching station arrays")
		second_obs.data["time"] += 0.00002718
		obs.data = np.concatenate((obs.data, second_obs.data))
		obs.data = np.sort(obs.data, order=["time", "t1", "t2"])
		input_paths.append(str(Path(second_input_path).resolve()))
	if len(obs.data) == 0:
		raise ValueError("No valid visibilities remain after loading and averaging")

	input_is_normalized = all("_norm" in path.lower() for path in input_paths)
	if input_is_normalized:
		flux = 1.0
	else:
		uv_distance = obs.unpack("uvdist")["uvdist"]
		amplitudes = obs.unpack("amp")["amp"]
		short_baselines = uv_distance <= np.percentile(uv_distance, 10)
		flux = float(np.nanmedian(amplitudes[short_baselines]))
	if not np.isfinite(flux) or flux <= 0:
		raise ValueError(f"Could not estimate a positive flux prior from UVFITS (got {flux})")

	parameter_url = (
		"https://raw.githubusercontent.com/eventhorizontelescope/2022-D02-02/"
		f"main/eht-imaging/params/eht-imaging_{day_of_year:03d}_params.csv"
	)
	parameter_note = f"Official EHT parameter survey row {parameter_set} from {parameter_url}."
	fov = fov_uas * eh.RADPERUAS
	prior = eh.image.make_square(obs, image_size, fov)
	fwhm = float(parameters.fwhm)
	prior = prior.add_tophat(flux, fwhm * eh.RADPERUAS / 2.0).blur_circ(obs.res())
	data_terms = {"amp": float(parameters.amp), "cphase": 1.0, "logcamp": 1.0}
	regularizer_terms = {
		"simple": float(parameters.simple),
		"tv": float(parameters.tv),
		"tv2": float(parameters.tv2),
		"flux": 1.0,
	}
	imager = eh.imager.Imager(
		obs,
		prior,
		prior_im=prior,
		flux=flux,
		data_term=data_terms,
		reg_term=regularizer_terms,
		maxit=max_iterations,
		stop=1e-6,
		norm_reg=True,
		epsilon_tv=1e-10,
		major=60 * eh.RADPERUAS,
		minor=60 * eh.RADPERUAS,
		PA=0.0,
		ttype="direct",
		show_updates=False,
	)
	result = None
	for _ in range(major_iterations):
		result = imager.make_image_I(show_updates=False)
		imager.init_next = result.blur_circ(obs.res())
	image_array = np.asarray(result.imarr(), dtype=np.float64)
	if not np.isfinite(image_array).all():
		raise RuntimeError("eht-imaging returned non-finite image values; refusing to write a misleading render")
	data_chi_squared = {
		term: float(obs.chisq(result, dtype=term, ttype="direct"))
		for term in data_terms
	}

	output_dir = Path(output_dir)
	render_path = output_dir / "renders" / "uvfits_reconstruction.png"
	fits_path = output_dir / "volumes" / "uvfits_reconstruction.fits"
	metadata_path = output_dir / "uvfits_reconstruction.json"
	fits_path.parent.mkdir(parents=True, exist_ok=True)
	result.save_fits(str(fits_path))
	figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
	image_handle = axis.imshow(image_array, origin="lower", cmap="magma")
	axis.set_title(f"Sgr A* Stokes I | {obs.rf / 1e9:.1f} GHz")
	axis.set_xlabel("Relative RA pixel")
	axis.set_ylabel("Relative Dec pixel")
	figure.colorbar(image_handle, ax=axis, label="Jy / pixel")
	render_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(render_path, dpi=200)
	plt.close(figure)

	metadata = {
		"render_type": "EHT interferometric Stokes-I reconstruction",
		"data_source": input_paths,
		"source_kind": "calibrated UVFITS visibility data",
		"target": obs.source,
		"polarization": "Stokes I only; this release has no usable Q/U polarization",
		"ehtim_version": version("ehtim"),
		"visibility_count_after_averaging": int(len(obs.data)),
		"station_count": int(len(obs.tarr)),
		"frequency_hz": float(obs.rf),
		"nominal_resolution_uas": float(obs.res() / eh.RADPERUAS),
		"image_shape": list(image_array.shape),
		"field_of_view_uas": fov_uas,
		"averaging_seconds": averaging_seconds,
		"max_iterations": max_iterations,
		"major_iterations": major_iterations,
		"flux_prior_jy": flux,
		"survey_row": {key: (value.item() if hasattr(value, "item") else value) for key, value in parameters.to_dict().items()},
		"data_terms": data_terms,
		"regularizer_terms": regularizer_terms,
		"final_data_chi_squared": data_chi_squared,
		"optimizer_objective": float(imager.objfunc(imager._xinit)),
		"epsilon_tv": 1e-10,
		"fourier_operator": "direct (portable; NFFT optional)",
		"parameter_reference": parameter_note,
		"preprocessing": "EHT collaboration preimcal.py from release 2022-D02-02; survey-selected systematic, refractive, PSD, time-average, and scattering-deblur settings applied.",
		"preprocessing_helper": str(preimcal_path.resolve()),
		"caveat": "One epoch/band Stokes-I image is not tomography and cannot recover magnetic-field polarization. NFFT is unavailable here; the direct Fourier operator is slower than the official nfft configuration.",
		"code_modules": ["train.py", "src/fits_router.py", "src/uvfits_imaging.py", "ehtim"],
	}
	metadata["png_metadata"] = str(write_render_metadata(render_path, metadata))
	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	if show:
		show_render_gallery([render_path])
	return {"image": result, "render_path": render_path, "fits_path": fits_path, "metadata_path": metadata_path, "metrics": metadata}