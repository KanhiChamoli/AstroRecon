# AstroRecon

**Differentiable general-relativistic ray tracing and volumetric reconstruction from real astronomical data.**

AstroRecon takes 2D astronomical observations — FITS images, integral-field cubes, and EHT UVFITS visibilities — and reconstructs a 3D density volume behind them through a differentiable, GPU-accelerated forward model. For compact objects it integrates null geodesics through a Schwarzschild spacetime in the Kerr–Schild chart, so black-hole shadows, photon rings, relativistic Doppler beaming, and gravitational lensing of a background starfield emerge from the physics rather than from artistic post-processing.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

| Capability | Status |
|---|---|
| Differentiable GR ray tracer (Schwarzschild, Kerr–Schild chart, RK4) | ✅ verified |
| Analytic Christoffel symbols, null-invariant conservation to ~1e-13 | ✅ verified |
| Event-horizon capture, photon sphere, Einstein-ring boundary triggers | ✅ verified |
| Radiative transfer with optical depth, Doppler beaming (δ³/δ⁴), gravitational redshift | ✅ verified |
| Stokes Q/U polarization + divergence-free (∇·B) MHD penalty | ✅ verified |
| EHT Sgr A* 2017 UVFITS imaging (ehtim, closure phases, survey calibration rows) | ✅ verified |
| FITS image / cube ingestion with WCS preservation | ✅ verified |
| PyQt5 desktop GUI with data-archive search (SkyView, MAST, VizieR, HEASARC) | ✅ verified |
| Cinematic rendering: ACES tonemap, bloom, blackbody color, lensed starfield | ✅ verified |
| Kerr (frame-dragging) metric, dirty-beam forward model in uv space | 🚧 planned |

## Key results

- **Null invariant** `|g_μν p^μ p^ν|` conserved to **7.1e-13 over 20,000 RK4 steps**; 4th-order convergence confirmed (8.8e-11 → 6.5e-13 as dλ halves).
- **Shadow geometry**: the rendered black-hole shadow matches the analytic critical impact parameter b_crit = √27 M; ray accounting closes exactly (absorbed + escaped = all pixels, 0 unresolved).
- **Doppler asymmetry**: inclined views of a Keplerian torus give mean Doppler factors of 1.014 (approaching) vs 0.895 (receding) — the correct relativistic beaming signature.
- **Real data**: 300-epoch inverse rendering against combined EHT Sgr A* Days 096+097 dirty images (8 UVFITS files, 4 calibration variants); ring-like structure recovered in the volume mid-slice with the MHD divergence penalty held at zero throughout.
- Performance: 65,536 rays × 200 RK4 geodesic steps in **1.7 s** on a laptop GPU (NVIDIA T1200, 4 GB); 64×64 render + full backward in **3.2 s**.

## Installation

```bash
# 1. Clone
git clone https://github.com/KanhiChamoli/AstroRecon.git
cd AstroRecon

# 2. Create the environment (or use pip + your own interpreter)
conda env create -f AstroRecon/environment.yml
conda activate astrorecon

# 3. PyTorch: install the build matching your CUDA from pytorch.org
python -m pip install -r AstroRecon/requirements.txt
```

Core dependencies: `torch`, `numpy`, `scipy`, `astropy`, `matplotlib`, `PyQt5`, `ehtim==1.2.4` (interferometric route), `astroquery` (archive access).

## Quick start

Run from the `AstroRecon/` subdirectory so the `src` package resolves.

**Desktop GUI**

```powershell
python main_gui.py
```

**Reconstruct a FITS observation into a 3D volume**

```powershell
python train.py --input data/raw/M13_sample.fits --epochs 1000 --depth 64 --denoise-sigma 1.25
```

**Image real EHT Sgr A* visibilities** (Day 096, both normalized bands, survey-selected parameter row)

```powershell
python train.py --input data/raw/EHTC_FirstSgrAResults_May2022/uvfits_norm/ER6_SGRA_2017_096_lo_casa_netcal-LMTcal-norm_StokesI.uvfits `
  --input2 data/raw/EHTC_FirstSgrAResultS_May2022/uvfits_norm/ER6_SGRA_2017_096_hi_casa_netcal-LMTcal-norm_StokesI.uvfits `
  --size 80 --epochs 500 --major-iterations 5 --parameter-set 0
```

**One-command cinematic Sgr A* render** (trains a 32³ torus-prior volume against the real EHT dirty image, then renders at 64² → 1024² with SSAA, adaptive stepping, ACES + bloom):

```powershell
python run_sgra_cinematic.py
```

**Orbit demo** (N-body CSV + plot)

```powershell
python orbit_simulation.py
python plot_orbit.py --show
```

Outputs land in `outputs/`: reconstruction comparison PNGs, VTK volumes, per-image metadata JSONs (data provenance, processing steps, library versions), and data-fit chi-squared diagnostics.

## The forward model

**Optically thin ray integration** (volumetric track):

$$\hat I(u,v)=\Delta t\sum_k V(x_k,y_k,z_k)$$

**GR radiative transfer** (compact-object track), integrated along null geodesics with per-event emission, Doppler factor δ, and optical depth τ:

$$\hat I=\sum_k j\,\delta^3\,g_k\,e^{-\tau_k}\,\Delta\lambda$$

**Objective** — reprojection error plus physical regularizers:

$$\mathcal L=\operatorname{MSE}(\hat I,I_{\mathrm{obs}})
+\lambda_{\mathrm{TV}}\,\mathrm{TV}(V)
+\lambda_{\rho}\lVert V\rVert_1
+\lambda_{\mathrm{MHD}}\lVert\nabla\cdot\mathbf B\rVert^2$$

Everything — camera intrinsics, RK4 geodesic steps, voxel sampling, the loss — is differentiable, so gradients flow from the rendered image all the way back into the density volume and camera parameters (`torch.autograd` verified end-to-end on CPU and CUDA).

## Repository layout

```
AstroRecon/
├── src/
│   ├── ray_marcher.py          # GRRT core: geodesics, Kerr–Schild metric, renderer
│   ├── losses.py               # Composite loss, Stokes Q/U, ∇·B operator
│   ├── uvfits_imaging.py       # EHT visibility imaging (ehtim, closure terms)
│   ├── pipeline_router.py      # Data-format → pipeline dispatcher (extensible)
│   ├── physics_dispatcher.py   # Object-class → physics-track router
│   ├── data_library.py         # SkyView / MAST / VizieR / HEASARC cache manager
│   ├── ui_epochs.py            # Multi-epoch tomography selection modal
│   └── ...                     # I/O, metadata, visualization, WWT bridge
├── gui/                        # PyQt5 application (main window, modals, theming)
├── scripts/                    # Utility checks
├── train.py                    # Reconstruction entry point
├── run_sgra_cinematic.py       # Cinematic Sgr A* runner
├── orbit_simulation.py         # N-body orbit generator
└── CHANGELOG.md                # Detailed development logbook
```

## Scientific caveats

This project is honest about its limits, as any research code should be:

- One 2D observation cannot recover true depth structure; the 3D volume from a single image is a uniform extrusion. Genuine tomography requires multiple calibrated projections with known geometry.
- Spectral FITS axes (wavelength / radial velocity) are **not** geometric distance; converting them to line-of-sight depth requires a defensible physical model.
- The local EHT route uses one survey parameter row and the portable direct Fourier operator (NFFT not installed); the published collaboration analysis compares an ensemble across rows with NFFT. Treat outputs as research reconstructions, not publication-grade reproductions.
- Rendered Stokes I is a clean-component image while EHT targets are dirty images; faithful comparison requires dirty-beam convolution (on the roadmap).

## Citation

If you use AstroRecon in your research, please cite it (see [CITATION.cff](AstroRecon/CITATION.cff)):

```bibtex
@software{Chamoli_AstroRecon_2026,
  author  = {Chamoli, Aaradhya},
  title   = {AstroRecon: Differentiable General-Relativistic Ray Tracing
             and Volumetric Reconstruction from Astronomical Data},
  year    = {2026},
  url     = {https://github.com/KanhiChamoli/AstroRecon}
}
```

## Acknowledgments

- The Event Horizon Telescope Collaboration for the public Sgr A* 2017 data release and `preimcal.py` calibration reference.
- NASA SkyView, MAST, CDS VizieR/SIMBAD, and HEASARC for archive access.
- The `ehtim`, `astropy`, `pywwt` (WorldWide Telescope), and PyTorch communities.

## License

[MIT](LICENSE) © 2026 Aaradhya Chamoli
