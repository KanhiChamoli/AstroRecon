# AstroRecon

AstroRecon reconstructs a nonnegative 3D density volume from a 2D astronomical
observation. Run commands from this directory so Python resolves the sibling
`src` package.

## Layout

- `data/raw/`: input FITS observations.
- `data/synthetic/`: generated baseline observations for smoke runs.
- `outputs/renders/`: reconstruction comparison and orbit PNG images.
- `outputs/volumes/`: reconstructed density volumes in legacy VTK format.
- `src/`: FITS I/O, ray integration, losses, and visualization.
- `train.py`: FITS-driven or synthetic reconstruction entry point.
- `orbit_simulation.py` and `plot_orbit.py`: orbit CSV generation and plotting.

## Setup (Windows PowerShell)

The project environment is installed at `D:\\miniconda_envs\\astrorecon`. The
`conda` command may not be available in a regular PowerShell terminal unless
Conda has been initialized there. You can launch AstroRecon directly with the
environment's Python (no activation required):

```powershell
Set-Location D:\\Astrorecon\\AstroRecon
& "D:\\miniconda_envs\\astrorecon\\python.exe" main_gui.py
```

To enable `conda activate` in PowerShell, open **Anaconda Prompt** and run
`conda init powershell`, then close and reopen PowerShell. Alternatively,
initialize a PowerShell session using Conda's hook (adjust the install location
if yours is different):

```powershell
. "C:\\Users\\$env:USERNAME\\miniconda3\\shell\\condabin\\conda-hook.ps1"
conda activate D:\\miniconda_envs\\astrorecon
```

If the environment does not exist yet, create it from the repository root with
`conda env create -f environment.yml`. PyTorch must be installed in the selected
environment; if needed, use the official PyTorch selector for the appropriate
CPU or CUDA build.

PyTorch must be installed in the selected Python environment. If the package
index chooses an unsuitable build, use the official PyTorch selector for the
appropriate CPU or CUDA build.

## Train

Run a reconstruction from the bundled sample FITS image at its native 300x300
resolution, with Gaussian denoising and 1,000 TV-regularized optimization steps:

```powershell
python train.py --input data/raw/M13_sample.fits --epochs 1000 --depth 64 --denoise-sigma 1.25
```

To process another observation, place its FITS file in `data/raw/` and pass its path:

```powershell
python train.py --input data/raw/observation.fits --epochs 1000 --depth 64
```

The EHT calibrated Sgr A* release is visibility data, so it is sent to the
interferometric imaging route rather than the image denoiser. For day 096, use
both normalized bands and a survey-selected EHT parameter row:

```powershell
python train.py --input data/raw/EHTC_FirstSgrAResults_May2022/uvfits_norm/ER6_SGRA_2017_096_lo_casa_netcal-LMTcal-norm_StokesI.uvfits --input2 data/raw/EHTC_FirstSgrAResults_May2022/uvfits_norm/ER6_SGRA_2017_096_hi_casa_netcal-LMTcal-norm_StokesI.uvfits --size 80 --epochs 500 --major-iterations 5 --parameter-set 0
```

The UVFITS route uses `ehtim==1.2.4`, converts the release's circular
correlations to Stokes I, averages visibilities coherently, and images using
visibility amplitude, closure phase, and log-closure-amplitude terms. It uses
the portable direct Fourier operator, so it does not require NFFT; it is slower
than NFFT on large images. The implementation applies the EHT collaboration's
released `preimcal.py` and day-specific survey row, then performs iterative
blur/refine passes. `--size` selects image width, `--epochs` is the maximum
inner optimizer iterations per pass, `--major-iterations` selects the number
of passes, and `--parameter-set` selects a row in the official day-specific
table. The metadata records the selected hyperparameters and final data
chi-squared diagnostics.

The published EHT analysis compares an ensemble across many survey rows and
uses the NFFT operator. This local run uses one selected row and the slower
direct Fourier operator because NFFT is not installed on this Windows
environment. Treat the output as a research reconstruction, not a
publication-grade reproduction; run multiple survey rows and compare against
the collaboration's released images before making scientific claims.

FITS tables and hypercubes with more than three image axes are detected and
reported as unsupported rather than guessed. Their calibration/product
semantics must be supplied before a correct processor can be selected.

The input is background-subtracted, contrast-stretched, and Gaussian-filtered;
TV regularization then smooths the image while retaining bright structures.
Outputs include the four-panel `outputs/renders/reconstruction.png`, the
standalone `outputs/renders/filtered_input.png`, a VTK volume, and
`outputs/reconstruction.json` with filter settings and metrics. The 3D volume is
a uniform extrusion of the 2D reconstruction; one observation cannot recover
depth structure. Without `--input`, a repeatable synthetic observation is used.

Every generated PNG has an adjacent `*.metadata.json` file documenting the
source data, processing steps, code modules, Python version, and installed
Astropy, Matplotlib, NumPy, SciPy, and PyTorch versions.

### Measured 3D FITS cubes

If you have an integral-field or spectral-line FITS cube, process it without
collapsing its third axis:

```powershell
python train.py --input-cube data/raw/observation_cube.fits --denoise-sigma 1.0
```

This preserves the FITS WCS header and writes a denoised cube plus projection
and slice views. Check `CTYPE3` and `CUNIT3`: wavelength or radial velocity is
not geometric distance. Converting spectroscopy to line-of-sight depth requires
a defensible physical velocity/distance model. A true tomographic reconstruction
instead needs multiple calibrated projections with known geometry; repeated
images from one viewing angle do not provide that information.

### Resolution limits

The quoted 30-60 nm RMS wavefront residual is an adaptive-optics/instrument
measurement, not an image-filter setting. Likewise, 6-12 milliarcseconds is an
angular-resolution claim that depends on aperture, wavelength, sampling, and
the measured point-spread function. This repository's sample FITS image does
not include enough calibration to verify either figure. Filtering can reduce
noise but cannot restore unresolved detail or establish those instrument-level
accuracies.

## Orbit demo

```powershell
python orbit_simulation.py
python plot_orbit.py
```

The CSV contains `Step`, `Time_Myr`, `X`, `Y`, and `Z`; the chart is saved to
`outputs/renders/orbit.png`. Add `--show` to open an interactive window.

## Model

The forward model integrates density along the depth axis:

$$\hat I(u,v)=\Delta t\sum_k V(k,u,v).$$

The objective combines reprojection MSE with 3D total variation and a weak
density penalty:

$$\mathcal L=\operatorname{MSE}(\hat I,I_{\mathrm{obs}})
+\lambda_{\mathrm{TV}}\operatorname{TV}(V)
+\lambda_{\rho}\lVert V\rVert_1.$$

Magnetic-field divergence is not modeled because no magnetic-field data is
loaded. FITS ingestion and synthetic smoke training are implemented; scientific
validation still depends on suitable observations and boundary conditions.$