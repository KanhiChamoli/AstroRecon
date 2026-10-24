# AstroRecon

AstroRecon reconstructs a nonnegative 3D density volume from a 2D astronomical
observation. The canonical project root is this directory (`D:\Astrorecon`);
run commands from here so Python imports the root-level `src` package.

## Project layout

- `data/raw/`: input FITS observations.
- `data/synthetic/`: generated baseline observations for smoke runs.
- `outputs/renders/`: reconstruction comparisons and orbit PNGs.
- `outputs/volumes/`: reconstructed density volumes in legacy VTK format.
- `src/`: FITS I/O, ray integration, losses, and visualization.
- `train.py`: FITS-driven or synthetic reconstruction entry point.
- `orbit_simulation.py` and `plot_orbit.py`: orbit CSV generation and plotting.

## Setup

Activate the environment you want to use, then install the dependencies:

```powershell
conda activate astrorecon
python -m pip install -r requirements.txt
```

PyTorch must be installed for the selected Python interpreter. If the default
package index selects an unsuitable build, use the official PyTorch installer
selector for the desired CPU or CUDA build.

## Train

From `D:\Astrorecon`, run a synthetic smoke reconstruction:

```powershell
python train.py --epochs 50 --size 64
```

To reconstruct an observation, pass a FITS file:

```powershell
python train.py --input data/raw/observation.fits --epochs 100 --size 64
```

Training writes `outputs/renders/reconstruction.png` and
`outputs/volumes/density.vtk`. Without `--input`, it generates and saves a
repeatable nebula-like target at `data/synthetic/baseline.npy`.

## Orbit demo

```powershell
python orbit_simulation.py
python plot_orbit.py
```

The simulator writes `orbit_data.csv` with `Step`, `Time_Myr`, `X`, `Y`, and
`Z` columns. The plot is saved to `outputs/renders/orbit.png`; add `--show` to
open an interactive window.

## Model

The forward model integrates density along the depth axis:

$$\hat I(u,v)=\Delta t\sum_k V(k,u,v).$$

The objective combines reprojection error with 3D total variation and a weak
density penalty:

$$\mathcal L=\operatorname{MSE}(\hat I,I_{\mathrm{obs}})
+\lambda_{\mathrm{TV}}\operatorname{TV}(V)
+\lambda_{\rho}\lVert V\rVert_1.$$

Magnetic-field divergence is not included because this project does not yet
load or model a magnetic field. FITS input support and the synthetic training
path are implemented; real scientific validation still requires suitable
observations and domain-specific boundary conditions.
