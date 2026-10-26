# Contributing to AstroRecon

Thank you for considering a contribution. This is research code, so clarity
and scientific honesty matter as much as the diff itself.

## Development setup

1. Fork and clone the repository.
2. Create the environment: `conda env create -f AstroRecon/environment.yml`
3. Install the PyTorch build matching your CUDA (see pytorch.org selector).
4. Run the GUI smoke test to confirm your setup: `python gui/smoke_test.py`

## Ground rules

- **Physics changes need verification numbers.** Any change to
  `src/ray_marcher.py` or `src/losses.py` should state in the PR how the
  null invariant, shadow geometry, or loss round-trip was checked, with the
  measured tolerances (see `CHANGELOG.md` for the format used so far).
- **No undocumented claims.** Don't add accuracy/resolution figures that the
  data can't support. The README's "Scientific caveats" section exists for a
  reason.
- **Keep metadata provenance.** New output artifacts should carry a
  `*.metadata.json` describing source data, processing steps, and library
  versions, matching the existing pattern.
- **Style.** Format with `black`, keep modules importable without a GPU, and
  avoid new heavy imports at module level (use lazy imports as the existing
  code does for `astropy`/`astroquery`).

## Pull requests

1. Branch from `master` with a descriptive name (`feat/dirty-beam-term`).
2. Update `CHANGELOG.md` at the top with the date, files touched, and what
   was verified.
3. Ensure `python -m py_compile` passes on every touched file and the GUI
   smoke test still passes.
4. Describe *what you verified*, not just what you changed.

## Reporting bugs

Open an issue with: your OS, Python/PyTorch versions, the exact command
run, the full traceback, and (for rendering issues) the output image.
