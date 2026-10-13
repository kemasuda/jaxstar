# jaxstar

Differentiable stellar model interpolation and fitting with NUTS, powered by JAX.

`jaxstar` provides reusable physical models and NumPyro inference for stellar
properties and spectra:

- **MistFit** fits photometry, parallax and stellar measurements with MIST
  evolutionary tracks. `MistModel` also supports isochrone grids and can be
  used independently for predictions.
- **SpecFit** provides spectral interpolation, broadening and stellar component
  composition through `SpecModel`, with continuum likelihoods and a single-star
  NumPyro model for NUTS.

## Installation

Python 3.10 or later is required. Install the released package from PyPI:

```bash
python -m pip install jaxstar
```

The guides below describe the current source checkout. To use this version,
install from the repository root:

```bash
python -m pip install -e .
```

This source installation includes the dependencies for MistFit diagnostics and
result saving, and SpecFit inference, plotting and residual GPs. The only extra
is `test`, which adds pytest for development (see below).

## MistFit

Start with a MIST track grid, construct `MistModel.from_tracks(path)`, and pass
the model and observations to `MistFit`. The same model supports predictions,
NUTS, result saving and posterior track plots.

- [MistFit guide](docs/mistfit.md): predictions, priors, initialization,
  sampling and diagnostics.
- [MIST grid guide](docs/mistgrid.md): build a full track grid, reuse downloads,
  and configure the data cache.
- [Track fitting notebook](examples/mistfit/fit_track.ipynb): an end-to-end
  fitting example; set its grid path to your prepared NPZ.

Isochrone fitting remains available for comparison and existing analyses; see
the [isochrone guide section](docs/mistfit.md#isochrone-fitting) and
[isochrone notebook](examples/mistfit/fit_isochrone.ipynb).

## SpecFit

`SpecModel` evaluates prepared spectral grids independently of observations and
priors. `SpecFit` combines it with `Observation` for sampling, fit masks and
diagnostics. Use the supplied `model_single` or your own NumPyro model.

- [SpecFit guide](docs/specfit.md): spectral data, predictions, continuum and NUTS.
- [SpecFit workflow](dev_notebooks/specfit/specfit_end_to_end.ipynb): a synthetic
  example of initialization, fitting, masks and diagnostics.
- [SpecModel tutorial](docs/tutorials/specfit/specmodel.ipynb): physical spectra and
  multiple stellar components; requires the external reference data described
  in the notebook.
- [Continuum tutorial](docs/tutorials/specfit/continuum.ipynb): a self-contained
  synthetic example of continuum marginalization and additive jitter.

## Development

Install test dependencies and run the default suite, which does not download
the full MIST archive:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

For the optional historical full-grid regression test:

```bash
JAXSTAR_MISTGRID_PATH=/path/to/mistgrid_iso.npz python -m pytest -m full_grid
```

## Reference

The original isochrone-fitting method is described in
[Masuda (2022)](https://arxiv.org/abs/2209.03279).
