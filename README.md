# jaxstar

Fast isochrone fitting using HMC-NUTS. The code is described in https://arxiv.org/abs/2209.03279

## Installation

### PyPI (recommended)

```
pip install jaxstar
```

If you already have `jaxstar` installed, upgrade to ensure the latest wheel (which now bundles the full `jaxstar.mistfit.mistgrid` package) is used:

```
pip install --upgrade --force-reinstall jaxstar
```

### From source

```
pip install -e .
```

This installs the package in editable mode so local changes under `src/` are immediately importable.

### Requirements

* `jax`, `numpyro`
* [dustmaps](https://dustmaps.readthedocs.io/en/latest/) for extinction correction
* [gaiadr3-zeropoint](https://pypi.org/project/gaiadr3-zeropoint/) for the Gaia parallax zero-point correction

Synthetic CMDs are downloaded from http://waps.cfa.harvard.edu/MIST/model_grids.html#synthetic into the `mistgrid` directory when the ``MistGridIso`` or ``MistFit`` class is instantiated for the first time.

### Troubleshooting imports

If you encounter `ModuleNotFoundError: No module named 'jaxstar.mistfit.mistgrid'`, ensure you are running `jaxstar` v0.1.1 or later. Earlier wheels missed an `__init__.py` marker file, so Python did not treat `jaxstar/mistfit/mistgrid` as a package when installing from PyPI. Upgrading as shown above fixes the issue permanently.

## Examples

See [isochrone fitting example.ipynb](https://github.com/kemasuda/jaxstar/blob/main/examples/isochrone%20fitting%20example.ipynb) in `examples/`.
