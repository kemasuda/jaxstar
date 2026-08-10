# jaxstar

Fast stellar isochrone fitting with HMC/NUTS, powered by JAX and NumPyro.
The method is described in [Masuda (2022)](https://arxiv.org/abs/2209.03279).

## Installation

jaxstar requires Python 3.10 or later. Install the released package from PyPI:

```bash
python -m pip install jaxstar
```

For development and testing from a source checkout:

```bash
python -m pip install -e ".[test]"
```

## MIST grid data

`MistFit()` and `MistGridIso()` use the MIST v1.2 synthetic photometry grid.
On first use, jaxstar downloads the source CMD archive, generates the NumPy
grid, and stores it in the platform-specific user cache. Later environments
and reinstalls reuse that cached grid automatically.

To use a grid at a specific path:

```python
from jaxstar.mistfit import MistFit

fit = MistFit(path="/path/to/mistgrid_iso.npz")
```

Alternatively, set the `JAXSTAR_MISTGRID_PATH` environment variable:

```bash
export JAXSTAR_MISTGRID_PATH=/path/to/mistgrid_iso.npz
```

The grid path is resolved in this order:

1. explicit `path=...` argument;
2. `JAXSTAR_MISTGRID_PATH`;
3. the versioned user cache.

If the resolved default cache file is missing, it is generated transparently.
The source CMDs come from the
[MIST synthetic photometry grids](https://mist.science/model_grids.html#synthetic).

## Optional correction utilities

The core package dependencies are installed automatically. Functions in
`jaxstar.utils.correction` additionally use:

- [dustmaps](https://dustmaps.readthedocs.io/en/latest/) for Bayestar extinction
  corrections;
- [gaiadr3-zeropoint](https://pypi.org/project/gaiadr3-zeropoint/) for Gaia EDR3
  parallax zero-point corrections.

Install them when those utilities are needed:

```bash
python -m pip install dustmaps gaiadr3-zeropoint
```

## Testing

The default test suite is fast, deterministic, and does not download the full
MIST archive:

```bash
python -m pytest -q
```

To run the optional historical full-grid regression test:

```bash
JAXSTAR_MISTGRID_PATH=/path/to/mistgrid_iso.npz \
python -m pytest -m full_grid
```

## Examples

Start with the
[isochrone fitting example](https://github.com/kemasuda/jaxstar/blob/main/examples/isochrone%20fitting%20example.ipynb),
or browse the [examples directory](https://github.com/kemasuda/jaxstar/tree/main/examples).
