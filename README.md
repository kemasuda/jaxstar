# jaxstar

Fast isochrone fitting using HMC-NUTS. The code is described in https://arxiv.org/abs/2209.03279



## Installation

```bash
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
```

* requirements: jax, numpyro, [dustmaps](https://dustmaps.readthedocs.io/en/latest/) for extinction correction, [gaiadr3-zeropoint](https://pypi.org/project/gaiadr3-zeropoint/) for zero-point correction for the Gaia parallax

* synthetic CMDs are downloaded from http://waps.cfa.harvard.edu/MIST/model_grids.html#synthetic when the ``MistGridIso`` or ``MistFit`` class is instantiated for the first time. The generated grid is stored in the platform-specific user cache and reused across environments and reinstalls.

  To reuse an existing grid, pass its path explicitly:

  ```python
  from jaxstar.mistfit import MistFit

  fit = MistFit(path="/path/to/mistgrid_iso.npz")
  ```

  Alternatively, set ``JAXSTAR_MISTGRID_PATH``. An explicit ``path`` takes precedence over the environment variable and the user cache.

  The historical full-grid regression test can be run with:

  ```bash
  JAXSTAR_MISTGRID_PATH=/path/to/mistgrid_iso.npz python -m pytest -m full_grid
  ```

## Examples

see [isochrone fitting example.ipynb](https://github.com/kemasuda/jaxstar/blob/main/examples/isochrone%20fitting%20example.ipynb) in demos
