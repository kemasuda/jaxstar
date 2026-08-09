# jaxstar

Fast isochrone fitting using HMC-NUTS. The code is described in https://arxiv.org/abs/2209.03279



## Installation

```python setup.py install```

* requirements: jax, numpyro, [dustmaps](https://dustmaps.readthedocs.io/en/latest/) for extinction correction, [gaiadr3-zeropoint](https://pypi.org/project/gaiadr3-zeropoint/) for zero-point correction for the Gaia parallax

* synthetic CMDs are downloaded from http://waps.cfa.harvard.edu/MIST/model_grids.html#synthetic under mistgrid directory when the ``MistGridIso`` or ``MistFit`` class is instantiated for the first time. 

  To reuse an existing grid without downloading it again, pass its path explicitly:

  ```python
  from jaxstar.mistfit import MistFit

  fit = MistFit(path="/path/to/mistgrid_iso.npz")
  ```

## Examples

see [isochrone fitting example.ipynb](https://github.com/kemasuda/jaxstar/blob/main/examples/isochrone%20fitting%20example.ipynb) in demos
