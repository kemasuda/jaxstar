# jaxstar

Fast isochrone fitting using HMC-NUTS. The code is described in https://arxiv.org/abs/2209.03279



## Installation

### Quick start

```
python -m pip install --upgrade pip
python -m pip install jax jaxlib numpy pandas numpyro astropy dustmaps gaiadr3-zeropoint
python -m pip install jaxstar  # or: python -m pip install -e .
```

This is the combination that currently runs in our development environment (macOS + Python 3.11). Installing the dependencies up front avoids missing-import errors when invoking the extinction and correction utilities.

### Tested versions

| Package | Version |
| --- | --- |
| Python | 3.11 |
| jax / jaxlib | 0.8.0 |
| numpyro | 0.19.0 |
| numpy | 2.3.4 |
| pandas | 2.3.2 |
| astropy | 7.1.0 |
| dustmaps | 1.0.14 |
| gaiadr3-zeropoint | 0.1.0 |

### Data downloads

* Synthetic CMDs are downloaded from http://waps.cfa.harvard.edu/MIST/model_grids.html#synthetic under `mistgrid` when the ``MistGridIso`` or ``MistFit`` class is instantiated for the first time.

## Examples

see [isochrone fitting example.ipynb](https://github.com/kemasuda/jaxstar/blob/main/examples/isochrone%20fitting%20example.ipynb) in demos
