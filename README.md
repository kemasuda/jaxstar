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

`MistFit(observations=...)` and `MistModel.from_isochrones()` use the MIST v1.2 synthetic photometry grid.
On first use, jaxstar downloads the source CMD archive, generates the NumPy
grid, and stores it in the platform-specific user cache. Later environments
and reinstalls reuse that cached grid automatically.

To use a grid at a specific path:

```python
from jaxstar.mistfit import MistModel

model = MistModel.from_isochrones(path="/path/to/mistgrid_iso.npz")
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

## Deterministic isochrone predictions

`MistModel` evaluates stellar properties independently of observations and priors:

```python
from jaxstar.mistfit import MistFit, MistModel

model = MistModel.from_isochrones(path="/path/to/mistgrid_iso.npz")
params = {"logage": 9.3, "feh_init": 0., "eep": 350., "distance": .1}
prediction = model(params, keys=("teff", "mass_init", "mass_current",
                                "feh_surface", "kmag_abs", "kmag", "parallax"))
```

Here logage is log10(age in years), distance is kpc, masses are solar masses,
and parallax is mas. `kmag_abs` is absolute magnitude and `kmag` is apparent
magnitude. Each call chooses its outputs without changing model state. The
immutable model and named results support JIT, gradients and batching.
`MistFit` uses the named grid and exposes this prediction layer as
`fit.physical_model`. `MistFit(..., legacy_interpolation=True)` selects the
first-node-spacing arithmetic without input sanitization for numerical comparisons;
named endpoint/fill behavior is retained. The former `grid_backend` argument is
removed, along with `MistGridIso`, `NamedMistGridIso` and `fit.mg`.
Both interpolation settings retain the existing priors and likelihood options,
including the independently interpolated centered-difference `dmdeep` field.

## Isochrone inference

Pass observations as `{output_key: (value, Gaussian_error)}` when constructing a
fit, then run the sampler directly:

```python
from jax import random
from jaxstar.mistfit import MistFit

fit = MistFit(model=model, observations={
    "teff": (5700., 70.), "feh": (0., .05),
    "kmag": (8., .03), "parallax": (10., .1),
})
mcmc = fit.run_nuts(
    random.PRNGKey(0), num_warmup=1000, num_samples=1000,
    model_kwargs={"linear_age": True, "flat_age_marginal": False},
    output_keys=("rho", "kmag_abs"),
)
samples = mcmc.get_samples(group_by_chain=True)  # (4 chains, 1000 draws, ...)
mcmc.print_summary()
```

The default is four chains with `chain_method="parallel"`; both settings can be
changed per run. If there are too few devices, NumPyro warns and runs sequentially.
Each call returns a new NumPyro `MCMC`. `run_hmc` is an alias
with this same API; `set_data`, `setup_hmc`, `fit.mcmc` and `fit.samples` are
removed. Reuse a model in another fit to change observations. Existing parallel
lists can be supplied as `MistObservations(keys, values, errors)`.

For four parallel CPU chains, configure devices at the very beginning of the
script/notebook, before JAX initializes its devices (restart an already-running
notebook kernel first):

```python
import numpyro
numpyro.set_host_device_count(4)
```

MistFit does not change the process-wide device configuration.

Initialization retains the existing parallax-based rule and NumPyro fallback;
no preliminary fit to the observations is performed. A custom `init_strategy`
is accepted, and required if parallax is absent. `output_keys` adds to the default
and observed outputs for that run, without changing future runs.

`model_isochrone(observations, physical_model, ...)` is the shared probability
function in `jaxstar.mistfit.inference`, alongside the NUTS/MultiNest execution
functions. `mistfit.py` contains the thin MistFit interface, and `fit.model(...)`
delegates to the common function. The same model can be used with the optional
MultiNest backend:

```python
result_dir = "results/isochrone_multinest"
runner = fit.run_multinest(
    outputfiles_basename=result_dir + "/run-", n_live_points=800, resume=False,
    model_kwargs={"linear_age": True}, output_keys=("rho", "kmag_abs"),
)
idata = fit.to_inference_data(runner)
fit.save_result(idata, result_dir)
figures = fit.plot_diagnostics(idata, output_dir=result_dir)
```

MultiNest requires `numpyro-inferutils`, `pymultinest` and the native MultiNest
library. These are loaded only on that path; NUTS has no new dependencies.
Both samplers keep the existing priors, Jacobian field and likelihood options.

## Save results and plot diagnostics

Install the optional plotting/NetCDF dependencies in a source checkout:

```bash
python -m pip install -e ".[mistfit-diagnostics]"
```

NUTS still returns MCMC; MultiNest still returns its runner. Both carry a snapshot
of the run configuration, observations, RNG information and model/grid provenance.
Convert once when using the same result for several operations:

```python
from jaxstar.mistfit import load_result, plot_diagnostics, summarize, get_run_metadata

idata = fit.to_inference_data(mcmc)
result_dir = fit.save_result(idata, "results/star")
figures = fit.plot_diagnostics(idata, output_dir=result_dir)
# Individual plots are also available:
# fit.plot_trace(idata, keys=("age", "eep"))
# fit.plot_corner(idata, keys=("mass", "radius", "age"))

# In another session: no MistFit, MistModel or grid is needed.
loaded = load_result("results/star")
figures = plot_diagnostics(loaded, keys=("mass", "radius", "age"))
summary = summarize(loaded)
settings = get_run_metadata(loaded)
```

`save_result` writes `posterior.nc` (ArviZ InferenceData), `run.json` and
`summary.csv` (mean, standard deviation, 68% HDI, and MCMC Rhat/ESS).
`plot_diagnostics` optionally writes `trace.png` and `corner.png`, and always
returns a dictionary of Matplotlib Figures. Existing files require explicit
`overwrite=True`; choose a new directory for each run. Missing requested outputs
must be included via `output_keys` when sampling.

NUTS keeps chain/draw dimensions and, by default, divergence flags, potential
energy, acceptance probability, leapfrog counts and step sizes. Trace plots
keep chains separate; corner plots pool draws. Observed reference lines use
matching keys only, preserving initial/surface FeH and absolute/apparent magnitude
conventions. Labels carry the physical meaning and units.

For MultiNest, use the same operations with `runner` or
`fit.to_inference_data(runner)`. Conversion reconstructs registered deterministic
outputs from the equal-weight posterior once; saved/reloaded data then need no
model. `plot_diagnostics` returns only `corner`, and summaries omit MCMC Rhat/ESS.
The run record includes `logZ`/`logZerr`. MultiNest output, posterior, summary and
figures share the same result directory. `save_result` keeps MultiNest files
already there in place; saving to another directory copies them alongside the
posterior. Files overwritten by a later run are rejected instead of being mixed
with an earlier posterior.
`outputfiles_basename` is a filename prefix, not a directory: for example,
`"results/isochrone_multinest/run-"` writes `run-stats.dat` and other files inside
`results/isochrone_multinest/`. Use a short relative prefix because older MultiNest
libraries truncate filenames at 100 characters. The native call preserves that
relative prefix; the returned runner and provenance use absolute paths for later
analysis and archiving.

Results are portable analysis artifacts, not pickles or NUTS restart checkpoints.
Custom callable implementations are described but not serialized. Results from
external/unrecorded sampler calls can still be saved, with unavailable provenance
explicitly marked. Re-running a native sampler directly does not refresh the
run record: use the jaxstar execution functions for each recorded run.

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
