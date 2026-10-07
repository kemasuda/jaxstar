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

Raw archives and extracted files are cached separately from generated grids.
`JAXSTAR_MIST_RAW_DIR` selects this cache (default: `mist/raw` under the jaxstar
platform cache); an explicit `cache_dir` takes precedence. Its keys include
the MIST release, data kind, and archive filename, which identifies rotation,
composition and photometry where applicable. Changing the jaxstar version or
the output NPZ path does not trigger another download. A different MIST release
has a separate cache entry.

The raw-data API can import existing archives or download missing ones:

```python
from jaxstar.mistfit.mistgrid.sources import track_archives, bolometric_correction_archive
from jaxstar.mistfit.mistgrid.cache import fetch_archive, extract_archive

# Descriptors only; feh=None lists all published metallicities without downloading.
sources = track_archives(version="1.2", rotation=0.4, afe=0., feh=[-0.25, 0., 0.25])
source = sources[0]
archive = fetch_archive(
    source, local_path=f"benchmark-results/mist-bc-inputs/{source.filename}",
)
# No local_path: reuse the cache, or download on a miss.
bc_archive = fetch_archive(bolometric_correction_archive())
# Optional extraction; archive-reading builders can use `archive` directly.
directory = extract_archive(source, offline=True)
```

An existing valid cache takes precedence over `local_path`. Imports copy the
source without modifying it. Reuse verifies byte size and SHA-256 locally;
`offline=True` forbids network access and fails on a cache miss. Explicit
`force=True` on `fetch_archive` refreshes an entry. Transfers are streamed and
validated as complete XZ/tar archives before publication. A failed transfer or
refresh leaves the previous successful entry intact; partial downloads restart
on retry. The receipt records the source URL, local hash, size and import/download
origin. The hash detects local corruption; it is not a publisher-supplied checksum.
Older content generations are retained when a refresh changes the archive.

Isochrone creation uses this cache and also recognizes existing CMD folders or
archives beside the output NPZ or in the earlier default cache:

```python
from jaxstar.mistfit.mistgrid.create_grid import create_mistgrid

create_mistgrid("/path/to/new/mistgrid_iso.npz", offline=True)
```

The registered sources currently cover MIST v1.2 EEP tracks at both published
rotation rates and solar alpha abundance, plus UBVRIplus CMD/BC archives.
Source catalogues are separate from the version-independent cache; MIST v2
catalogues, parsers and BC conventions remain future work. This raw-data API
supplies the track grid builder below.

### Building track grids

Select a release, rotation and alpha abundance; by default the builder downloads
all published FeH archives and retains all native masses and EEPs:

```python
from jaxstar.mistfit.mistgrid.create_tracks import create_track_grid

path = create_track_grid("mistgrid_tracks.npz", version="1.2", rotation=0.4, afe=0.)
# Optional subset: feh=[-0.25, 0., 0.25], mass_range=(0.8, 1.5), eep_range=(202, 550)
# Enable JAX x64 before loading with MistModel.from_tracks(path).
```

The builder reuses the shared raw cache and downloads missing inputs. To import
earlier downloads, add `local_dir="benchmark-results/mist-bc-inputs"`; files use
their official archive names. `offline=True` allows only cached/local inputs.
`raw_cache_dir=...` overrides the common cache location. Existing NPZ output
requires `overwrite=True`; a failed build leaves the previous output intact.
Regenerating a grid or updating jaxstar reuses the cached source archives. A new
MIST release has separate source files and cache entries.

The same operation is available from the command line:

```bash
python -m jaxstar.mistfit.mistgrid.create_tracks mistgrid_tracks.npz --rotation 0.4 --afe 0
# Optional: --feh -0.25 0 0.25 --mass-range 0.8 1.5 --eep-range 202 550
# Optional: --local-dir /path/to/archives --offline --raw-cache-dir /path/to/cache
```

Mass and EEP axes are unions of the selected native tracks, with no thinning.
Explicit bounds are inclusive; at least two masses, two FeH values and three
EEPs are required. Missing mass/FeH pairs and differing track endpoints remain
missing cells. `*_INTERP` alternatives are excluded. No age cut or additional
mass/EEP interpolation is applied during construction. `allow_missing=False`
(`--strict`) instead requires a complete, finite rectangle with increasing ages.
Malformed input files always raise an error.

MIST v1.2, both rotation rates, afe=0 and UBVRIplus BCs (Av=0, Rv=3.1) are
supported. BC interpolation follows the public recipe: Steffen in logT (linear
in the first/last intervals), linear in logg, then Steffen in metallicity (the
MESA limited three-point recipe at the edges). Inclusive table endpoints are
supported. Atmospheres outside the BC table have missing photometry; they are
not clipped or extrapolated. Available physical fields are still retained.
The release catalogue and parser are separate; adding MIST v2 requires its own
validated catalogue/parser before it can be selected.

Format 2 NPZ files store `track_present` and NaN values for missing fields,
plus `metadata_json` with build settings, coverage counts, native track members,
input archive SHA-256 values, BC conventions and builder/parser hashes. Read it
with `json.loads(np.load(path)["metadata_json"].item())`. Masks/metadata are not
model output keys. Legacy complete NPZ grids remain supported.

`MistModel` checks support separately for each requested output. All nonzero
interpolation corners must have that output's required native fields; exact
nodes/faces remain usable even beside a missing cell. Unsupported predictions
return `-inf`. Use `model(...)` for predictions: masked `model.grid` stores finite
placeholders for safe differentiation. The centered `dtdeep` uses valid native
three-node age stencils, with adjacent interior values copied to track ends;
missing nodes or nonpositive age increments invalidate the affected stencils.

Track inference excludes unsupported age/derivative or likelihood values.
Requesting an additional saved-only output does not change the target density;
that output can contain `-inf` when unavailable. Track plots leave gaps across
missing regions. Full grids describe available model coverage; posterior sampling
across every evolutionary phase has not been validated merely by building one.
Construction uses NumPy float64; JAX x64 is required when loading the model.

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

`MistFit.run_nuts` initializes both model kinds with a grid search (no SVI).
At fixed initial FeH = 0 and distance = 1 / observed positive parallax, it
searches native mass × EEP for tracks, or native log-age × EEP for isochrones.
It uses all EEP nodes within the sampling bounds and scores the existing joint
log density, including the run's prior/Jacobian and optional likelihood factors.
Uniform prior endpoints are excluded; a bound interval with no interior native
node uses its overlap midpoint. The best candidate with finite density and
gradients initializes all chains. All parameters are then free in NUTS;
this search does not guarantee finding every posterior mode.

```python
options = {"linear_age": True}
initial = fit.find_initial(model_kwargs=options)  # inspect constrained latent values
mcmc = fit.run_nuts(random.PRNGKey(0), model_kwargs=options)
# Or override the quantities fixed only during initialization:
mcmc = fit.run_nuts(random.PRNGKey(0), model_kwargs=options,
                   init_kwargs={"feh_init": -0.1, "distance": 0.15})  # kpc
```

Without a positive observed parallax, supply `init_kwargs={"distance": ...}`
or explicit initialization. `grid_init=False` uses the historical parallax-based
rule and NumPyro fallback. Explicit `init_strategy` or unconstrained `init_params`
always take precedence and bypass the search. Selected grid starting values are
saved in the run metadata as the `init_to_value` strategy's values.
`output_keys` adds to the default and observed outputs for that run, without
changing future runs. Standalone `run_nuts` retains NumPyro's default initialization.

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

## Evolutionary-track models

An explicit, prebuilt track NPZ can use the same prediction and fitting APIs:

```python
import jax
jax.config.update("jax_enable_x64", True)
from jaxstar.mistfit import MistModel, MistFit

track_model = MistModel.from_tracks("benchmark-results/mist-track-grid-extended/grid.npz")
prediction = track_model(
    {"mass_init": 1.16, "feh_init": 0., "eep": 435.5, "distance": .22},
    keys=("age", "teff", "radius", "gmag3", "dtdeep"),
)
track_fit = MistFit(model=track_model, observations={
    "teff": (6175., 60.), "feh": (0., .04), "parallax": (4.55, .05),
})
track_mcmc = track_fit.run_nuts(
    jax.random.PRNGKey(0), num_warmup=1500, num_samples=2000,
    model_kwargs={"linear_age": True},
    output_keys=("rho", "kmag_abs"),
)
```

Track inputs are initial mass, initial FeH and EEP; age is an output. The
NumPyro mass site remains named `mass`, with `mass_init` also saved as an alias.
`MistFit` selects `model_track` automatically. Mass, FeH and EEP bounds default
to the grid endpoints; `massmin`/`massmax`, `fmin`/`fmax`, and `eepmin`/`eepmax`
may narrow those ranges. Track mass bounds are hard limits. The age prior uses
the separately interpolated centered `dtdeep` (Gyr/EEP); `linear_age=False`
selects the log-age target. The track model does not take `flat_age_marginal`.

EEP uses a fixed interval for both NUTS and MultiNest. **The default is a soft
upper age bound at 13.8 Gyr**, with `age_bound_width=0.1` Gyr. Set
`model_kwargs={"age_max": None}` to disable it; `age_min` optionally adds a
lower penalty. These values are in Gyr. Each sigmoid contributes a log
penalty of −15 at its bound, approaching −30 outside (`age_bound_depth=30`).
This softly suppresses ages outside the bounds without excluding them or
changing EEP limits, and remains active for `nodata=True`. The track API no
longer accepts `logamin`, `logamax`, or `bound_eep_by_age`; isochrone age options
are unchanged.

This initial loader requires JAX x64, a complete rectangular grid with uniform
EEP spacing, and precomputed UBVRIplus photometry. It performs no downloads or
grid generation. The example grid is a local benchmark artifact, not bundled
package data: see [track grid construction](#building-track-grids)
and the [example notebook](examples/mistfit/fit_track.ipynb).

To draw a track at fixed initial mass and initial FeH:

```python
from jaxstar.mistfit import plot_track

fig = plot_track(track_model, mass_init=1.1, feh_init=0.,
                 x="teff", y="kmag_abs", age_range=(0., 13.8))
# Alternatively: eep_range=(250., 500.). Apparent magnitudes need distance (kpc).
fig = track_fit.plot_track(1.1, 0., distance=.2, y="kmag")
```

The range defaults to the full grid EEP extent, independently of inference age
bounds. Axes accept scalar MistModel output keys. `ax=...` allows overlaid
single tracks; temperature and magnitude axes follow HR-diagram directions.

`track_fit.plot_diagnostics(idata)` adds a `tracks` figure, with one panel per
observed magnitude versus Teff. It draws 20 joint posterior tracks, marks the
sampled positions and adds the recorded observations with error bars. Each
track uses its draw's initial mass, initial FeH and distance. By default, the
drawn EEP interval spans the nearest primary EEP boundaries enclosing the
minimum and maximum EEP across **all posterior draws**, not just the selected
20. Boundaries come from native track metadata and are clipped to grid coverage.
Explicit `eep_range` or `age_range` takes precedence. `auto_eep_range=False`
shows the full grid extent; grids without primary EEP metadata also retain
full coverage. These are drawing limits only and do not change the samples
or inference bounds. To customize it:

```python
figures = track_fit.plot_diagnostics(
    idata, track_kwargs={"x": "teff", "y": ("jmag", "kmag"),
                         "n_samples": 20, "seed": 0, "age_range": (0., 13.8)},
)
# Draw only this figure; y=None selects all observed magnitude bands.
fig = track_fit.plot_posterior_tracks(idata, y="kmag")
# Show the full track extent instead of the posterior's primary EEP interval:
fig = track_fit.plot_posterior_tracks(idata, y="kmag", auto_eep_range=False)
# The same option can be passed via plot_diagnostics(track_kwargs=...).
```

Selection is reproducible and uses the same draws across bands. Requested
axes are evaluated from the grid, so they need not have been saved as outputs.
For reloaded results, `plot_posterior_tracks(idata, track_model)` uses an
explicit matching model; `show_tracks=False` keeps diagnostics grid-free.

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
`plot_diagnostics` optionally writes `trace.png`, `corner.png`, and `tracks.png`
(when track plotting is enabled), and always
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
model for trace/corner. `plot_diagnostics` omits `trace` and returns `corner`
plus `tracks` when enabled; summaries omit MCMC Rhat/ESS.
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
