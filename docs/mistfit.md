# MistFit guide

[README](../README.md) · [MIST grids and downloads](mistgrid.md) · [SpecFit](specfit.md)

`MistModel` predicts stellar quantities; `MistFit` combines a model with fixed
observations and provides sampling and diagnostics. This guide starts with
evolutionary tracks, where the coordinates are initial mass, initial FeH and
equivalent evolutionary point (EEP), and age is an output. The
[track notebook](../examples/mistfit/fit_track.ipynb) shows an end-to-end example
using a small v1.2 grid included in the repository, with no grid download or
build needed. Its [data notes](../examples/mistfit/data/README.md) give the
coverage, sources, and regeneration command.

The [MIST v2.5 track notebook](../examples/mistfit/fit_track_v25.ipynb) includes
the grid-building command, NUTS fitting, and posterior predictions of structural
quantities such as envelope thickness and convective turnover times. It also
overlays the saved v1.2 posterior from `fit_track.ipynb` on a shared corner plot.
Track-grid construction currently supports `afe=0` only.

[Isochrone fitting](#isochrone-fitting) uses the same interface with different
coordinates and prior options.

## Load a track model and predict

**Create the grid once, then reuse it.** First
[build a track grid](mistgrid.md#build-a-track-grid) or choose an existing track
NPZ. In the example below, `"mistgrid_tracks.npz"` is the path to that saved file.
The track builder accepts `version="2.5"` only with `afe=0`; v1.2 remains the default.
Both use the same prediction and fitting API. `model.metadata` records the
release and composition when the NPZ supplies them.
`from_tracks` requires an explicit path and JAX x64; it does not download or
generate data. Run this setup before initializing JAX devices:

```python
import jax
import numpyro

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)  # for four parallel CPU chains below

from jaxstar.mistfit import MistFit, MistModel

model = MistModel.from_tracks("mistgrid_tracks.npz")
prediction = model(
    {"mass_init": 1.1, "feh_init": 0., "eep": 400., "distance": .1},
    keys=("age", "teff", "radius", "mass_current", "kmag_abs", "kmag"),
)
print(prediction["teff"])
```

Restart an already-running notebook kernel if its device configuration has
been initialized. MistFit itself does not change JAX precision or device counts.

Each call selects outputs without changing the model. `keys` accepts a name,
an ordered sequence, or `None` for all available outputs; inspect
`model.input_keys` and `model.output_keys` for the loaded grid.

| Key | Meaning and units |
| --- | --- |
| `mass_init`, `mass` | Initial mass in solar masses |
| `mass_current`, `star_mass` | Current mass in solar masses |
| `feh_init` | Initial FeH in dex |
| `feh_surface`, `feh`, `feh_photosphere` | Surface FeH in dex |
| `age`, `logage` | Age in Gyr; log10(age in years) |
| `teff`, `radius`, `logg` | K; solar radii; log10 surface gravity in cgs |
| `distance`, `parallax` | kpc; mas |
| `kmag`, `kmag_abs` | Apparent and absolute K magnitudes; other bands follow the same suffix convention |
| `rho`, `rho_initial` | Current or initial mass divided by radius cubed, in solar density units |
| `dtdeep` | Separately interpolated centered age derivative, in Gyr per EEP |

Distance is needed only when requesting apparent magnitudes, parallax or
distance itself. No extinction is added. Tracks interpolate logarithmic age,
temperature and radius before exponentiating; the raw `age_gyr` field instead
interpolates physical age directly. Use `age` for the public age convention.

The immutable model and named results support JIT, gradients and broadcastable
coordinates. To evaluate a mass × EEP mesh, give the arrays complementary
dimensions, such as `mass_init=masses[:, None]` and `eep=eeps[None, :]`.
For large grids, pass the model as a PyTree argument to a compiled function:

```python
predict = jax.jit(lambda model, params: model(params, keys=("teff", "age")))
```

Missing track coverage is checked per output and returns `-inf`. Physical
predictions remain available in regions where only photometry is missing.
See [grid support](mistgrid.md#coverage-and-interpolation-support) for details.

## Fit observations with NUTS

Observations map output names to `(value, Gaussian_error)` pairs. Their keys
use the same units and initial/surface conventions as model predictions:

```python
fit = MistFit(model=model, observations={
    "teff": (5700., 70.),
    "feh": (0., .05),
    "kmag": (8., .03),
    "parallax": (10., .1),
})
mcmc = fit.run_nuts(
    jax.random.PRNGKey(0),
    num_warmup=1000, num_samples=1000,
    model_kwargs={"linear_age": True},
    output_keys=("rho", "kmag_abs"),
)
samples = mcmc.get_samples(group_by_chain=True)
mcmc.print_summary()
```

`model_kwargs` holds scientific model options; sampler options such as
`num_chains`, `chain_method` and `target_accept_prob` go directly to `run_nuts`.
The default is four parallel chains. With too few devices, NumPyro warns and
runs them sequentially. Each call returns a new NumPyro `MCMC`.
Add required saved quantities through `output_keys` for that run.

The physical model is available as `fit.physical_model`, and `fit.model` is its
default NumPyro probability model. To change observations, construct another
`MistFit` and reuse the physical model. Parallel lists can also be supplied as
`MistObservations(keys, values, errors)`.

### Track priors and bounds

The sampled coordinates are `mass` (initial mass), `feh_init`, `eep` and
`distance`. `mass_init` is also saved as an alias. Default choices are:

| Option | Default and effect |
| --- | --- |
| `massmin`, `massmax` | Grid endpoints; hard initial-mass bounds |
| `fmin`, `fmax` | Grid endpoints; initial-FeH bounds |
| `eepmin`, `eepmax` | Grid endpoints; a fixed EEP interval |
| `linear_age=True` | Weight the EEP density by `dtdeep`; `False` uses `dtdeep / (age * ln(10))` |
| `age_min=None`, `age_max=13.8` | Optional soft age bounds in Gyr |
| `age_bound_width=0.1`, `age_bound_depth=30` | Width in Gyr and depth of each soft-bound log penalty |
| `dist_scale=1.35` | Distance prior `Gamma(3, rate=1 / dist_scale)`, in kpc |

Mass, FeH and EEP start with Uniform densities on their bounds. Explicit bounds
must lie inside the loaded track grid. The age weighting uses `dtdeep`, computed
from centered finite differences of native physical ages and interpolated as a
separate grid field. The resulting marginal mass and age priors also depend on
the available age span within the selected EEP interval and grid coverage.

The default age limit is **soft**: each bound contributes −15 to log density
at its boundary and approaches −30 outside. Ages above 13.8 Gyr retain nonzero
density, and the sampled EEP interval stays fixed. Set `age_max=None` to disable
the upper penalty, or supply `age_min` to add a lower one.

`nodata=True` disables the main Gaussian observations while retaining the
priors, age penalties and any explicitly supplied extra constraints. Optional
constraints include `age_obs`/`age_err`, `rho_obs`/`rho_err`, and rotation-period
`prot`/`prot_err`. The `rho_obs` constraint uses `rho_initial`: **initial** mass
divided by radius cubed, in solar density units.

Unsupported age/derivative or likelihood values have zero density. An unavailable
saved-only output can be `-inf` without changing the inference target.

### Initialization

`run_nuts` performs a grid search by default. At fixed initial
FeH = 0 and distance = 1 / observed positive parallax, it searches native
mass × EEP nodes within the run's bounds and scores the joint log density,
including priors and optional constraints. It uses all EEP nodes. The best
candidate with finite density and gradients initializes all chains; all sampled
parameters are then free in NUTS. This does not guarantee finding every mode.

```python
options = {"linear_age": True, "age_max": 13.8}
initial = fit.find_initial(model_kwargs=options)
mcmc = fit.run_nuts(
    jax.random.PRNGKey(1), model_kwargs=options,
    init_kwargs={"feh_init": -0.1, "distance": .15},  # distance in kpc
    num_warmup=1000, num_samples=1000,
)
```

`init_kwargs` changes the quantities fixed during the search only. Choose
`feh_init` inside the configured FeH bounds. Without a positive observed
parallax, provide a starting distance or explicit initialization. An explicit
`init_strategy` or unconstrained `init_params` takes precedence and bypasses the
search. Selected starting values are included in the run metadata.

`grid_init=False` disables the search and initializes matching latent sites from
the observations, using NumPyro's fallback for unspecified sites. This option
requires a parallax observation: distance starts at its reciprocal when positive,
or at 8 kpc otherwise. The standalone `jaxstar.mistfit.run_nuts` function uses
NumPyro's default initialization unless an `init_strategy` is supplied.

## Save results and inspect diagnostics

Plotting and result-storage dependencies are included in the standard
installation described in the [README](../README.md#installation).
Convert a sampler result once when using it for several operations:

```python
from jaxstar.mistfit import load_result, plot_diagnostics, summarize, get_run_metadata

idata = fit.to_inference_data(mcmc)
result_dir = fit.save_result(idata, "results/star")
figures = fit.plot_diagnostics(
    idata, keys=("mass", "radius", "age"), output_dir=result_dir,
)

# In a later session, trace/corner plots and summaries need no grid.
loaded = load_result("results/star")
figures = plot_diagnostics(loaded, keys=("mass", "radius", "age"))
summary = summarize(loaded)
settings = get_run_metadata(loaded)
```

`save_result` writes `posterior.nc` (ArviZ result data), `run.json` and
`summary.csv` (mean, standard deviation, 68% HDI and MCMC Rhat/ESS).
`to_inference_data` and `load_result` return `InferenceData` with ArviZ 0.x,
or `xarray.DataTree` with ArviZ 1.x. Conversion, summaries and trace plots use
the installed version's API. The summary keeps the requested HDI probability
in both versions; column labels follow ArviZ's naming conventions.
The run record includes observations, configuration, RNG information and model
provenance. Saved results support later analysis and plotting; they contain
neither sampler state nor the implementations of custom callables.

Diagnostics return a dictionary of Matplotlib figures and optionally write
`trace.png`, `corner.png` and `tracks.png`. Existing result or plot files require
`overwrite=True`; use a new directory for each run. Trace/corner keys must be
saved outputs. Individual `fit.plot_trace` and `fit.plot_corner` calls are also
available. NUTS preserves chain/draw dimensions and, by default, divergence
flags, potential energy, acceptance probability, leapfrog counts and step sizes.

### Plot tracks and posterior uncertainty

Plot a track at fixed initial mass and initial FeH, with either an EEP interval
or an age interval in Gyr:

```python
from jaxstar.mistfit import plot_track, plot_posterior_tracks

fig = plot_track(model, mass_init=1.1, feh_init=0.,
                 x="teff", y="kmag_abs", age_range=(0., 13.8))
fig = fit.plot_track(1.1, 0., distance=.2, y="kmag", eep_range=(250., 500.))
```

Single-track plots default to the full grid EEP extent, independently of the
inference age bounds. Apparent magnitudes need distance in kpc. Axes accept
scalar output keys; `ax=...` allows overlaid tracks. Missing support creates
gaps, and temperature/magnitude axes use HR-diagram directions.

For track fits, `fit.plot_diagnostics(idata)` adds a posterior track figure when
there are observed magnitude bands. The default is one panel per observed
magnitude versus Teff, with 20 joint posterior draws, their sampled positions,
and matching observations with error bars. Each curve uses its draw's initial
mass, initial FeH and distance. Selection is reproducible and shared across bands.

The default drawing interval uses the nearest **primary EEP boundaries**
enclosing the minimum and maximum EEP across all posterior draws, then clips
to grid coverage. An explicit `eep_range` or `age_range` takes precedence.
`auto_eep_range=False`, or a grid without primary EEP metadata, gives the full
grid extent. These limits affect only the figure.

```python
fig = fit.plot_posterior_tracks(idata, y="kmag", n_samples=20, seed=0)
figures = fit.plot_diagnostics(
    idata, track_kwargs={"x": "teff", "y": ("jmag", "kmag"),
                         "age_range": (0., 13.8)},
)
fig = fit.plot_posterior_tracks(idata, y="kmag", auto_eep_range=False)
# Reloaded results need a matching model to recompute tracks.
fig = plot_posterior_tracks(loaded, model, y="kmag")
```

Track axes are recomputed from the grid and need not have been saved as outputs.
Use `show_tracks=False` to omit track recomputation from diagnostics.

## Isochrone fitting

Isochrone grids use `logage` (log10 years), `feh_init` and `eep` as coordinates,
with initial mass as a predicted quantity:

```python
iso_model = MistModel.from_isochrones()  # cached grid; generated on first use
prediction = iso_model(
    {"logage": 9.3, "feh_init": 0., "eep": 350., "distance": .1},
    keys=("mass_init", "teff", "kmag"),
)
iso_fit = MistFit(model=iso_model, observations=fit.observations)
iso_mcmc = iso_fit.run_nuts(
    jax.random.PRNGKey(0), num_warmup=1000, num_samples=1000,
    model_kwargs={"linear_age": True, "flat_age_marginal": False},
)
```

The loader also accepts `path=...`; see [isochrone data](mistgrid.md#isochrone-grids)
for cache selection. `MistFit(observations=...)` without an explicit model
loads an isochrone model. Track fitting requires `model=MistModel.from_tracks(...)`.
The [isochrone notebook](../examples/mistfit/fit_isochrone.ipynb) demonstrates
sampling and diagnostics.

Isochrone inference uses `dmdeep`, the separately interpolated centered
initial-mass derivative. `linear_age` chooses linear/log-age sampling;
`flat_age_marginal` optionally corrects for the available mass span. Age bounds
use `logamin`/`logamax` (log10 years), and `massmin`/`massmax` apply soft mass
penalties. These differ from the track model's age and mass bounds.
Automatic initialization searches native log-age × EEP at fixed FeH/distance.
Saving and diagnostics use the same API, with posterior track plots reserved
for track models.

## Optional MultiNest backend

Both model kinds can use MultiNest with the same scientific options:

```python
result_dir = "results/star_multinest"
runner = fit.run_multinest(
    outputfiles_basename=result_dir + "/run-", n_live_points=800, resume=False,
    model_kwargs={"linear_age": True}, output_keys=("rho", "kmag_abs"),
)
idata = fit.to_inference_data(runner)
fit.save_result(idata, result_dir)
figures = fit.plot_diagnostics(idata, output_dir=result_dir)
```

This path requires `numpyro-inferutils`, `pymultinest` and the native MultiNest
library. `outputfiles_basename` sets the filename prefix: the example writes
files such as `results/star_multinest/run-stats.dat`. Use a short relative prefix
because older MultiNest libraries truncate filenames at 100 characters, and
choose a fresh directory for each independent run.

Gamma prior sites, including the default distance prior, use SciPy's inverse
CDF in the CPU MultiNest callback and do not require TensorFlow Probability.
User-supplied `site_overrides` take precedence over these automatic transforms.

Conversion reconstructs deterministic outputs from the equal-weight posterior
once. MultiNest diagnostics omit MCMC traces and Rhat/ESS, while metadata records
`logZ`/`logZerr`. Saving keeps native MultiNest files in the result directory,
or copies them when saving elsewhere; files replaced by another run are rejected.
Use jaxstar's execution functions for each recorded run so provenance stays current.

## Optional correction utilities

`jaxstar.utils.correction` provides Bayestar extinction and Gaia EDR3 parallax
zero-point utilities using optional [dustmaps](https://dustmaps.readthedocs.io/en/latest/)
and [gaiadr3-zeropoint](https://pypi.org/project/gaiadr3-zeropoint/) dependencies:

```bash
python -m pip install dustmaps gaiadr3-zeropoint
```

These utilities are separate from the extinction-free `MistModel` predictions.
