# SpecFit guide

[README](../README.md) · [MistFit](mistfit.md) · [MIST grids](mistgrid.md)

`jaxstar.specfit` fits observed stellar spectra with prepared atmosphere grids.
The workflow below starts from your own observed arrays and uses Coelho grids
as an example. The [Coelho notebook](../examples/specfit/coelho.ipynb) demonstrates
the same steps with a concrete dataset.

**Observation → grid selection → CCF → model and priors → SVI → outlier mask
and refit → NUTS → plots and saved results.**

`SpecFit` holds the observation, physical spectral model and fit mask. In this
guide, `sf` is the fitter, `specmodel` is its deterministic `SpecModel`, and
`model` is the configured NumPyro model passed to inference and prediction.
The [reference](#reference) below covers grid preparation, physical models,
continuum and priors, diagnostics, result storage and custom models.

## Basic workflow

### 1. Prepare your observed arrays

Install the current checkout with `pip install -e .` from the repository root.
Run the following cells in a fresh Jupyter kernel. Read your spectrum in your
own format and prepare these NumPy arrays:

| Array | Contents |
| --- | --- |
| `wav_obs` | Wavelength in Å, in the same air/vacuum convention as the model grid |
| `flux_obs` | Observed flux; continuum-normalized for the normalized grids used here |
| `error_obs` | Per-pixel standard deviation, in the same units as `flux_obs` |
| `mask_obs` (optional) | Boolean mask, with `True` for pixels to exclude; omit to use all pixels |

All arrays must have the same shape: `(n_pixels,)` for one region, or
`(n_regions, n_pixels)` for several regions. A region can be a spectral order
or any wavelength segment. Wavelengths must be finite, positive and strictly
increasing within each region, including masked pixels. Usable flux and errors
must be finite, with positive errors.

```python
import numpyro

num_chains = 2
numpyro.set_host_device_count(num_chains)  # before JAX initializes CPU devices

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from numpyro.infer import init_to_value
from jaxstar.specfit import Observation, SpecFit
from jaxstar.specfit.numpyro_model import make_model_single

# Supply wav_obs, flux_obs, error_obs and mask_obs from your own data.
# If there are no existing flags: mask_obs = np.zeros_like(flux_obs, dtype=bool)
mask_obs = np.asarray(mask_obs, dtype=bool).copy()
mask_obs |= (
    ~np.isfinite(flux_obs) | ~np.isfinite(error_obs) | (error_obs <= 0)
)
obs = Observation(wav_obs, flux_obs, error_obs, mask=mask_obs)
```

The mask above also excludes invalid flux/error values. If all pixels are
usable, you can omit `mask` when constructing `Observation`. Optionally pass
`region=region_labels` with one string or integer label per region; these labels
appear in plots and diagnostics.

### 2. Select the grids and check the CCF

Set `gridpath` to your local directory of Coelho `*_normed.npz` grids covering
the observed wavelength regions. `SpecFit.from_grid` reads these files without
downloading them.

```python
gridpath = Path("/path/to/coelho_grids")  # replace with your grid directory
sf = SpecFit.from_grid(
    gridpath, obs, library="coelho", pattern="*_normed.npz",
    wav_margin=3.0, pixels=5000, vmax=50.0, flux_kind="normalized",
)
specmodel = sf.specmodel
template_atmosphere = {"teff": 6000., "logg": 4., "feh": 0., "alpha": 0.}
ccf_rv, ccf_width = sf.check_ccf(
    template_atmosphere, v_limit=100., ccfvmax=100., plot=True,
)
plt.show()
```

`from_grid` selects one covering NPZ per region from its stored `wavgrid`, then
resamples to log-uniform wavelengths. `wav_margin=3` requires at least 3 Å on
each side of the observed interval. `pixels=5000` sets the model sampling;
`vmax=50` sets the broadening kernel half-width in km/s. Instrumental resolving
power is specified separately below. The files must already match the observed
wavelength medium and normalization. See [grid selection](#select-legacy-grids-for-an-observation)
to inspect selected paths or construct a `SpecModel` separately.

The CCF returns the median region RV and the combined CCF FWHM, both in km/s.
The model factory uses them to set default RV and rotation priors. The width
is a guide for the rotation prior, not a direct `vsini` measurement.

### 3. Configure the model and inspect priors

```python
model = make_model_single(
    sf, resolving_power=np.full(obs.n_regions, 70000.), degree=4,
    use_gp=True, gp_solver="quasisep", use_physical_logg_max=True,
    use_empirical_vmic=False, use_empirical_vmacro=False,
)
display(model.check_prior())
print("RV coverage bounds [km/s]:", sf.rv_bounds)
```

This uses a fourth-degree Chebyshev continuum, analytically marginalized under
Gaussian coefficient priors, and a Matérn-3/2 residual GP. Resolving power is
fixed at 70000 in every region in this example; set it for your instrument.
The temperature-dependent gravity ceiling is enabled here as in the notebook;
the factory checks its support against the grid. RV priors are not clipped to
`sf.rv_bounds`, so check the prior table
against the available wavelength coverage.

Use `atmosphere_bounds={"teff": (5000., 6500.)}` for a partial bounds override
or `fixed={"logg": 4.4, "vmacro": 0.}` to fix sites. Rebuild `model` to change
its settings. [Priors and model configuration](#priors-and-model-configuration)
describes the defaults and options.

### 4. Run SVI, update the outlier mask and refit

```python
p_fit = sf.run_svi(
    rng_key=jax.random.PRNGKey(0), model=model, step_size=0.1, num_steps=2000,
)
point_figures = sf.plot_models(point=p_fit, model=model, show_gp=True)
display(pd.DataFrame(sf.residual_diagnostics(point=p_fit, model=model)))
plt.show()

sf.mask_outliers(point=p_fit, model=model, sigma_threshold=3., plot=True)
display(pd.DataFrame(sf.mask_summary()))
plt.show()

p_fit = sf.run_svi(
    rng_key=jax.random.PRNGKey(1), model=model, step_size=0.1, num_steps=1000,
    p_initial=p_fit,
)
```

The first SVI run initializes from the priors; `p_fit` contains native
sample-site values that can be reused directly. The learning rate `0.1` is the
notebook's explicit choice; `run_svi` defaults to `0.01`.

Outlier detection uses the physical spectrum times its conditional continuum
(`continuum_flux`), excluding the GP mean. It preserves the fixed observation
mask and updates a separate fit mask. The second SVI run starts from the first
point with the updated mask. Reuse the same configured model: each inference
call binds the current masked observation.

### 5. Sample with NUTS

Keep the mask and model settings fixed after the refit, and pass the native
point directly as the NUTS initialization:

```python
mcmc = sf.run_nuts(
    rng_key=jax.random.PRNGKey(2), model=model,
    num_warmup=500, num_samples=1000, num_chains=num_chains,
    nuts_kwargs={
        "init_strategy": init_to_value(values=p_fit),
        "target_accept_prob": 0.90, "dense_mass": True, "max_tree_depth": 6,
    },
)
mcmc.print_summary(exclude_deterministic=True)
samples = mcmc.get_samples()
```

These are the notebook's settings: two chains with 500 warmup and 1,000 retained
draws each. Inspect convergence, effective sample sizes and divergences, along
with the plots below, and adjust the run length or sampler settings as needed.

### 6. Plot posterior spectra and MCMC diagnostics

```python
model_figures = sf.plot_models(
    samples=samples, model=model, show_gp=True, max_draws=50,
)
mcmc_figures = sf.plot_mcmc_diagnostics(
    mcmc,
    corner_vars=["teff", "logg", "feh", "alpha", "vsini", "vmacro",
                 "gp_amplitude", "gp_scale", "jitter"],
    log_vars=["gp_amplitude", "gp_scale", "jitter"],
    labels={"vmacro": "zeta", "gp_amplitude": "lna",
            "gp_scale": "lnc", "jitter": "lnsigma"},
)
plt.show()
```

`plot_models` predicts at up to 50 posterior draws and averages the curves.
Each region gets a figure showing the continuum-corrected physical spectrum
above and residuals with the GP mean below. Its return value is a list of
`(figure, axes)` pairs in region order. Use
`sf.predict(samples=samples, model=model, max_draws=50)` to obtain the arrays.

The diagnostic helper returns a dictionary with `trace` and `corner` figures.
The trace includes every sampled site by default, including each region's RV;
`corner_vars` selects the corner variables independently. Logs and labels only
change the displayed values.

### 7. Save and reload results

```python
from jaxstar.specfit import save_fit_result, load_fit_result, plot_mcmc_diagnostics

result_dir = Path("results/target")  # choose an output directory
save_fit_result(
    result_dir, sf=sf, model=model, point=p_fit, mcmc=mcmc,
    overwrite=False,
)

figure_dir = result_dir / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)
for region, (fig, axes) in zip(sf.regions, model_figures):
    fig.savefig(figure_dir / f"model_region{region}.pdf", bbox_inches="tight")
for name, fig in mcmc_figures.items():
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")

result = load_fit_result(result_dir)
saved_diagnostics = plot_mcmc_diagnostics(
    result, corner_vars=["teff", "logg", "feh", "vsini"],
)
plt.show()
```

The bundle contains the point, posterior and diagnostics, observation, masks
and model/grid metadata in `arrays.npz`, `posterior.nc` and `result.json`.
Omit `mcmc` for a point-only bundle; use `overwrite=True` to replace a saved
result. The PDF loop saves already drawn figures without recomputing predictions.

Loading and plotting traces/corners need no spectral grid. `result.point`
holds the point, `result.samples` flattens chain/draw for prediction, and
`result.idata` retains those dimensions for diagnostics. Spectral predictions
require the original physical and configured models, plus restoration of the
saved fit mask; see [Save and reload fit results](#save-and-reload-fit-results).

## Reference

The following sections describe the individual components and their options.
Grid preparation and direct `SpecModel` evaluation can also be used separately
from the fitting workflow. Use these snippets as individual examples when
adapting the basic workflow.

### Prepare and load spectra

The spectral model uses a prepared grid with atmosphere coordinates and
log-uniform wavelength sampling. Prepare it once, save it, and load the common
NPZ in later sessions:

| Source | Raw-data preparation | jaxspec NPZ loader | Atmosphere axes |
| --- | --- | --- | --- |
| Coelho | `prepare_coelho` | `load_coelho` | `teff`, `logg`, `feh`, `alpha` |
| BOSZ | `prepare_bosz` | `load_bosz` | `teff`, `logg`, `mh`, `alpha`, `carbon`, `vmic` |
| TLUSTY | `prepare_tlusty` | `load_tlusty` | `teff`, `logg`, `logZ` |

All functions above are exported from `jaxstar.specfit`. Raw preparers read
local library files. For fitting, use `sampling="log"` and an explicit
`velocity_step` in km/s; this controls numerical sampling independently of
instrumental resolving power and observed detector pixels. Prepare wavelength
regions wide enough for broadening and the requested RV range.

Use `save_spectral_grid(path, spectra)` to persist a prepared result.
`load_spectral_grid(path)` then needs only that artifact. The jaxspec NPZ loaders
preserve stored wavelength samples and flux normalization. Use
`resample_spectral_grid` to convert non-log-uniform samples before constructing
`SpecModel`, and ensure the wavelength medium and normalization match your data.

```python
import jax
from jaxstar.specfit import SpecModel, load_spectral_grid

jax.config.update("jax_enable_x64", True)
spectra = load_spectral_grid("coelho_log.npz")  # your prepared artifact
specmodel = SpecModel(spectra, vmax=50.)  # broadening kernel support in km/s
print(spectra.grid.axis_names)
```

#### Select legacy grids for an observation

For a directory of legacy jaxspec NPZ files containing `wavgrid`, use
`SpecFit.from_grid` to select files and build the physical model in one step:

```python
from jaxstar.specfit import Observation, SpecFit

obs = Observation(wav_obs, flux_obs, error_obs,
                  mask=mask_obs, region=region_labels)
sf = SpecFit.from_grid(
    gridpath, obs, library="coelho", pattern="*_normed.npz",
    wav_margin=3., pixels=5000, vmax=50., flux_kind="normalized",
)
specmodel = sf.specmodel
print(sf.rv_bounds)  # (n_region, 2), lower/upper effective RV in km/s
```

`library` selects the existing Coelho, BOSZ or TLUSTY loader. `wav_margin` is
the minimum required wavelength margin in Angstrom on **each** side of every
observation row, including masked pixels. Selection reads actual stored
`wavgrid` values; filenames only determine which files match `pattern`
(default `"*.npz"`). Zero or multiple covering candidates produce an error
listing the affected region and candidate coverages. A source file may serve
more than one observation region.

The selected full grid spans are resampled to log-uniform wavelengths using
exactly one explicit `pixels` or `velocity_step` setting. `pixels=5000` above
is a model-grid choice, independent of the detector pixel count. The existing
loader/resampler requirements still apply: shared atmosphere axes, compatible
dtypes and equal region pixel counts, including counts produced by a requested
velocity step. Wavelength medium and flux kind are caller declarations;
these factories do not convert air/vacuum wavelengths or normalize flux.
This entry point takes legacy NPZs, not raw atmosphere files or common-format
`save_spectral_grid` artifacts, and performs no downloads.

To prepare only the model, or inspect just the selected paths, use:

```python
from jaxstar.specfit import SpecModel, select_grid_paths

paths = select_grid_paths(
    gridpath, obs.wavelength, regions=obs.region,
    pattern="*_normed.npz", wav_margin=3.,
)
specmodel = SpecModel.from_grid(
    gridpath, obs.wavelength, regions=obs.region,
    library="coelho", pattern="*_normed.npz", wav_margin=3.,
    pixels=5000, vmax=50., flux_kind="normalized",
)
sf = SpecFit(obs, specmodel)
```

Both construction paths compute `sf.rv_bounds` once from the entire observed
wavelength interval and the default broadening kernel's remaining model
coverage, using the model's relativistic Doppler convention. The returned
array is read-only. Each row applies independently; a shared RV must satisfy
all rows. A lower bound greater than its upper bound indicates no RV can cover
that observation region. Exact endpoints are subject to floating-point
roundoff. This is coverage information, not a prior: fitting does not clip RVs
or modify priors, and model evaluation continues to reject insufficient
coverage. For a custom `broadening_operator`, `rv_bounds` is `None` because its
output wavelength range can depend on the parameters; evaluation still checks
the actual returned coverage.

#### CCF diagnostics

For Coelho, `sf.check_ccf()` uses a default atmosphere template with
`teff=5800` K, `logg=4.4`, `feh=0` and `alpha=0`. Override it by passing the
complete atmosphere mapping:

```python
rv, width = sf.check_ccf()
rv, width = sf.check_ccf({"teff": 6000., "logg": 4., "feh": 0., "alpha": 0.})
```

An omitted argument or `atmosphere=None` selects the default. The template
must be inside the loaded grid; no clipping is applied. BOSZ, TLUSTY and custom
libraries require an explicit mapping using their own atmosphere axis names.
The returned values are median region RV and combined CCF width in km/s;
diagnostic arrays are available in `sf.ccf`.

To display and save the CCF diagnostic:

```python
rv, width = sf.check_ccf(plot=True, save_path="ccf.pdf")
```

The figure shows each region's CCF and the combined median CCF, each normalized
to its own peak for display. It marks the median RV with a vertical line and
reports **Median RV** and **CCF FWHM** in km/s above the axes. The width is that
of the combined CCF, not a direct measurement of `vsini`. The displayed velocity
range is median RV +/- `ccfvmax`. The stored arrays and returned values are
unchanged by plotting.

`save_path` saves all regions together in one figure using the supplied filename
(no region suffix). Without an extension, Matplotlib's default save format is
used. `plot=False` is the default; with a `save_path`, the figure is still saved
and then closed. `plot=True` leaves it open for notebook display or `plt.show()`.

The following examples assume a normalized Coelho grid containing the stated
parameters and prior ranges. Adapt them to your library; spectra and observed
data are external inputs.

### Predict a physical spectrum

`SpecModel` interpolates the atmosphere, applies rotation, radial-tangential
macroturbulence and Gaussian instrumental broadening, then shifts by the
relativistic RV and samples the requested wavelengths:

```python
from jaxstar.specfit import single_star_params

params = single_star_params(
    {"teff": 5700., "logg": 4.4, "feh": 0., "alpha": 0.},
    vsini=3., vmacro=3., q1=.36, q2=.3, rv=0., resolving_power=70000.,
)
flux = specmodel(params, wav_obs)
```

`single_star_params` builds the nested component/instrument dictionary and
converts `q1`, `q2` to quadratic limb-darkening coefficients. Velocity parameters
are in km/s. `specmodel.intrinsic`, `specmodel.broadened` and `specmodel.full`
expose the successive physical stages; calling `specmodel` is equivalent to `full`.

Wavelength arrays have shape `(region, pixel)`; one region may use a 1D array.
Atmosphere parameters are scalar for each star. Broadening, RV and resolving
power may be shared scalars or one value per region. The prepared grid must
cover the full evaluation range after broadening and RV shifts; the model does
not add wavelength padding or extrapolate automatically.

For multiple stars, supply a tuple of `components` plus nonnegative
`flux_weights`. All components use the same spectral library. Optional
`dilution` is the featureless fraction of total continuum light:
`total = dilution + (1 - dilution) * sum(weight_i * flux_i) / sum(weight_i)`.
`specmodel.decompose` returns the individual and weighted spectra. The
[SpecModel tutorial](tutorials/specfit/specmodel.ipynb) includes SB2/SB3
examples. Use a custom NumPyro model to fit multiple stellar components.

### Fixed two-Gaussian instrumental profile

The example below uses `parameters.csv` generated from a ThAr spectrum with
[gaoesrvip](https://github.com/ryunosukeoishi/gaoesrvip):

```python
from gaoesrvip import analyze_thar_ip_2g

result = analyze_thar_ip_2g(
    thar=thar,
    wavelength_file=wavelength_file,
    output_dir="thar_ip_analysis_for_jaxstar/",
    orders=[104, 105],
    gain=1.935,
    read_noise=4.0,
    saturation=5000,
    warmup=1000,
    samples=1000,
    chains=2,
)
```

Use the resulting `thar_ip_analysis_for_jaxstar/parameters.csv` as the IP
calibration file, with the same wavelength calibration used for the science
spectrum.

An externally calibrated IP can replace the resolving-power Gaussian. Supply
the association between spectral regions and calibration orders yourself; the
labels need not match and the number of selected regions is arbitrary. For
example, if the prepared library has regions `("blue_part", "red_part")`:

```python
from jaxstar.specfit import read_ip_calibration, SpecModel, SpecFit, make_model_single

dir_name = "thar_ip_analysis_for_jaxstar/"
order_for_region = {"blue_part": 104, "red_part": 105}
region_orders = [order_for_region[region] for region in spectra.regions]
ip_calibration = read_ip_calibration(dir_name, orders=region_orders)
specmodel = SpecModel(spectra, vmax=50., fixed_ip=ip_calibration.median)
sf = SpecFit(obs, specmodel)
model = make_model_single(sf, rv_bounds=(-5., 5.), vsini_max=15., use_gp=False)
```

The reader opens exactly `dir_name/parameters.csv`. Required columns are
`order, parameter, median`; each selected order must have one row for each of
`mu1_kms`, `mu2_kms`, `sigma1_kms`, `sigma2_kms`, and `fraction2`. It returns
read-only `median` arrays in the supplied order, plus `orders` and the `source`
path. CSV row order is irrelevant. An explicit sequence such as
`orders=[105, 104, 104]` is also valid: the last two spectral regions share the
same order's IP. Missing or duplicate selected entries raise an error.
Additional columns (including `p16`/`p84`) and derived parameter rows (`R`,
`delta_kms`, `centroid_kms`, `intrinsic_fwhm_kms`) are ignored.

The fixed instrumental kernel is
`(1 - fraction2) * G(mu1_kms, sigma1_kms) + fraction2 * G(mu2_kms, sigma2_kms)`.
Each Gaussian has unit integrated area, so `fraction2` is the **area fraction of
component 2**, not its peak height. Shifts and standard deviations are in km/s;
sigmas must be positive and the fraction must be in `[0, 1]`. Positive shifts
move spectral lines redward in the log-wavelength grid. Shifts refer to the
wavelength calibration used for the spectrum; the IP centroid is retained.

`fixed_ip` also accepts a manually constructed mapping of these five names to
scalars or arrays of length `n_region`. Values are copied at construction, and
the instrumental Fourier factor is precomputed. The same IP is applied to all
stellar components in each region. `SpecModel.from_grid` and `SpecFit.from_grid`
also accept `fixed_ip` in the selected region order.

With fixed IP, omit `resolving_power` in `single_star_params`, `model_single`,
and `make_model_single`; physical params can omit `instrument` or supply `{}`.
Specifying both modes raises an error, as does combining `fixed_ip` with a
custom `broadening_operator`. No IP sample sites or priors are added. Omitting
`fixed_ip` preserves the existing resolving-power mode, including sampled R.

Choose model velocity sampling fine enough to resolve the IP. Each active
Gaussian must fit `abs(mu) + 5*sigma` within the actual kernel half-support;
increase `vmax` if this check fails. Allow additional room for stellar rotation
and macroturbulence, and check convergence as support/sampling change. Prepared
wavelength coverage must still accommodate the kernel and stellar RV.

The [Coelho IP comparison notebook](../examples/specfit/coelho_ip_comparison.ipynb)
uses the included two-region grid and sample calibration to compare line shapes
at R=65,000 versus fixed IP, then fits the same synthetic observation under both
assumptions, including atmosphere parameters and broadening. The current
settings use S/N=200; change `SNR` to choose a different noise level.
Outputs are saved under `results/ip_comparison/`, with a new `run_*` directory
per execution.

### Observations and continuum

Observed arrays must have identical 1D or 2D shapes and use the same region
order and wavelength convention as the prepared model:

```python
from jaxstar.specfit import Observation, chebyshev_basis

obs = Observation(wav_obs, flux_obs, error_obs, mask=mask_obs)
basis = chebyshev_basis(obs.wavelength, degree=4)
```

`mask=True` excludes a pixel; omit the mask to use every pixel. Wavelengths
must remain finite, positive and increasing even at masked pixels. Usable flux
and errors must be finite, with positive errors. Optional region labels must
agree with the library's row order when used with `model_single`.
Use `region` to label each row, whether it is a whole spectral order or a
wavelength segment. `SpecFit(obs, specmodel)` exposes these labels as `sf.regions`
and uses them in diagnostics and plots; if `obs.region` is omitted, it uses
`specmodel.spectra.regions`. There is no separate order label or display-label
argument.

A multiplicative Chebyshev continuum is applied after the physical model.
Gaussian coefficient priors shrink the constant toward one and the other
coefficients toward zero. The likelihood integrates these coefficients
analytically. `sigma_constant` and `sigma_continuum` set the two prior scales.
Jitter is in absolute flux units; the total noise variance is
`error_obs**2 + jitter**2`.

At a chosen physical model and noise setting, recover the conditional continuum:

```python
from jaxstar.specfit import continuum_posterior, apply_continuum

physical_flux = specmodel(params, obs.wavelength)
continuum = continuum_posterior(
    obs, physical_flux, basis=basis,
    sigma_constant=.1, sigma_continuum=.03, jitter=.01,
)
fitted_flux = apply_continuum(physical_flux, basis, continuum.mean)
```

The [continuum tutorial](tutorials/specfit/continuum.ipynb) is a synthetic
example requiring no external spectra. For custom models,
`marginalized_continuum_log_likelihood` exposes the normalized Gaussian
likelihood directly.

### Priors and model configuration

Use `make_model_single` to prepare a configured model from the loaded grid and
CCF. The factory prepares bounds and fixed settings; `model_single` defines
priors and likelihood in ordinary NumPyro code that can serve as a custom-model
example. Each call returns a new `ConfiguredModel`; nothing is stored on the fit:

```python
from jaxstar.specfit.numpyro_model import make_model_single

sf.check_ccf()  # Coelho default; other libraries require an atmosphere mapping
model = make_model_single(
    sf, resolving_power=70000., degree=4, use_gp=True,
    atmosphere_bounds={"teff": (5400., 6000.)},  # optional partial overrides
    fixed={"vmacro": 3.},                       # optional numeric fixed sites
)
model.check_prior()  # Compact table of parameters and priors/fixed values
```

`run_svi`, `run_nuts` and `predict` require `model=model`; there is no separate
`priors` or `model_kwargs` argument. Inference accepts any callable of
`(observation, specmodel)`, including an ordinary custom function or partial.
Prediction requires a configured model with its own `predict` method.
Every run uses the current fit mask. A factory-created model is associated with
its original `SpecModel` and rejects a different one, preventing inference and
reconstruction from silently using different physics. `model.settings` contains
copied, read-only bounds/settings (including resolved defaults); rebuild the
model to change them. `ConfiguredModel` takes omitted keyword defaults from the
model function's signature when it is created; explicit settings take precedence.
Reconstruction reads these resolved settings without supplying separate defaults.
CCF defaults are captured when the model is created.

`run_svi` draws its initial values from the model's priors when `p_initial` is
omitted. Explicit initial values remain available. Reuse the same model for
refitting after mask changes and for NUTS/prediction.

| Parameter | Default |
| --- | --- |
| Atmosphere | Loaded axis names; Uniform over endpoints, singleton axes fixed |
| RV | Region-vector Uniform centered on median CCF RV; half-width 5 km/s if CCF width <20 km/s, otherwise half its width |
| `vsini` | Shared Uniform(0, CCF width), capped at 90% of default kernel support |
| `vmacro` | Shared Uniform(1, 15) km/s unless an empirical prior is requested |
| `q1`, `q2` | Shared Uniform(0, 1) each |
| Continuum | `degree=4`, fixed coefficient scales `0.2/sqrt(3)` and `0.1/sqrt(3)` |
| Jitter | Shared LogUniform(exp(-10), exp(-3)), in flux units |
| GP, enabled by default | Shared amplitude LogUniform(exp(-5), exp(-0.5)), in flux units; scale LogUniform(exp(-5), exp(2)), in Angstrom; `gp_solver="auto"` |

The factory accepts `atmosphere_bounds`, `rv_bounds`, `vsini_max`,
`vmacro_bounds`, `jitter_bounds`, `gp_amplitude_bounds` and `gp_scale_bounds`.
Bounds are in physical units; omitted atmosphere axes keep their grid bounds.
`rv_bounds=(lower, upper)` broadcasts scalar endpoints to all regions, or accepts
one lower/upper endpoint per region. `fixed` maps native site names to numeric
values. Atmosphere and GP sites are scalar; broadening, RV and jitter can be
scalar or region vectors. Resolving power is required as a positive scalar,
region vector or NumPyro distribution; infinity disables Gaussian IP. For a
`SpecModel` with `fixed_ip`, omit resolving power instead. CCF is
unnecessary when RV and vsini are both supplied explicitly through bounds or
fixed values. The continuum/noise defaults assume normalized flux.

RV priors are not clipped to `sf.rv_bounds`; spectrum evaluation still checks
coverage. `use_empirical_vmic=None` enables BOSZ's deterministic relation;
`False` makes vmic explicit. `use_empirical_vmacro=True` selects the conditional
prior with `vmacro_empirical_sigma` (default 1 km/s). Fixed values for active
empirical vmic/vmacro conflict with those options.

`use_physical_logg_max=True` intersects the chosen Teff bounds with 4500--7000 K
and checks gravity support against the grid. Fixed Teff must already satisfy
this interval. Empty support or incompatible gravity ranges raise errors.
The conditional gravity ceiling replaces the Uniform's upper bound and retains
its lower bound; it is not clipped. Other empirical relations remain unclipped.
These setup checks belong to the factory; direct model calls must choose
compatible support.

`model.check_prior()` executes the model's parameter sites with `prior_only=True`,
without spectra or likelihoods. It returns a compact DataFrame indexed by
parameter, with a `prior` column such as `Uniform(5000, 6500)` or `0 (fixed)`.
Conditional rows are labeled `logg | teff` and show their distribution's
expression. An additional `conditional` column shows numerical values only
when dependencies are fixed or explicitly supplied:

```python
model.check_prior(condition={"teff": 5800.})
```

This displays gravity bounds at that temperature, not an unconditional gravity
prior. The expression/dependency annotations describe the model; numerical
parameters come from the actual NumPyro trace. `model.prior_trace(rng_key=...)`
provides the underlying sites and prior draws for more detailed inspection.

### Outlier plots

For an outlier-specific quicklook, enable plotting when updating the fit mask:

```python
sf.mask_outliers(point=p_fit, model=model, sigma_threshold=3., plot=True)
```

`mask_outliers` calls `predict` once using the current mask, before updating it.
It uses `continuum_flux` as the baseline and the largest component `vsini` for
the median-filter width; `mask_v` can override that velocity in km/s. Previously
flagged pixels are reconsidered, so they can return to the sf. Custom model
predictions must include `continuum_flux`, plus physical `params` unless
`mask_v` is supplied. Pass a parameter point and the model, not a precomputed
prediction dictionary.

`plot=True` creates three panels on the left for each region: data and
`continuum_flux` (excluding the GP mean); residuals and their median-filter trend;
and detrended residuals divided by `1.4826 * MAD`, with the positive and negative
clipping thresholds. A histogram on the right shows the same detrended quantity,
with retained and excluded pixels in separate colors, logarithmic counts and
the same thresholds. A standard-normal `N(0, 1)` reference is overlaid, scaled by
the number of pixels outside the fixed mask before clipping and the bin width.
Above it, annotations show `1.4826 * MAD`, the median
observed uncertainty (both in flux units) and their ratio. Both scales use all
pixels outside the fixed mask, before this clipping pass, without adding jitter
to the observed uncertainty. The plots use the quantities from that same
masking pass.
Highlighted points show the final fit mask, including any mask extension, so
some highlighted neighbors can lie inside the thresholds. Fixed masked pixels
are excluded. If the MAD scale is zero, the last panel and histogram show
unscaled detrended residuals and the zero threshold, with an explanatory label;
the standard-normal reference is omitted in that case.

`plot` defaults to False. The return value remains the updated Boolean fit mask;
figures are left open for notebook display or `plt.show()`.

To save one file per region, add `save_path`:

```python
sf.mask_outliers(point=p_fit, model=model, sigma_threshold=3.,
                  plot=True, save_path="outliers.pdf")
```

This saves `outliers_region8.pdf`, `outliers_region9.pdf`, etc., with the same
filename conventions as `plot_models`. Without an extension, Matplotlib's
default save format is used. With `plot=False` (the default), `save_path` still
creates and saves the figures, then closes them; use `plot=True` to keep them
open as well. Saving uses the diagnostics from the same masking pass.

### Mask summary and residual diagnostics

Both helpers return one dictionary per region. Format and display the results
in the notebook:

```python
import pandas as pd

display(pd.DataFrame(sf.mask_summary()))
display(pd.DataFrame(sf.residual_diagnostics(point=p_fit, model=model)))
display(pd.DataFrame(sf.residual_diagnostics(
    point=p_fit, model=model, baseline="continuum_flux",
)))
```

`mask_summary()` reports `region`, `fixed_mask`, `fit_mask` and `usable`.
The mask counts are separate and can overlap; `usable` counts pixels outside
their union, `effective_mask`.

`residual_diagnostics` calls `predict` at the supplied point using the current
effective mask. The default `baseline="mean_flux"` includes the GP conditional
mean; `baseline="continuum_flux"` uses the physical spectrum multiplied by its
continuum, before adding the GP. Jitter comes from the same prediction and
defaults to zero when omitted by a custom reconstruction callback.

Statistics use only usable pixels. Standardized residuals divide by
`sqrt(error_obs**2 + jitter**2)`; they are not whitened by the GP covariance.
Lag-1 correlation uses pairs adjacent in the original wavelength array with
both pixels usable, without bridging masked gaps. The return fields remain
`region`, `n_usable`, `rms`, `mad_scale`, `standardized_rms`, `n_gt_3sigma`,
`n_gt_5sigma`, `n_adjacent_pairs` and `lag1_correlation`.

### Predictions and plots

`predict` accepts exactly one of `point` or `samples`, together with `model`.
Predictions use the model's paired reconstruction callback and the current fit
mask. A point returns the full callback result, including the GP conditional
mean when present:

```python
prediction = sf.predict(point=p_fit, model=model)
gp_mean = prediction["gp_mean"]  # when the model includes a GP
```

For posterior draws, pass the dictionary returned by `mcmc.get_samples()` with
its flattened leading draw axis:

```python
prediction = sf.predict(samples=mcmc.get_samples(), model=model, max_draws=50)
```

This predicts at up to 50 evenly spaced draws, then averages the returned curves
(`physical_flux`, `continuum`, `continuum_flux`, `mean_flux`, `gp_mean`, where
provided). `max_draws=None` uses all draws. The result also includes
`draw_indices`; physical parameter dictionaries, conditional covariance objects
and jitter are not averaged. In particular, GP means are computed at each draw
before averaging. When `continuum_flux` is available, the result also includes
`continuum_flux_quantiles` and `gp_mean_quantiles`: pointwise 16th and 84th
percentiles of `continuum_flux` and `mean_flux - continuum_flux`, respectively,
across the same draws, with shape `(2, *observation.shape)`. The GP difference
is computed within each draw before taking percentiles. These describe the
spread of conditional means across posterior draws. They exclude the conditional
continuum/GP uncertainty remaining within each draw and fresh observational
noise (including jitter); they are not a full posterior predictive interval.

`plot_models` calls `predict` internally:

```python
plots = sf.plot_models(point=p_fit, model=model, show_gp=True)
plots = sf.plot_models(samples=mcmc.get_samples(), model=model,
                       show_gp=True, show_interval=True, max_draws=50)
```

It returns `[(fig, axes), ...]` in region order, with one figure per region and
two axes per figure. Flux and residual panels share their x axis and have a
small vertical gap; only the lower panel shows x tick labels. The x limits match
the region's wavelength endpoints. A common legend sits vertically to the right
of the panels. Observed points use open gray circles labeled `data`, matching
the outlier check; mask entries are omitted when that region has no such masked pixels.

`save_path="models.png"` saves one file per region, such as
`models_region8.png` and `models_region9.png`. The suffix is added even for one
region; special characters in region labels are URL-escaped for filenames.
Without a filename extension, Matplotlib's default save format is used.

The upper panel shows data and the physical model including its continuum
prediction (`continuum_flux`). Bands are hidden by default (`show_interval=False`).
With `samples` and `show_interval=True`, it shades the pointwise 16th-84th
percentile range of the conditional means described above. The lower
shows `data - continuum_flux`, subtracting the mean curve for `samples`. Its
zero line uses C0 to match the continuum model in the upper panel.
`show_gp=True` overlays `mean_flux - continuum_flux` in C1 in the lower panel,
with its own 16th-84th percentile band when using `samples` and
`show_interval=True`. `show_gp` defaults to
False and controls display, not the model's GP calculations.
Masked points are distinguished. Upper y limits include unmasked data and all
finite displayed model and band values, including those at masked pixels. Residual limits
use unmasked residuals and all finite displayed GP curve/band values.
Both `point` and `samples` paths require `model`; precomputed prediction
dictionaries are not accepted by `plot_models`.

### MCMC diagnostic plots

After plotting the posterior spectra, create trace and corner figures together:

```python
figures = sf.plot_mcmc_diagnostics(
    mcmc,
    corner_vars=["teff", "logg", "feh", "vsini", "vmacro", "jitter"],
    log_vars=["jitter"],
    labels={"vmacro": "zeta", "jitter": "lnsigma"},
)
```

The default trace includes **every sampled site**, with separate rows for vector
components such as each region's RV and separate chains. Fixed and derived
deterministic outputs are excluded by default. ArviZ's subplot limit is disabled
for this call so that long parameter lists are not truncated. `trace_vars` can
select an explicit subset. `corner_vars` independently selects the corner
variables and also defaults to all sampled sites. Both selections use native
site names and can explicitly include deterministic outputs.

`log_vars` applies natural logs to positive variables in a plotting copy, in
both figures; the original samples are unchanged. `labels` maps native names to
display labels, with `ln(name)` as the default for logged variables. Further
options go in `trace_kwargs` and `corner_kwargs`. The return value is
`{"trace": figure, "corner": figure}`; use `plt.show()` to display them or
`figures["trace"].savefig(...)` to save an individual figure.

Trace plotting uses `arviz.plot_trace` with ArviZ 0.x and
`arviz.plot_trace_dist` with 1.x. `trace_kwargs` goes to the corresponding
function; `legend=False` is supported in both. Other display options follow
the installed ArviZ API (for example, `figsize` in 0.x versus
`figure_kwargs={"figsize": (10, 8)}` in 1.x). The ArviZ 1.x path requires
`corner>=2.3`, which supports `DataTree` input.

Only stored samples and diagnostics are read: plotting does not evaluate the
model, likelihood or spectral grid. Chain/draw coordinates and saved divergence
flags are preserved. The standalone function accepts MCMC, `InferenceData`
(0.x), `DataTree` (1.x), or a loaded `FitResult`, including in a session without
a spectral grid:

```python
from jaxstar.specfit import plot_mcmc_diagnostics

figures = plot_mcmc_diagnostics(result, corner_vars=["teff", "vsini"])
```

Results saved from MCMC retain the sampled-site names in `posterior.nc` as well
as the result metadata. For external ArviZ data without those names, the
default is all posterior variables; specify `trace_vars` and `corner_vars` to
exclude fixed/derived outputs. Existing coordinate labels are used unchanged.

### Save and reload fit results

Save a point estimate, an MCMC posterior, or both with the observation and final
fit mask. The output is a directory, independent of notebook execution state:

```python
from jaxstar.specfit import save_fit_result, load_fit_result

directory = save_fit_result(
    "results/target", sf=sf, model=model, point=p_fit, mcmc=mcmc,
    metadata={"target": "target", "seed": 1},  # optional user context
)
result = load_fit_result(directory)
```

For an optimization-only result, omit `mcmc`. Alternatively, supply an existing
ArviZ object as `idata=idata`; pass at most one of `mcmc` and `idata`.
MCMC conversion copies stored samples and collected diagnostics, including
divergences, without evaluating the likelihood or making new predictions.
Chain and draw dimensions are retained, and vector RV sites receive a region
coordinate. Supplied `idata` groups, coordinates and display variables are
preserved as provided.

ArviZ 0.x uses `InferenceData`; ArviZ 1.x uses xarray `DataTree`. Conversion and
loading select the installed version's API. Both retain the same posterior
groups and chain/draw dimensions in `posterior.nc`.

`load_fit_result` returns a `FitResult` containing:

- `observation`: the original wavelength, flux, uncertainty, fixed mask and labels.
- `fit_mask`: the separate final fit mask.
- `point`: the native-site point dictionary, or `None` when not saved.
- `idata`: the posterior and diagnostics, or `None` for point-only results.
- `samples`: posterior arrays with chain/draw flattened for `predict` and
  `plot_models`; trailing site dimensions such as region RV are preserved.
- `metadata`: model/grid provenance and save-time software information.

The same physical and configured models must be prepared explicitly before
predicting again. For example, with the original `sf.specmodel` and `model`:

```python
restored_sf = SpecFit(result.observation, sf.specmodel)
restored_sf.set_fit_mask(result.fit_mask)
prediction = restored_sf.predict(point=result.point, model=model)
plots = restored_sf.plot_models(samples=result.samples, model=model, show_gp=True)
figures = restored_sf.plot_mcmc_diagnostics(result, corner_vars=["teff", "vsini"])
```

Restoring the fit mask is essential: continuum and GP predictions condition on
the selected observed pixels. When saving a MCMC object from `SpecFit.run_nuts`,
the supplied fit/model and effective observation must still match that run;
changed data or masks raise an error. For externally supplied `idata`, the
caller associates it with the appropriate fit and model.

`metadata["model"]["settings"]` records resolved settings, including the numeric
CCF-derived RV bounds and rotation limit. Arrays carry values, shapes and dtypes;
priors carry distribution names and parameters. Callable references and source
hashes are recorded where available. These are descriptive records, not a
dictionary to pass directly to a model factory. Recreate built-in or custom
models explicitly using the original analysis configuration. Custom objects
that cannot be described numerically are marked as not serialized.

Grid provenance includes source paths, atmosphere axes, actual prepared
wavelength/velocity sampling, broadening settings and a fingerprint of the
in-memory grid. Fixed IP values are recorded under `metadata["grid"]["fixed_ip"]`
in spectral region order; they are model settings, not posterior sample sites.
For reconstruction, extract each parameter's `"values"` and pass that mapping
as `fixed_ip` when recreating the `SpecModel`. The large grid flux array and
executable model code are not bundled; use `save_spectral_grid` separately if a
portable grid artifact is needed. Loading results and plotting posterior
traces/corners require no grid.
This bundle supports reuse of saved results, not continuation of a running NUTS
chain. RNG seeds and additional run context can be supplied in `metadata`.

The directory contains `arrays.npz`, optional `posterior.nc`, and `result.json`.
Reads use no pickle, load the posterior eagerly, and check file hashes. Existing
managed files require `overwrite=True`; unrelated files in the directory are
kept. Saving a point-only result over a posterior bundle removes the old
`posterior.nc` to avoid retaining stale samples.

### Custom model and reconstruction

`model_single` remains a standalone NumPyro function. Its low-level explicit
`atmosphere`, `broadening`, `rv`, `jitter` and GP arguments also accept custom
NumPyro distributions or fixed values. Bind a custom function to settings with:

```python
from jaxstar.specfit import ConfiguredModel

model = ConfiguredModel(
    custom_model, settings=custom_settings,
    reconstruct_fn=custom_reconstruction,
    observation=sf.observation, specmodel=sf.specmodel,
)
```

The callback has signature
`custom_reconstruction(observation, specmodel, returned, trace, settings)`.
`ConfiguredModel.predict` replays the model with every latent supplied by
the point; missing latents raise an error. The callback receives the model's
return value and full trace, and decides how to compute predictions. For example,
a model sampling continuum coefficients uses those coefficients directly.
`reconstruct_single` is the existing Gaussian marginalized-continuum/GP callback;
it returns physical, continuum and mean flux, plus continuum posterior, jitter
and optional GP mean. Custom callbacks used with the standard diagnostics should
return the corresponding prediction fields. An inference-only model needs no
reconstruction callback.

Prior inspection requires a custom model's `prior_only=True` path. Conditional
sample sites may annotate `infer={"prior": {"depends_on": (...), "expression": "..."}}`;
`ConfiguredModel(site_annotations=...)` provides equivalent display metadata,
including for derived deterministic sites. The inspection tool does not infer
symbolic dependencies from a single execution.

The physical IP kernel belongs to `SpecModel`/its broadening operator, while its
parameter priors belong to the NumPyro model. The fixed two-Gaussian IP above is
supported by the built-in physical model. Sampling additional IP parameters or
using other IP families requires extending that interface. This custom-model
API does not add a built-in sampled-continuum model.

`model_single` is an ordinary NumPyro model. Supply fixed numbers or NumPyro
distributions for the physical and noise parameters. Fixed values are recorded
as deterministic sites; only distributions introduce sampled parameters:

```python
from functools import partial
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jaxstar.specfit import model_single

probability_model = partial(
    model_single,
    atmosphere={
        "teff": dist.Uniform(5400., 6000.),
        "logg": dist.Uniform(4., 4.7),
        "feh": dist.Uniform(-.5, .3), "alpha": 0.,
    },
    broadening={"vsini": dist.Uniform(1., 10.), "vmacro": 3.,
                "q1": .36, "q2": .3},
    rv=dist.Uniform(-5., 5.), resolving_power=70000.,
    sigma_constant=.1, sigma_continuum=.03, jitter=dist.HalfNormal(.01),
    degree=4, basis=basis, use_empirical_vmic=False,
)
mcmc = MCMC(NUTS(probability_model), num_warmup=1000, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), sf.fit_observation(), sf.specmodel)
mcmc.print_summary()
```

These ranges and continuum/noise scales are illustrative, not package defaults.
Choose prior support inside the atmosphere grid and the model's wavelength,
RV and broadening coverage. Atmosphere keys must match the loaded library.

Retrieve posterior draws with `mcmc.get_samples()`. Set `save_model_flux=True`
to save physical spectra at every draw, or reconstruct physical and
continuum-corrected spectra afterward with the helpers above.

#### Empirical relations and residual GP

`model_single` also supports these choices:

- `use_empirical_vmic=None` enables the empirical microturbulence relation for
  BOSZ automatically. Omit `vmic` from `atmosphere` in this mode, or set the
  option to `False` to supply an explicit value/prior.
- `use_empirical_vmacro=True` adds a shared empirical macroturbulence prior;
  omit `vmacro` from `broadening`. Its width is `vmacro_empirical_sigma=1.` km/s.
- `use_physical_logg_max=True` applies a Teff-dependent gravity ceiling. For a
  scalar Uniform logg prior, it replaces the supplied upper bound and retains
  the lower bound. The relation is not clipped to the grid; choose compatible
  support throughout the Teff range.
- Supply both `gp_amplitude` and `gp_scale` to add a Matérn-3/2 residual Gaussian
  process (GP). Regions are independent and share the amplitude and scale.
  Amplitude is in flux units and scale is in wavelength units. Set the
  observation mask before sampling; `MCMC(jit_model_args=True)` is unsupported
  in GP mode.

The standard installation includes the GP and plotting dependencies;
see [installation](../README.md#installation). For GP continuum reconstruction,
use `gp_continuum_posterior` and `gp_conditional_mean` with the fitted GP settings.

#### Experimental parallel GP solver

`make_model_single`, `model_single`, and the three low-level GP helpers accept
the static option `gp_parallel=False`, `True`, or `"auto"`. This changes the
numerical solver, preserving the Matérn-3/2 covariance, observation noise,
masking, and analytic continuum marginalization.

The default `False` preserves existing behavior: `gp_solver="auto"` uses
sequential Quasisep on CPU and Direct on GPU. Set `gp_parallel="auto"` to opt
into parallel Quasisep on GPU when the installed tinygp supports it:

```python
model = make_model_single(
    sf, resolving_power=70000., gp_solver="auto", gp_parallel="auto",
)
```

| `gp_solver` | `gp_parallel` | Behavior |
| --- | --- | --- |
| `"auto"` | `False` (default) | Sequential Quasisep on CPU; Direct on GPU |
| `"auto"` | `"auto"` | Parallel Quasisep on a supported GPU; otherwise the default policy |
| `"quasisep"` | `"auto"` | Parallel on a supported GPU; sequential otherwise |
| `"direct"` | `False` or `"auto"` | Always Direct |
| `"auto"` or `"quasisep"` | `True` | Require parallel Quasisep, including on CPU |

Support is detected from the installed `QuasisepSolver` signature, not its
version number. Explicit `True` raises a setup error if tinygp is unsupported
or `gp_solver="direct"`; numerical errors are never silently retried using a
different solver. Detection and selection use Python at setup/tracing time
and JAX's default backend. If running on a different device/backend, choose
the solver and parallel setting explicitly. Keep these options static in JIT.

Configured-model reconstruction retains `gp_parallel`. Low-level callers
should pass the same settings to `gp_continuum_posterior` and
`gp_conditional_mean`. Parallel mode covers the GP factorization and triangular
solves; the final cross-coordinate kernel product in GP-mean reconstruction
still uses tinygp's default implementation.

Parallel mode is experimental and is not enabled by default. Compare
likelihood **and gradients**, including continuum marginalization, at the
pixel counts and precision used for inference. The standalone
[GP benchmark](../benchmarks/benchmark_gp_likelihood.py) compares Direct,
sequential Quasisep, and parallel Quasisep without running sampling:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_gp_likelihood.py \
    --pixels 2048 --regions 2 --dtype float64 --require-parallel \
    --json-out benchmark-results/gp-parallel/gpu-f64.json
```

Use a tinygp build exposing `parallel` and record its exact source revision
with `--tinygp-revision <commit>` when testing a development build. The
benchmark separates compilation from synchronized warm calls and reports
compiler buffer estimates, not measured peak GPU memory. It uses synthetic
spectra with a simple differentiable line model, not the full stellar forward
model or an end-to-end NUTS run.

## Notebooks

- [Coelho workflow](../examples/specfit/coelho.ipynb) demonstrates the basic
  workflow with local Coelho grids and an example IRD dataset under
  `examples/specfit/coelho/`. It covers CCF, SVI, outlier masking and refitting,
  NUTS, plots and result saving.
- [SpecModel](tutorials/specfit/specmodel.ipynb) demonstrates the physical
  stages and multiple stellar components. It requires reference data from a
  sibling `jaxspec` checkout; see the notebook's setup instructions.
- [Continuum](tutorials/specfit/continuum.ipynb) is a self-contained synthetic
  demonstration of marginalization, shrinkage and jitter.
