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
    use_gp=True, gp_solver="auto", use_physical_logg_max=True,
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

sf.mask_outliers(point=p_fit, model=model, plot=True)
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
(`model_flux`), excluding the GP mean. It preserves the fixed observation
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

For reusable BOSZ/Coelho/TLUSTY generation, see the
[recipe-based preparation script](../scripts/specfit/README.md). It shares
instrument wavelength CSVs between libraries, stores a common NPZ with recipe
and input hashes, and keeps generated data outside the repository. The small
multi-region examples use explicit equal pixel counts with log sampling.

BOSZ, Coelho and TLUSTY OB2025 check all input paths before reading spectra. Use
`check_spectral_inputs(library, data_dir, axes=axes)` to inspect the plan without
loading spectra. Missing files are errors by default. For BOSZ,
`missing_nodes="logg"` explicitly enables averaging the native logg +/- 0.5
spectra at fixed Teff/composition/vmic. At the requested upper logg edge only,
an absent upper neighbour permits copying the immediate lower neighbour.
For Coelho, `missing_nodes="teff"` averages the native Teff +/- 250 K spectra
at fixed logg/FeH/alpha, after resampling the selected normalized or
unnormalized flux row onto the output wavelengths. Both neighbours must exist;
there is no one-sided copy or recursive filling. The GAOES Coelho recipe
enables this policy. Neighbours may lie outside the requested atmosphere axes.
Other unresolved nodes fail together before any spectra are read. The script
prints the plan in English and retains replacement methods, source coordinates
and weights in its manifest. See its guide for the numerical procedure and
the resulting flat logg dependence at copied upper-edge nodes.

`prepare_tlusty` also recognizes the OB2025 `BSTAR_z*v2`/`OSTAR_z*v2`
directories (or select `source_format="ob2025"` explicitly). Synthesis vt is
fixed to 2 km/s; the axes remain `teff/logg/logZ`. BSTAR supplies Teff below
30000 K, OSTAR supplies 30000 K and above. The OSTAR atmosphere still uses
vt=10, recorded separately in `spectra.provenance`, the common NPZ and saved
fit metadata. OB2025 supports vacuum or converted air wavelengths, matched
continuum normalization, and preflight of every spectrum/continuum pair.
With `missing_nodes="lowest_logg"`, missing nodes below the native lower-logg
boundary copy the lowest available logg at fixed Teff/Z/family/vt. Existing
nodes retain their own spectra, and partial pairs, interior holes and upper
gaps still fail. The recipe enables this policy for logg=3.0--4.75; omitting
it retains strict mode. Replacement provenance is saved in the NPZ and manifest.
See the [OB2025 recipe and boundary-copy policy](../scripts/specfit/README.md#tlusty-ob2025).
Legacy TLUSTY inputs retain their previous family selection and wavelength
declaration behavior.

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

#### Select regions from a common prepared NPZ

Pass the file written by `save_spectral_grid` or the preparation script to
`SpecFit.from_grid` to select regions for an observation:

```python
from jaxstar.specfit import Observation, SpecFit

obs = Observation(wav_obs, flux_obs, error_obs,
                  mask=mask_obs, region=region_labels)
sf = SpecFit.from_grid("/path/to/prepared/grid.npz", obs,
                       wav_margin=3., vmax=50.)
```

The factory uses each row's full observed wavelength interval, including masked
pixels, to find exactly one saved region covering that interval plus `wav_margin`
Angstrom on **each** side. It selects and reorders regions to match the observed
rows. One saved region may serve several rows. Zero or multiple covering regions
fail before flux loading, with the candidate labels and coverages in the error.
Observation labels name the output rows; they do not select saved identifiers or
resolve overlapping candidates. If observation labels are omitted, selected
saved labels are retained.

Selected regions retain their complete saved wavelength spans and samples.
The full flux array is read into host memory and selected in NumPy; only the
selected subset enters the JAX grid. All atmosphere axes and their ordering,
flux dtype (subject to JAX precision settings), and saved interpretation metadata
are preserved. Source paths remain provenance for the full artifact.

The common NPZ supplies `library`, `wavelength_medium` and `flux_kind`. If supplied,
`library`, a medium other than `"unknown"`, and a flux kind other than `"as_stored"`
must agree with the saved metadata. `pattern` applies only to legacy directories.
The factory performs no wavelength conversion or flux normalization. Ensure that
the artifact's wavelength medium and flux product match your observation.

No sampling arguments are needed for a saved log-uniform grid. Specify either
`pixels` or `velocity_step` to resample explicitly; this is required if the
saved samples are not log-uniform. Resampling uses the selected regions' full
spans, independently of the observed detector pixel count.

`SpecModel.from_grid(path, obs.wavelength, regions=obs.region, ...)` provides the
same selection without storing an observation. To load the selected carrier
directly, use:

```python
spectra = load_spectral_grid(
    "/path/to/prepared/grid.npz", wavelength=obs.wavelength,
    regions=obs.region, wav_margin=3.,
)
```

Omitting `wavelength` retains the original `load_spectral_grid(path)` behavior:
all saved regions are restored. The NPZ schema and spectral carrier are unchanged.

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

`library` selects the existing Coelho (default), BOSZ or TLUSTY loader. `wav_margin` is
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
The directory form accepts legacy NPZs; pass a common-format artifact as a file
as shown above. Neither form accepts raw atmosphere files or performs downloads.

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
macroturbulence and instrumental broadening (resolving-power Gaussian or
`fixed_ip` with the built-in operator), then shifts by the
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

#### Broadening kernel FWHM

Get the combined broadening width directly from the physical parameter dict:

```python
fwhm_kms = specmodel.broadening_fwhm(params)  # shape (n_component, n_region)
```

This uses the same rotation/macroturbulence/instrumental kernels as the spectrum,
on the model's velocity grid. It automatically uses `fixed_ip` when configured;
no resolving power or Gaussian approximation is needed in that mode. The
returned JAX array retains both axes even for one star or one region. JIT is
supported. Only each component's `broadening` and the configured instrument
parameters are needed; atmosphere, RV, flux weights and dilution are ignored.
These are per-component kernel widths, excluding intrinsic line widths and
separation between stars' radial velocities.

FWHM is measured by linear interpolation of the two half-maximum crossings.
Asymmetric profiles are supported when the part at or above half maximum is
one connected interval. Disconnected intervals and crossings outside the
velocity grid raise errors instead of assigning an ambiguous width. Exactly
disabled broadening returns zero. Resolve nonzero widths with fine enough
model velocity sampling, and check convergence with `vmax`; bracketed crossings
alone do not guarantee that the kernel tails fit inside its support. Custom
`broadening_operator` callables do not expose kernels, so this method raises
`NotImplementedError` for them.

Calling `broadening_fwhm` does not update the fit mask.

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
coefficients toward zero. The default Gaussian likelihood integrates these
coefficients analytically; the Gaussian-mixture option below infers them explicitly.
`sigma_constant` and `sigma_continuum` set the two prior scales.
Jitter is in absolute flux units; the Gaussian noise variance is
`error_obs**2 + jitter**2`.

For the Gaussian likelihood, recover the conditional continuum at a chosen
physical model and noise setting:

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

`run_svi`, `run_nuts` and `predict` require `model=model`; these methods have no
separate `priors` or `model_kwargs` argument. Inference accepts any callable of
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
| Continuum | `degree=4`, fixed coefficient scales `0.1` and `0.05` |
| Jitter | Shared LogUniform(1e-3, 0.05), in normalized flux units |
| GP amplitude | Shared LogUniform(1e-3, 0.05), in normalized flux units |
| GP scale | Shared LogUniform from 5 km/s equivalent to region wavelength span / 5; GP enabled by default, `gp_solver="auto"` |

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

With noise bounds omitted (`None`), the factory uses fixed bounds `(1e-3, 0.05)`
for jitter and GP amplitude, independent of observational errors and masks.
For each region, the GP scale lower bound is
`5 * wavelength_midpoint / 299792.458` and the upper bound is its full wavelength
span divided by five, both in wavelength units (Angstrom). The shared scale
prior uses the largest lower bound and smallest upper bound across regions.
These bounds are fixed when the model is created and do not change during
outlier masking. Explicit bounds override them. If the derived interval is
empty, supply explicit bounds; fixed parameters and disabled GP do not require
derived bounds. The data-dependent scale setup belongs to `make_model_single`;
direct `model_single` calls retain their existing numeric GP scale bounds.

RV priors are not clipped to `sf.rv_bounds`; spectrum evaluation still checks
coverage. `use_empirical_vmic=None` enables BOSZ's deterministic relation;
`False` makes vmic explicit. `use_empirical_vmacro=True` (or
`"valenti_fischer2005"`) selects the Valenti & Fischer conditional prior;
`"doyle2014"` and `"lubin2026"` also depend on logg. With
`vmacro_empirical_sigma=None`, their widths are 1.0, 0.37 and 0.09 km/s,
respectively. A positive scalar overrides the width. The Doyle and Lubin
defaults adopt observed residual RMS values, not measurement-error-corrected
intrinsic scatter. All three helpers return a zero-truncated Normal prior;
the inputs and empirical locations are not clipped. Calibration ranges and
sources are documented in `empirical_vmacro_valenti_fischer2005`,
`empirical_vmacro_doyle2014` and `empirical_vmacro_lubin2026`. Fixed values for active
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

### Gaussian mixture for local mismatch

The Gaussian-mixture option models each pixel independently as

```text
r_i = y_i - physical_flux_i * continuum_i
v_i = error_obs_i**2 + jitter_region**2
p(r_i) = (1 - f) * Normal(r_i | 0, variance=v_i)
           + f * Normal(r_i | 0, variance=v_i + s_mismatch**2)
```

The narrow component represents pixels adequately described by the model;
the broader component allows local mismatch, such as missing or incorrect
spectral lines. `mismatch_fraction` is the prior fraction `f`, shared by all
regions. `mismatch_scale` is the global **absolute flux amplitude**
`s_mismatch`, not a multiplier of the observational errors. The full mixture
variance is `v_i + f * s_mismatch**2`.

```python
import numpyro.distributions as dist

model = make_model_single(
    sf, resolving_power=70000., degree=4,
    likelihood="gaussian_mixture", use_gp=False,
    mismatch_fraction=dist.Beta(1., 9.),
    mismatch_scale=dist.HalfNormal(.05),
)
model.check_prior()
p_fit = sf.run_svi(model=model, rng_key=jax.random.PRNGKey(0))
```

Both `model_single` and `make_model_single` accept these arguments. Supply
both mixture parameters explicitly as numbers to fix them or as scalar
NumPyro distributions to infer them. The factory has no default mixture priors;
the prefit convenience method below supplies its own normalized-flux priors
when omitted. Fractions must be in `[0, 1]` and
scales nonnegative, with finite values. For inference choose priors inside
`(0, 1)` and on positive scales. Fixed `f=0`/`f=1` select the pure components;
`s_mismatch=0` makes them identical. Near these limits the mixture parameters
are weakly identified; near `f=1`, mismatch scale and jitter can trade off.

Mixture mode requires no GP. Set `use_gp=False` explicitly on the factory,
whose default remains `True`; conflicting GP settings raise an error.
`likelihood="gaussian"` remains the default and preserves analytic continuum
marginalization. In mixture mode, `continuum_coefficients` is an explicit
inference site under the existing Gaussian priors, optimized by `sf.run_svi`
or sampled by `sf.run_nuts`, with shape `(degree + 1,)` for 1D observations
or `(n_regions, degree + 1)` for 2D observations. It appears in `check_prior`
and in inference results. Predictions use these same coefficients.

Pixel membership indicators are **not sampled**. The likelihood sums the two
weighted, normalized Gaussian densities using `logsumexp` separately for
each pixel, then sums pixel log densities. This follows the standard
[finite-mixture marginalization](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#summing-out-the-responsibility-parameter).

The broad component reduces the influence of mismatched pixels. Known
bad-pixel and other fixed observation masks still apply.
The mixture likelihood does not automatically change either mask. If reusing
a fit with previous residual clipping, explicitly call `sf.reset_fit_mask()` to clear
that additional mask while retaining the observation's fixed mask. Broad
pixels still carry weight; this model neither ignores them completely nor
models wavelength correlations or repairs incorrect spectral lines.

`predict` and `plot_models` leave masks unchanged. To turn the mixture model's
probabilities into a fit mask, call `mask_outliers(method="gaussian_mixture")`:
it uses the predicted probabilities directly, without re-estimating the
mixture fraction or scale.

Posterior mismatch probabilities (responsibilities) are available as:

```python
prediction = sf.predict(point=p_fit, model=model)
q = prediction["mismatch_probability"]  # Same shape as observation.flux
sf.plot_models(point=p_fit, model=model, show_mismatch=True)
```

For posterior draws obtained by sampling this same mixture model:

```python
prediction = sf.predict(samples=mcmc.get_samples(), model=model, max_draws=50)
q = prediction["mismatch_probability"]
sf.plot_models(samples=mcmc.get_samples(), model=model, show_mismatch=True)
```

At a supplied parameter point, `q_i = f * Normal_broad(r_i) / p(r_i)`.
For posterior samples, probabilities are evaluated at each selected draw and
then averaged; they are not computed from the average spectrum. With the
built-in mixture model, `sf.predict` evaluates probabilities on all pixels
outside the fixed observation mask, including fit-masked pixels, at the supplied
parameters. Only fixed-masked pixels have `NaN` probabilities. Inference still
excludes both fixed- and fit-masked pixels. Gaussian continuum/GP predictions
continue to condition on the current fit mask.
`show_mismatch=True` adds a third plot panel with probabilities between 0 and 1.
It includes fit-masked probabilities when `show_masked=True` (the default) and
hides them when `show_masked=False`; fixed-masked probabilities are always
omitted. Plotting does not change masks.
Point predictions also return `continuum_coefficients`, `mismatch_fraction`
and `mismatch_scale`, without a Gaussian `continuum_posterior`. Supply every
sampled site when predicting a point.

Custom models can use `gaussian_mixture_log_likelihood` and
`gaussian_mixture_responsibility`, both with arguments
`(observation, model_flux, mismatch_fraction=..., mismatch_scale=..., jitter=...)`.
Here `model_flux` already includes the explicit continuum. The density ignores
masked pixels exactly, and the responsibility helper creates no sample sites.

### Mixture prefit

`run_mixture_prefit` builds the no-GP mixture model, prepares all initial values
and calls `run_svi`. It returns `(model, point)` for use with the existing
prediction, plotting and masking APIs:

```python
model_kwargs = dict(
    resolving_power=70000., use_physical_logg_max=True,
    use_empirical_vmic=False, use_empirical_vmacro=False,
)
prefit_model, prefit_point = sf.run_mixture_prefit(
    rng_key=jax.random.PRNGKey(0),
    model_kwargs=model_kwargs,
    p_initial={"teff": 5800.},  # Optional; use values inside the loaded grid
    step_size=.01, num_steps=2000,
)
prefit_model.check_prior()
sf.plot_models(model=prefit_model, point=prefit_point, show_mismatch=True)

# This separate call changes the fit mask using the prefit probabilities.
sf.mask_outliers(
    model=prefit_model, point=prefit_point, method="gaussian_mixture",
    p_outlier=.5, mask_velocity=10.,
)
```

`model_kwargs` forwards factory settings, including physical priors, fixed
parameters, continuum degree/scales and optional mixture priors. The wrapper
sets `likelihood="gaussian_mixture"`, `use_gp=False` and rejects conflicts.
CCF is still required if the factory needs default RV or rotation bounds.
The SVI defaults are `step_size=.01`, `num_steps=2000`, `progress_bar=True`.

| Parameter | Default prior | Prefit initial value, when sampled |
| --- | --- | --- |
| `continuum_coefficients` | Existing Gaussian continuum priors | `[1, 0, ...]` per region |
| `jitter` | Factory LogUniform(0.001, 0.05) | `0.002` |
| `mismatch_fraction` | Beta(1, 9) | `0.01` |
| `mismatch_scale` | HalfNormal(0.1) | `0.1` |

These scales assume normalized flux. Prior settings and initial values are
independent: change the former through `model_kwargs` and the latter through
`p_initial`. For example, `model_kwargs={..., "mismatch_scale": dist.HalfNormal(.2)}`
changes its prior width, while `p_initial={"mismatch_scale": .2}` changes its
starting point. Numeric mixture parameters and entries in `fixed` stay fixed;
do not include fixed or derived sites in `p_initial`.

Partial `p_initial` mappings override the prefit defaults. Shapes must match
the sample sites, including continuum and region-vector parameters. Remaining
parameters are drawn from their priors conditional on the supplied values
(for example, `logg` conditional on the starting `teff`). Every latent is then
passed to SVI, avoiding the uniform fallback used by an ordinary `run_svi`
call with a partial initial-value mapping. Initial values outside support,
nonfinite values and boundaries mapping to infinite unconstrained coordinates
raise an error before SVI. If a custom prior excludes a default initial value,
supply a compatible override; the method does not clip it into the prior.

The prefit uses the current fixed and fit masks and changes neither. Clear an
earlier fit mask explicitly with `reset_fit_mask()` if needed. After selecting
and fixing the mask, construct a separate Gaussian GP/Vecchia model with the
shared physical settings for final inference. Passing `p_initial=prefit_point`
to its `run_svi` call reuses shared parameters; mixture-only and deterministic
entries are ignored. The new GP parameters use NumPyro's `init_to_uniform`
fallback in unconstrained coordinates. The final continuum is analytically
marginalized, so it needs no coefficient initialization. The
[prefit notebook](../examples/specfit/coelho_mixture_prefit.ipynb) shows this
transition, a final SVI and NUTS run.

### Outlier plots

`mask_outliers` defaults to `method="gaussian_mixture"`, with `p_outlier=0.5`
and `mask_velocity=10.` km/s. Calls that omit `method` now use mixture masking
and can produce different masks than earlier versions. To retain the legacy
median-filter/MAD method, select `method="sigma_clip"` explicitly.

For a legacy sigma-clipping quicklook, enable plotting when updating the fit mask:

```python
sf.mask_outliers(point=p_fit, model=model, method="sigma_clip",
                 sigma_threshold=3., plot=True)
```

With `method="sigma_clip"`, `mask_outliers` calls `predict` once before updating the mask.
It uses `model_flux` as the baseline and the largest component `vsini` for
the median-filter width; `mask_v` can override that velocity in km/s. Previously
flagged pixels are reconsidered, so they can return to the sf. Custom model
predictions must include `model_flux`, plus physical `params` unless
`mask_v` is supplied. Pass a parameter point and the model, not a precomputed
prediction dictionary.

`plot=True` creates three panels on the left for each region: data and
`model_flux` (excluding the GP mean); residuals and their median-filter trend;
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
sf.mask_outliers(point=p_fit, model=model, method="sigma_clip", sigma_threshold=3.,
                  plot=True, save_path="outliers.pdf")
```

This saves `outliers_region8.pdf`, `outliers_region9.pdf`, etc., with the same
filename conventions as `plot_models`. Without an extension, Matplotlib's
default save format is used. With `plot=False` (the default), `save_path` still
creates and saves the figures, then closes them; use `plot=True` to keep them
open as well. Saving uses the diagnostics from the same masking pass.

For the default mixture-based masking after a normal Gaussian/GP fit:

```python
sf.mask_outliers(point=p_fit, model=model,
                 p_outlier=0.5, plot=True)  # Configurable; default 0.5
q = sf.outlier_diagnostics["mismatch_probability"]
# Refit with the same model and the new fixed fit mask.
```

The mixture uses `data - model_flux`, including the fitted conditional continuum but
excluding the GP mean. It fits one global fraction `f` and absolute flux scale
`s` by maximum likelihood with component variances `V` and `V + s**2`.
`V` is the fitted GP's combined covariance diagonal (kernel, observational
noise and jitter counted once); without a GP it is observational variance plus
jitter variance. It is not the conditional GP predictive variance or continuum
uncertainty. The continuum marginalization and inference likelihood are unchanged.

If the model prediction already contains `mismatch_probability`, the same call
uses those probabilities instead of fitting a mixture to residuals. For example,
after SVI with `likelihood="gaussian_mixture"`:

```python
sf.mask_outliers(point=p_fit, model=model, method="gaussian_mixture",
                 p_outlier=0.5, mask_velocity=10., plot=True)
```

This only applies the probability threshold and mask extension. It preserves
the model's inferred `mismatch_fraction` and `mismatch_scale`. Probabilities
must have the observation shape and be finite in `[0, 1]` outside the fixed
observation mask, including previously fit-masked pixels. Invalid supplied
probabilities raise an error without changing the mask or falling back to a
residual fit. `method="sigma_clip"` ignores these probabilities and retains its
existing behavior.

Pixels with `q > p_outlier` are flagged. For mixture masking, the default
`mask_velocity=10.` extends each original flagged pixel by ±10 km/s within its
region, using log-wavelength distances. Overlapping windows are combined
without recursive extension.
`mask_velocity=None` uses the legacy `extension_factor`; zero adds no neighbors,
and `extend_outlier_mask=False` disables extension. `mask_velocity` applies only
to the mixture method and is separate from the sigma-clipping filter's `mask_v`.
As with sigma clipping, fixed masked pixels are excluded and previous fit-mask
pixels are reconsidered. Probabilities are NaN on fixed
masks. `outlier_diagnostics` also records `mismatch_fraction`, `mismatch_scale`,
and `normal_residual_variance` from that call. `probability_source` is `"model"`
when probabilities are supplied by the prediction, or `"residual_fit"` when a
mixture is fitted to the residuals. The diagnostics are a snapshot of the
probabilities and parameters actually used for that pass. These are conditional
classification probabilities treating pixels independently, not a replacement
for the correlated GP fit. `sigma_threshold` and `mask_v` apply only to sigma
clipping. Mixture plots retain the sigma-clipping layout: data and physical
model at the top, physical residuals in the middle, and outlier probabilities
with their threshold at the bottom. The right-hand histogram shows residuals
divided by `sqrt(V)`, stacking retained and masked counts so the total is
visible. Thin, smooth curves show the normal and outlier component densities
and their sum, scaled by the before-clipping pixel count and bin width as
with the original normal reference. They use all pixels outside the fixed mask:
the normal component has weight `1-f` and unit width, while the outlier component has weight `f`
and pixel-dependent width `sqrt(1 + s**2/V)`. Thus a large outlier fraction
reduces the normal component's expected counts. The overlays are omitted if
either mixture parameter is unavailable. Retained/masked histogram colors
show the final classification including mask extension, not latent component
membership. Extended flags are highlighted throughout; no GP mean or
median-filter trend is subtracted.
Custom model callbacks must supply `normal_residual_variance` alongside
`model_flux`; the built-in model obtains it internally during masking.
Custom models may supply `mismatch_probability` for direct masking, with the
same full-pixel coverage described above. Also return scalar `mismatch_fraction`
and `mismatch_scale` to report those values in diagnostics; if omitted, they
are recorded as `None` and displayed as "not supplied", without fitting them.

### Mask summary and residual diagnostics

Both helpers return one dictionary per region. Format and display the results
in the notebook:

```python
import pandas as pd

display(pd.DataFrame(sf.mask_summary()))
display(pd.DataFrame(sf.residual_diagnostics(point=p_fit, model=model)))
display(pd.DataFrame(sf.residual_diagnostics(
    point=p_fit, model=model, baseline="model_flux",
)))
```

`mask_summary()` reports `region`, `fixed_mask`, `fit_mask` and `usable`.
The mask counts are separate and can overlap; `usable` counts pixels outside
their union, `effective_mask`.

`residual_diagnostics` calls `predict` at the supplied point using the current
effective mask. The default `baseline="mean_flux"` includes the GP conditional
mean; `baseline="model_flux"` uses the physical spectrum multiplied by its
continuum, before adding the GP. Jitter comes from the same prediction and
defaults to zero when omitted by a custom reconstruction callback.

Statistics use only usable pixels. Standardized residuals divide by
`sqrt(error_obs**2 + jitter**2)`; they are not whitened by the GP covariance.
For Gaussian mixtures this is the narrow-component standard deviation, not
the full mixture standard deviation. The `n_gt_3sigma`/`n_gt_5sigma` counts
refer to this same narrow-component scale; use mismatch probabilities for
the mixture's assessment of individual pixels.
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
model_flux = prediction["model_flux"]  # physical spectrum multiplied by continuum
gp_mean = prediction["gp_mean"]  # when a latent GP mean is reconstructed
```

At a parameter point, `model_flux = physical_flux * continuum` and
`mean_flux = model_flux + gp_mean` (or just `model_flux` without a GP or with
`gp_solver="vecchia1"`, which reconstructs the continuum only).
`continuum` is the multiplicative correction itself.

For posterior draws, pass the dictionary returned by `mcmc.get_samples()` with
its flattened leading draw axis:

```python
prediction = sf.predict(samples=mcmc.get_samples(), model=model, max_draws=50)
```

This predicts at up to 50 evenly spaced draws, then averages the returned curves
(`physical_flux`, `continuum`, `model_flux`, `mean_flux`, `gp_mean`, where
provided). `max_draws=None` uses all draws. The result also includes
`draw_indices`; physical parameter dictionaries, conditional covariance objects
and noise parameters are not averaged. GP means are computed at each draw
before averaging. `model_flux` averages the product evaluated at each draw,
not the product of the separately averaged `physical_flux` and `continuum`.
When `model_flux` is available, the result also includes
`model_flux_quantiles` and `gp_mean_quantiles`: pointwise 16th and 84th
percentiles of `model_flux` and `mean_flux - model_flux`, respectively,
across the same draws, with shape `(2, *observation.shape)`. The GP difference
is computed within each draw before taking percentiles. These describe the
spread of model curves across posterior draws. Gaussian predictions exclude
the conditional continuum/GP uncertainty remaining within each draw. In
Gaussian-mixture mode the continuum coefficients are sampled, so their
posterior uncertainty is included in the bands. Neither includes fresh
observational noise, so these are not full posterior predictive intervals.
When supplied by the model,
`mismatch_probability` is also averaged across the selected draws.

`plot_models` calls `predict` internally:

```python
plots = sf.plot_models(point=p_fit, model=model, show_gp=True)
plots = sf.plot_models(samples=mcmc.get_samples(), model=model,
                       show_gp=True, show_interval=True, max_draws=50)
```

It returns `[(fig, axes), ...]` in region order, with one figure per region and
two axes per figure, or three when `show_mismatch=True`. The panels share their
x axis and have a small vertical gap; only the bottom panel shows x tick labels. The x limits match
the region's wavelength endpoints. A common legend sits vertically to the right
of the panels. Observed points use open gray circles labeled `data`, matching
the outlier check; mask entries are omitted when that region has no such masked pixels.

`save_path="models.png"` saves one file per region, such as
`models_region8.png` and `models_region9.png`. The suffix is added even for one
region; special characters in region labels are URL-escaped for filenames.
Without a filename extension, Matplotlib's default save format is used.

The upper panel shows data and the physical model including its continuum
prediction (`model_flux`). Bands are hidden by default (`show_interval=False`).
With `samples` and `show_interval=True`, it shades the pointwise 16th-84th
percentile range of the model curves described above. The lower
shows `data - model_flux`, subtracting the mean curve for `samples`. Its
zero line uses C0 to match the continuum model in the upper panel.
`show_gp=True` overlays `mean_flux - model_flux` in C1 in the lower panel,
with its own 16th-84th percentile band when using `samples` and
`show_interval=True`. `show_gp` defaults to
False and controls display, not the model's GP calculations.
Vecchia models omit the GP overlay even with `show_gp=True`, because their
latent GP mean is not reconstructed.
Masked points are distinguished by default. Set `show_masked=False` to hide
both observation- and fit-masked data points in the upper and residual panels.
It also hides fit-masked probabilities in the optional mismatch panel.
Fixed-masked probabilities are always omitted. Model and GP curves/bands
still cover all wavelengths.
Upper y limits include unmasked data and all
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
`reconstruct_single` supports the built-in likelihoods. Gaussian mode returns
physical, continuum and mean flux, plus continuum posterior, jitter and optional
GP mean. Gaussian-mixture mode uses the sampled coefficients and also returns
mismatch probabilities and parameters, without a Gaussian continuum posterior.
Custom callbacks used with the standard diagnostics should
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
API also supports the Gaussian-mixture model's sampled continuum described above.

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
to save physical spectra in the NumPyro trace's `model_flux` site at every draw.
This trace option stores the spectrum before the continuum correction;
`predict()["model_flux"]` includes that correction. Reconstruct physical and
continuum-corrected spectra afterward with the helpers above.

#### Empirical relations and residual GP

`model_single` also supports these choices:

- `use_empirical_vmic=None` enables the empirical microturbulence relation for
  BOSZ automatically. Omit `vmic` from `atmosphere` in this mode, or set the
  option to `False` to supply an explicit value/prior.
- `use_empirical_vmacro=True` (or `"valenti_fischer2005"`), `"doyle2014"` or
  `"lubin2026"` adds a shared empirical macroturbulence prior; omit `vmacro`
  from `broadening`. `vmacro_empirical_sigma=None` uses the relation's default
  width (1.0, 0.37 or 0.09 km/s); a positive scalar overrides it.
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

### Experimental nearest-neighbor likelihood

Set `gp_solver="vecchia1"` on `make_model_single`, `model_single`, or
`gp_marginalized_continuum_log_likelihood` to opt into the experimental k=1
Vecchia approximation. Defaults are unchanged: `auto`, `quasisep`, and `direct`
still evaluate the exact Matérn-3/2 likelihood with tinygp. `auto` selects
`quasisep` on CPU and `direct` on GPU.

Vecchia provides a fast approximation to the GP likelihood. It can
provide substantial speedups, especially on GPU; the gain depends on hardware
and data. To assess its effect on inference, compare posterior estimates and
uncertainties with `gp_solver="auto"`, keeping the priors and masks identical.
For example:

```python
model = make_model_single(sf, resolving_power=70000., use_gp=True,
                          gp_solver="vecchia1")
```

For retained pixels in increasing wavelength order, start from the same
covariance `C = K_M32 + diag(uncertainty**2 + jitter**2)`. Set
`V = diag(C)`, `b[i] = C[i,i-1]/V[i-1]`, and
`v[i] = V[i] - C[i,i-1]**2/V[i-1]`. The approximate density is
`N(r[0]; 0,V[0]) * product(N(r[i]; b[i]*r[i-1],v[i]))`.
This uses observed-residual conditionals from the target covariance; it does
not divide residuals by observational errors or substitute an exponential
kernel. The induced joint covariance agrees on diagonal/adjacent entries,
but generally differs at longer separations. Masked pixels are removed first;
adjacent covariance uses the actual wavelength gap across a mask. The kernel
parameter `gp_scale` keeps its meaning and units, but its fitted value can
change with the approximate density.

For the existing multiplicative continuum, `X = model_flux[:,None] * basis`,
`r = y-X*prior.mean`, and `Z = W*X*diag(prior.scale)`. Apply the same bidiagonal
whitening `W` to `r` and every design column. The existing small system
`H = I+Z.T@Z` then integrates the continuum exactly under the approximate
density. The noise determinant `log(V[0])+sum(log(v[1:]))` and the continuum
determinant `log|H|` each enter the Gaussian normalization once. A stable
nonnegative-sum expression for `v` avoids cancellation without a variance floor.

Pixel operations are vectorized and require linear work/storage at fixed
continuum degree; there is no pixel scan or dense pixel covariance. Independent
regions retain the existing loop so different retained-pixel counts are allowed.
Use `prepare_gp_observation` before passing data dynamically through JIT, as
with the exact GP functions. Amplitude, scale, jitter and model parameters
remain differentiable.

`gp_continuum_posterior(..., gp_solver="vecchia1")` gives the continuum
posterior under this approximate likelihood. Standard `SpecFit.predict` uses
that posterior to return `model_flux = physical_flux * continuum` and
`mean_flux = model_flux`, without a `gp_mean` entry. This means the latent GP
mean is not reconstructed, rather than estimated to be zero. `plot_models`
omits its GP overlay, including when `show_gp=True`. Direct calls to
`gp_conditional_mean` still raise for `vecchia1`; no latent prediction
extension or exact-GP fallback is applied.

Point predictions also return `normal_residual_variance`, equal to
`gp_amplitude**2 + uncertainty**2 + jitter**2`. This is the marginal variance `V`, not the
conditional innovation variance `v`. Standard `mask_outliers` uses it for
Gaussian-mixture selection without constructing a dense covariance or GP mean.
The model and variance cover previously fit-masked pixels as well, so those
pixels can be reconsidered. In this Gaussian/Vecchia configuration, continuum
coefficients remain analytically marginalized and the mixture is used only
for masking. The separate `likelihood="gaussian_mixture"` option requires
`use_gp=False` and infers continuum coefficients explicitly.

The [comparison benchmark](../benchmarks/benchmark_vecchia1.py) measures
likelihoods and gradients separately, including the analytic continuum
integral, and can fit both densities to a spectrum drawn from the exact model.

## Notebooks

- [Coelho workflow](../examples/specfit/coelho.ipynb) demonstrates the basic
  workflow with local Coelho grids and an example IRD dataset under
  `examples/specfit/coelho/`. It covers CCF, SVI, outlier masking and refitting,
  NUTS, plots and result saving. Set `GP_SOLVER="auto"` (default) for exact GP
  inference, or `GP_SOLVER="vecchia1"` to try the experimental approximation.
- [Coelho mixture prefit](../examples/specfit/coelho_mixture_prefit.ipynb) uses
  the mixture only for prefit/masking, then runs a Gaussian GP/Vecchia model
  with analytic continuum marginalization and a fixed mask.
- [SpecModel](tutorials/specfit/specmodel.ipynb) demonstrates the physical
  stages and multiple stellar components. It requires reference data from a
  sibling `jaxspec` checkout; see the notebook's setup instructions.
- [Continuum](tutorials/specfit/continuum.ipynb) is a self-contained synthetic
  demonstration of marginalization, shrinkage and jitter.
