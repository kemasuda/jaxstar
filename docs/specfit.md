# SpecFit guide

[README](../README.md) · [MistFit](mistfit.md) · [MIST grids](mistgrid.md)

`jaxstar.specfit` provides prepared spectral grids, a deterministic `SpecModel`,
an `Observation` container, continuum likelihoods and `model_single` for
single-star NumPyro inference. These components can also be used directly in
custom probability models.

## Prepare and load spectra

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
model = SpecModel(spectra, vmax=50.)  # broadening kernel support in km/s
print(spectra.grid.axis_names)
```

The following examples assume a normalized Coelho grid containing the stated
parameters and prior ranges. Adapt them to your library; spectra and observed
data are external inputs.

## Predict a physical spectrum

`SpecModel` interpolates the atmosphere, applies rotation, radial-tangential
macroturbulence and Gaussian instrumental broadening, then shifts by the
relativistic RV and samples the requested wavelengths:

```python
from jaxstar.specfit import single_star_params

params = single_star_params(
    {"teff": 5700., "logg": 4.4, "feh": 0., "alpha": 0.},
    vsini=3., vmacro=3., q1=.36, q2=.3, rv=0., resolving_power=70000.,
)
# wavelength: your evaluation wavelengths, in the library's Angstrom/medium convention
flux = model(params, wavelength)
```

`single_star_params` builds the nested component/instrument dictionary and
converts `q1`, `q2` to quadratic limb-darkening coefficients. Velocity parameters
are in km/s. `model.intrinsic`, `model.broadened` and `model.full` expose the
successive physical stages; calling `model` is equivalent to `full`.

Wavelength arrays have shape `(region, pixel)`; one region may use a 1D array.
Atmosphere parameters are scalar for each star. Broadening, RV and resolving
power may be shared scalars or one value per region. The prepared grid must
cover the full evaluation range after broadening and RV shifts; the model does
not add wavelength padding or extrapolate automatically.

For multiple stars, supply a tuple of `components` plus nonnegative
`flux_weights`. All components use the same spectral library. Optional
`dilution` is the featureless fraction of total continuum light:
`total = dilution + (1 - dilution) * sum(weight_i * flux_i) / sum(weight_i)`.
`model.decompose` returns the individual and weighted spectra. The
[SpecModel tutorial](../examples/tutorials/specmodel.ipynb) includes SB2/SB3
examples. Use a custom NumPyro model to fit multiple stellar components.

## Observations and continuum

Observed arrays must have identical 1D or 2D shapes and use the same region
order and wavelength convention as the prepared model:

```python
from jaxstar.specfit import Observation, chebyshev_basis

obs = Observation(wavelength, measured_flux, uncertainty, mask=excluded_pixels)
basis = chebyshev_basis(obs.wavelength, degree=4)
```

`mask=True` excludes a pixel; omit the mask to use every pixel. Wavelengths
must remain finite, positive and increasing even at masked pixels. Usable flux
and errors must be finite, with positive errors. Optional region labels must
agree with the library's row order when used with `model_single`.

A multiplicative Chebyshev continuum is applied after the physical model.
Gaussian coefficient priors shrink the constant toward one and the other
coefficients toward zero. The likelihood integrates these coefficients
analytically. `sigma_constant` and `sigma_continuum` set the two prior scales.
Jitter is in absolute flux units; the total noise variance is
`uncertainty**2 + jitter**2`.

At a chosen physical model and noise setting, recover the conditional continuum:

```python
from jaxstar.specfit import continuum_posterior, apply_continuum

physical_flux = model(params, obs.wavelength)
continuum = continuum_posterior(
    obs, physical_flux, basis=basis,
    sigma_constant=.1, sigma_continuum=.03, jitter=.01,
)
fitted_flux = apply_continuum(physical_flux, basis, continuum.mean)
```

The [continuum tutorial](../examples/tutorials/continuum.ipynb) is a synthetic
example requiring no external spectra. For custom models,
`marginalized_continuum_log_likelihood` exposes the normalized Gaussian
likelihood directly.

## Single-star NUTS

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
mcmc.run(jax.random.PRNGKey(0), obs, model)
mcmc.print_summary()
```

These ranges and continuum/noise scales are illustrative, not package defaults.
Choose prior support inside the atmosphere grid and the model's wavelength,
RV and broadening coverage. Atmosphere keys must match the loaded library.

Retrieve posterior draws with `mcmc.get_samples()`. Set `save_model_flux=True`
to save physical spectra at every draw, or reconstruct physical and
continuum-corrected spectra afterward with the helpers above.

### Empirical relations and residual GP

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

The `spectral-inference` extra includes the GP and plotting dependencies;
see [installation](../README.md#installation). For GP continuum reconstruction,
use `gp_continuum_posterior` and `gp_conditional_mean` with the fitted GP settings.

## Existing notebooks

- [SpecModel](../examples/tutorials/specmodel.ipynb) demonstrates the physical
  stages and multiple stellar components. It requires reference data from a
  sibling `jaxspec` checkout; see the notebook's setup instructions.
- [Continuum](../examples/tutorials/continuum.ipynb) is a self-contained synthetic
  demonstration of marginalization, shrinkage and jitter.
- [Real IRD order-8 inference](../examples/exploratory/real_order8_inference.ipynb)
  compares independent-noise and GP likelihoods using external reference data
  and short exploratory chains.
