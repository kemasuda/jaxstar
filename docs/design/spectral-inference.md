# Minimal single-star spectral inference

This milestone adds an ordinary NumPyro convenience model, not SpecFit or an
optimizer API. `SpecModel` remains deterministic and has no data/noise state.
The same `model_single` is used for MAP-like SVI and NUTS. User-written NumPyro
models can use the physical and continuum helpers directly without adopting
the packaged prior choices or numpyro-inferutils.

## Explicit inputs and priors

```python
import numpyro.distributions as dist
from jaxstar.specfit import model_single, chebyshev_basis

kwargs = dict(
    atmosphere={"teff": dist.Uniform(5000, 6500), "logg": dist.Uniform(3.5, 4.7),
                "mh": dist.Uniform(-0.6, 0.4), "alpha": dist.Uniform(0, 0.4),
                "carbon": 0.0, "vmic": dist.Uniform(0.5, 2.5)},
    broadening={"vsini": dist.Uniform(2, 13),
                "vmacro": dist.TruncatedNormal(3, 1, low=0.5, high=6),
                "q1": 0.36, "q2": 0.3},
    rv=dist.Uniform(8, 17), resolving_power=70000,
    sigma_constant=0.1, sigma_continuum=dist.LogNormal(-3.2188758, 0.7),
    jitter=dist.HalfNormal(0.01), degree=4,
    basis=chebyshev_basis(obs.wavelength),
    use_empirical_vmic=False,
)
# Explicit BOSZ example; all ranges/scales above are illustrative.
```

`model_single(obs, specmodel, **kwargs)` accepts a fixed number/JAX array or
NumPyro distribution at every parameter input. Atmosphere keys must exactly
match the loaded library and be scalar, except that `vmic` must be omitted
when empirical vmic is active. Broadening normally requires exactly
`vsini/vmacro/q1/q2`; omit `vmacro` when empirical vmacro is active. Broadening,
RV, resolving power, continuum scales and jitter can be scalar or one value
per region. Optional GP amplitude/scale are shared scalars. A vector distribution
is promoted to one event site; there
is no accidental plate/region broadcasting. Rows are aligned explicitly;
optional Observation.region labels must equal library labels in row order.
No positional parameter vector, automatic CCF bounds or universal prior ranges
are introduced. The caller must choose prior support inside the library and
the model's broadening/RV coverage. Invalid support is a setup error; the
deterministic model does not silently extrapolate or catch it as an HMC reject.

Sample sites use native physical names: loaded atmosphere axes, `vsini`,
`vmacro`, `q1`, `q2`, `rv`, `resolving_power`, `sigma_constant`,
`sigma_continuum`, `jitter`. Only inputs supplied as distributions are latent;
fixed inputs are deterministic. Native NumPyro constraints/transforms replace
the legacy `*_scaled` bookkeeping. `u1=2*sqrt(q1)*q2` and
`u2=sqrt(q1)-u1` are deterministic sites, preserving the legacy convention.
`single_star_params(atmosphere, **physical_values)` performs this conversion
and builds the one-component PyTree for reconstruction/custom models.

`save_model_flux=False` keeps the default trace compact. Opt in to the physical
array site only when useful. No conditional continuum or residual array is
stored for every draw; compute it afterwards using `continuum_posterior`,
`evaluate_continuum`, and `apply_continuum` at a representative point or draws.
The model returns its physical parameter dict when called under a trace.

## Stellar relation options

The readable model resolves Teff, then logg, then the mh coordinate needed
for empirical vmic. Remaining atmosphere axes use their supplied fixed values
or distributions. Relations are opt-in except for empirical vmic on BOSZ:

| Argument | Default | Behavior |
| --- | --- | --- |
| `use_physical_logg_max` | `False` | Enable a Teff-dependent logg ceiling |
| `use_empirical_vmic` | `None` | Auto-enable on BOSZ; `True` requests the relation, `False` disables it |
| `use_empirical_vmacro` | `False` | Enable the Valenti & Fischer (2005) vmacro prior |
| `vmacro_empirical_sigma` | `1.0` | Positive finite scalar width of that prior, in km/s |

These switches are static Python bools (`use_empirical_vmic` also accepts
`None`). Library identity comes from `specmodel.spectra.library == "bosz"`,
set by the BOSZ loaders/preparers and preserved by common-grid storage and
resampling. Axis names or source filenames alone do not enable BOSZ auto mode.

With `use_physical_logg_max=True`, `atmosphere["logg"]` supports a scalar
`dist.Uniform` or a fixed scalar. For a Uniform, its lower bound is retained
and its upper bound is replaced by `physical_logg_max(teff)`; the originally
supplied upper bound is ignored, reproducing legacy jaxspec behavior. Logg
remains a native `logg` sample site. Other distribution types, including
transformed or event-wrapped Uniforms, are rejected in this mode. A fixed
logg remains deterministic and must be finite and at or below the ceiling.
The Uniform lower bound must be finite and strictly below the computed
upper bound; invalid intervals fail clearly. This option requires teff/logg
axes and does not clip the input Teff to 4500--7000 K or clamp the computed
ceiling to a grid boundary. The caller must select a compatible Teff prior
and lower bound over the full support.

Empirical vmic requires teff/logg/mh/vmic axes. `None` enables it automatically
for BOSZ, `True` enables it explicitly for any compatible library, and
`False` uses the ordinary explicit vmic value or prior. The active mode
records `vmic` deterministically from `empirical_vmic(teff, logg, mh)`;
there is no latent vmic. Omit `atmosphere["vmic"]` in active modes. Supplying
it is an error even in BOSZ auto mode; use `False` to fit or fix it explicitly.
Non-BOSZ auto mode requires explicit vmic if the loaded library has that axis.
All other atmosphere keys remain mandatory and extra keys are rejected.

Empirical vmacro requires a teff axis and samples exactly one shared scalar
stellar `vmacro` from `empirical_vmacro_valenti_fischer2005(teff,
sigma=vmacro_empirical_sigma)`. Its lower bound is zero and its default
normal width before truncation is 1 km/s. Sigma must be finite, positive and
scalar; a per-region sigma is rejected. Omit `broadening["vmacro"]` when
this mode is enabled; providing both is an error. `vsini/q1/q2` remain
mandatory and extra broadening keys are rejected. Explicit broadening,
kinematics, continuum scales and jitter retain their scalar/per-region
behavior. No `*_scaled` sites are introduced.

For a BOSZ fit using all three relations, the configuration is explicit:

```python
kwargs = dict(
    atmosphere={"teff": dist.Uniform(5200, 6200),
                "logg": dist.Uniform(3.8, 4.9),  # upper bound is replaced
                "mh": dist.Uniform(-0.3, 0.3), "alpha": 0.0, "carbon": 0.0},
    broadening={"vsini": dist.Uniform(2, 13), "q1": 0.36, "q2": 0.3},
    rv=dist.Uniform(8, 17), resolving_power=70000,
    sigma_constant=0.1, sigma_continuum=0.03, jitter=0.0,
    use_physical_logg_max=True,
    use_empirical_vmic=None,  # BOSZ auto; vmic omitted above
    use_empirical_vmacro=True,
    vmacro_empirical_sigma=1.0,
)
# Illustrative configuration; choose a grid and coverage supporting the priors.
```

## Likelihood and deliberate legacy changes

The `spectrum` factor calls the tested iid
`marginalized_continuum_log_likelihood`, or
`gp_marginalized_continuum_log_likelihood` when both GP arguments are supplied.
Chebyshev degree defaults to
4, but both prior scales remain explicit. The likelihood analytically
integrates five coefficients per region; they are never sampled. This replaces
the old sampled linear `norm`/`slope`. Jitter is nonnegative additive absolute
noise in flux units, with variance `uncertainty**2+jitter**2`; it affects the
whitening, determinant normalization and conditional coefficient posterior.
The convenience model defaults only jitter to zero; a positive sampled jitter
prior is the user's choice. Masked NaN/Inf values retain the existing safe
low-level semantics. No second implementation of the Gaussian algebra exists.

The frozen baseline inspected was `jaxspec/numpyro_model.py` and
`examples/spectrum_fitting_example.ipynb`. Legacy sample sites included bounded
physical `*_scaled` values, q1/q2, per-order RV, shared/per-order wavres,
norm/slope, optional dilution, and GP `lna`, `lnc`, `lnsigma`. It used a
Matern-3/2 quasisep GP plus additive `exp(lnsigma)` jitter, combined observation
and fit masks, AutoLaplace/Adam initialization and NUTS initialized from scaled
site dictionaries. It always saved physical flux/residual arrays; optional GP
predictions and `get_mean_models` reconstructed plotting outputs.

Preserved here: atmosphere interpolation, combined rotation/RT macro/Gaussian
IP, relativistic effective RV, quadratic limb darkening, scalar/per-region RV
and resolving power, fixed versus sampled parameters, and the same NumPyro
model for point initialization and NUTS. GP likelihoods/predictions and their
optional single-star NumPyro integration are available as described below;
binary prior models and dilution remain separate migration steps. The stellar
relation options now reproduce the legacy single-star logg/vmic/vmacro
expressions using the helpers below, native physical site names and strict
configuration checks. All other physical priors remain explicit arguments.
To reproduce a legacy jitter prior, pass
`dist.TransformedDistribution(dist.Uniform(-10, -3), dist.transforms.ExpTransform())`;
the example HalfNormal prior is a documented synthetic choice, not legacy parity.

Empirical relations or conditional physical constraints can be expressed in a
custom model using the relation helpers below and the low-level
physical/likelihood helpers, or selected through the single-star options above.
Future SB2/SB-N priors can reuse the small sampling helper and shared SpecModel
component physics, rather than duplicate physical evaluation or likelihoods.

## Empirical relation helpers

`jaxstar.specfit.numpyro_model` defines two numerical relation helpers and a
NumPyro prior factory, also exported from `jaxstar.specfit`. Their coefficients
are preserved from
`jaxspec` commit `e7b2f30`, `src/jaxspec/numpyro_model.py`:

| Helper | Output | Applicability noted in the legacy source |
| --- | --- | --- |
| `physical_logg_max(teff)` | Upper bound on logg, in log10 cgs | 4500--7000 K |
| `empirical_vmic(teff, logg, feh)` | Microturbulent velocity, km/s | Teff > 5000 K and logg > 3.5 |
| `empirical_vmacro_valenti_fischer2005(teff, sigma=1.0)` | Nonnegative macroturbulence prior, km/s | No range specified in the legacy source |

Teff inputs are in kelvin and metallicity is in dex. The legacy BOSZ model
passed its `mh` coordinate to the vmic helper's `feh` argument; this convention
is explicit in the example below, with no automatic abundance conversion.
The macroturbulence location is the relation of
[Valenti & Fischer (2005), ApJS 159, 141](https://doi.org/10.1086/430500):
`3.98 + (teff - 5770) / 650` km/s. The normal scatter with default
`sigma=1 km/s` and truncation at zero reproduce the old single-star model's
prior choices; that scatter is not attributed to the paper. The legacy source
does not cite publications for the logg and vmic expressions.

These helpers accept scalar or broadcastable array inputs and support JIT,
gradients and vmap (applied to distribution calculations for the prior
factory). The logg/vmic helpers evaluate the original expressions without
clipping or range validation; the vmacro helper returns a `TruncatedNormal`
distribution with `low=0`. None of the helpers creates sample sites. A custom
model controls their application and checks applicability and spectral
coverage. For example, with `teff` and `mh` already sampled and a suitable
`logg_min` supplied:

```python
import numpyro
import numpyro.distributions as dist
from jaxstar.specfit import (
    physical_logg_max, empirical_vmic, empirical_vmacro_valenti_fischer2005,
)

logg = numpyro.sample("logg", dist.Uniform(logg_min, physical_logg_max(teff)))
vmic = numpyro.deterministic("vmic", empirical_vmic(teff, logg, feh=mh))
vmacro = numpyro.sample(
    "vmacro", empirical_vmacro_valenti_fischer2005(teff),
)
```

Use `sigma=...` to change the standard deviation of the normal before
truncation. Sigma is fixed by default, as in the old model; a custom model may
also sample it explicitly and pass that value to the helper. The author/year
suffix identifies the relation. Future alternatives, including a Teff/logg
relation, should have their own named prior factories with the same
distribution-returning convention. Sampling remains visible in the model
body. The current `model_single` options use the existing helpers directly;
custom models can choose other relations without adopting these switches.

## Custom model workflow

```python
import numpyro
import numpyro.distributions as dist
from jaxstar.specfit import single_star_params, marginalized_continuum_log_likelihood

def research_model(obs, specmodel, basis, fixed_atmosphere):
    teff = numpyro.sample("teff", dist.Uniform(5500, 6000))
    rv = numpyro.sample("rv", dist.Uniform(8, 17))
    jitter = numpyro.sample("jitter", dist.HalfNormal(0.01))
    sc = numpyro.sample("sigma_continuum", dist.LogNormal(-3.2188758, 0.7))
    atmosphere = {**fixed_atmosphere, "teff": teff}
    params = single_star_params(atmosphere, vsini=7.5, vmacro=3.2,
                               q1=0.36, q2=0.3, rv=rv, resolving_power=70000)
    flux = specmodel(params, obs.wavelength)
    numpyro.factor("spectrum", marginalized_continuum_log_likelihood(
        obs, flux, basis=basis, sigma_constant=0.1, sigma_continuum=sc, jitter=jitter))
```

Here fixed_atmosphere supplies all other axes. Nothing in SpecModel depends
on NumPyro distributions, and low-level continuum code remains prior agnostic.

## Same probabilistic model: MAP-like point and NUTS

```python
from numpyro_inferutils import find_map_svi
from numpyro.infer import MCMC, NUTS, init_to_value

point = find_map_svi(model_single, 0.01, 2000, rng_key=key0,
    p_initial=initial_site_dict, progress_bar=False,
    observation=obs, specmodel=specmodel, **kwargs)
mcmc = MCMC(NUTS(model_single, init_strategy=init_to_value(values=point)),
            num_warmup=300, num_samples=400)
mcmc.run(key1, obs, specmodel, **kwargs)
```

`numpyro-inferutils` 0.2.1 was locally available, inspected and used. Its
find_map_svi runs AutoLaplaceApproximation/Adam/Trace_ELBO and returns the guide
median without a covariance. This is a MAP-like point in transformed latent
space; do not equate it with a uniquely defined constrained-space MAP. The
helper's returned dict may include deterministic sites too. No separate custom
objective, least-squares prefit or optimizer wrapper was added to jaxstar.
The `spectral-inference` extra includes inferutils >=0.2.1,<0.3, matplotlib
and optional GP dependency tinygp >=0.3,<0.4;
the base model requires only already-declared NumPyro/JAX dependencies.

Pass large Observation/SpecModel arrays as dynamic arguments to ordinary
NumPyro/JAX evaluation where practical. The benchmark's instrumented SVI does
this. inferutils currently forwards model inputs as SVI static kwargs, so that
helper may close over them; jaxstar does not impose this on custom inference.
The benchmark also actually calls the external helper and compares results.

## Reproducible synthetic recovery and CPU/GPU experiment

```bash
python -m pip install '.[spectral-inference]'
JAX_PLATFORMS=cpu PYTHONPATH=src python benchmarks/benchmark_spectral_inference.py \
  --require-cpu --output-dir benchmark-results/spectral-inference-cpu
```

For the later GPU run use `JAX_PLATFORMS=cuda`, `--require-gpu`, and another
output directory. The user requested CPU validation for this milestone; GPU
timings are not measured here. Preserve version/device/dtype records when
comparing machines. No GPU optimization or least-squares implementation exists.

The default fixture has three regions (5000, 6000, 16000 Angstrom), 1201
log-uniform model samples over +/-300 km/s, and 384 observed samples per region
over approximately +/-136 km/s. Observed wavelengths use offset, slightly
nonuniform velocity coordinates. An explicit nearest-grid separation assertion
ensures every observed wavelength differs from internal pixels. Twelve Gaussian
lines have independent atmosphere-dependent depth patterns. This identifiable
synthetic library exercises the actual grid/SpecModel rather than claiming
realistic stellar spectral physics. Injected continuum has degree 2; analysis
has degree 4. Seed 20261004, uncertainty 0.004 and nonzero additive jitter 0.004
are used; masked stored flux/error contain NaN/Inf. All injected values and
example priors are in `examples/_spectral_inference_data.py`.

The atmosphere, vsini/vmacro, RV, per-region R, global jitter and global
sigma_continuum are inferred. q1/q2 and sigma_constant are fixed. The R priors
represent known calibration (3000 uncertainty around the injected values);
the macro prior is weakly informative. This acknowledges broadening
degeneracies rather than claiming all width contributions are independently
identified by the spectrum. The sc prior is LogNormal(log(0.04),0.7), chosen
for this injected 1--4% continuum, not a universal scientific recommendation.

The injected continuum coefficients are fixed, not random draws from a
hierarchical prior. Consequently sc=0.04 is the analysis hyperprior reference,
not a known injected physical truth. Its inferred value/interval is reported
without a truth-sigma offset. A smaller inferred scale is expected when the
degree-4 analysis shrinks two extra coefficients toward zero. This should not
be mistaken for a failure to recover an injected jitter or stellar parameter.

## CPU validation recorded on 2026-10-04

Final full suite: **630 passed, 1 skipped, 2 xfailed in 356.04 s** with the
frozen sibling references enabled. Seventeen new checks cover traces, fixed/
sampled and scalar/vector mappings, normalized factor parity, masks, JIT/AD,
invalid shapes, custom-model parity, 1D observations, tiny SVI/NUTS/inferutils,
float32/64 synthetic offset pixels, and real order-8 setup/artifact reuse.
Wheel/sdist builds and separate target installs succeeded; installed-wheel
JIT/value-and-grad, masks and unchanged x64 configuration were checked.

The full recovery benchmark was rerun alone on macOS ARM, Python 3.12.11,
JAX/jaxlib 0.6.2, NumPyro 0.19.0, inferutils 0.2.1, CPU float64. Results are
locally saved under `benchmark-results/spectral-inference-cpu-final/` (ignored
generated artifacts). Default three-region/384-pixel/1201-model-pixel problem,
12 latent dimensions, 15 continuum coefficients marginalized:

| Operation | Compile/setup | Warm execution |
| --- | --- | --- |
| Log density | 0.474 s lowering+compile | 3.633 ms median |
| Density + reverse-mode gradient | 1.215 s lowering+compile | 8.774 ms median |
| SVI, first initial point | 6.088 s init, 1.476 s scan compile | 13.213 s / 2000 steps |
| SVI, second initial point | 0.250 s init, 1.440 s scan compile | 13.132 s / 2000 steps |
| External find_map_svi | 14.854 s total after prior model runs | same estimate |
| NUTS 300 warmup | 96.508 s init+compile+warmup | not separately decomposed |
| First NUTS 400 samples | 24.711 s compile+sampling | not a compile-only time |
| Second NUTS 400 samples | previously compiled/adapted | 24.384 s; 16.404 draws/s |

Both SVI runs reached loss -4042.90302495; final 100-step variation was
1.5e-10 / 9.2e-9, and the estimates agreed within 0.00001 K in Teff and
0.033 in R. External helper agreement was <5e-16 relative to parameter scales.
NUTS median leapfrog count was 7, throughput 130.412 leapfrog steps/s,
mean acceptance 0.929, zero divergences and finite potential/samples. The one
short chain is not a convergence certificate. XLA's reported gradient temporary
memory was about 35 MiB; no dense pixel covariance or region-query cross product
was introduced. Large library arrays remained dynamic in density/instrumented
SVI checks.

| Parameter | Truth | MAP-like | Posterior median | 90% interval | Mean truth offset / SD |
| --- | --- | --- | --- | --- | --- |
| Teff [K] | 5770 | 5794.92 | 5794.92 | 5774.29--5813.87 | +1.990 |
| logg | 4.2 | 4.18281 | 4.18330 | 4.16256--4.20352 | -1.446 |
| mh | -0.12 | -0.09148 | -0.09061 | [-0.13685, -0.04988] | +1.022 |
| alpha | 0.12 | 0.11909 | 0.11919 | 0.11236--0.12577 | -0.255 |
| vsini [km/s] | 7.5 | 7.55033 | 7.56097 | 7.40765--7.69930 | +0.627 |
| vmacro [km/s] | 3.2 | 3.28012 | 3.24729 | 2.82445--3.65244 | +0.149 |
| RV [km/s] | 12.4 | 12.39171 | 12.39159 | 12.36259--12.42228 | -0.431 |
| R[0] | 68000 | 67614 | 67603 | 64017--71729 | -0.142 |
| R[1] | 71000 | 71780 | 71684 | 68278--75858 | +0.318 |
| R[2] | 73000 | 72042 | 72181 | 68462--76302 | -0.353 |
| jitter | 0.004 | 0.003845 | 0.003858 | 0.003594--0.004155 | -0.810 |
| sc | reference 0.04 | 0.017513 | 0.018465 | 0.012950--0.028945 | no injected truth |

Teff is about 2 SD above truth in this finite noise realization, slightly
outside the empirical 95% interval [5772.34,5817.18]. Other physical/noise
truths are inside their 90% intervals. Do not claim every short-chain interval
contains truth, or use this single realization to establish unbiasedness.
Physical-spectrum RMSE is 0.000876, reconstructed continuum RMSE 0.001129,
usable residual RMS 0.005506 versus injected noise scale 0.005657. Figures show
overlaid truth/recovery, structureless residuals at the injected noise level,
continuum recovery, both SVI histories, and posterior histograms. Degree-3/4
conditional coefficient means remain around 0.001 or less. The inferred sc
reflects shrinkage of this fixed low-order continuum, not recovery of a
generative scale.

On this modest CPU problem, SVI provides a practical initial point without a
second least-squares objective. This is evidence for retaining the normal
NumPyro workflow, not evidence that GPU/full real-data least squares will never
help. That decision remains open until those workloads are benchmarked. No
specialized optimizer or GP was added.

The script records log-density/gradient lowering and compilation, synchronized
warm evaluation, SVI initialization/scan compilation/optimization and losses
for two initial points, actual inferutils wall time, and NUTS initialization+
compilation+warmup followed by a second warmed posterior segment. Throughput
is reported for draws and leapfrog steps. It saves truth/point/posterior
intervals and sigma offsets, spectra/residual, continuum/convergence and
posterior plots, samples and JSON. One short chain is an end-to-end recovery
check, not a certified posterior analysis. Long recovery runs remain outside CI.

Continuum coefficient means/covariances in the diagnostic are conditional on
the representative nonlinear parameter point. They exclude uncertainty in the
physical parameters and hyperparameters; a fully marginal continuum band
would require reconstruction over their posterior draws. This distinction
matters when comparing individual conditional coefficients against truth.

## Real-data notebook groundwork

`examples/spectral_inference_real_setup.py` locates the existing frozen IRD
CSV and Coelho NPZ, converts the grid once to 1 km/s sampling and writes a common
artifact. Subsequent setup loads the common artifact. It creates Observation,
SpecModel, a cached basis, explicit fixed/sample priors and a native sample-site
initialization dictionary; no old flat vector/bounds bookkeeping is restored.
Example continuum/jitter scales are illustrative. Supply an approximate RV:

```bash
PYTHONPATH=src python examples/spectral_inference_real_setup.py --rv-initial 12.4
```

The numeric value above is only a setup demonstration, not a measured RV for
the real star. The command checks a finite density but performs no inference.
The legacy notebook fits orders 8 and 9. Only a matching order-8 grid is present
among the frozen supplied artifacts (order 9 extends beyond it). A polished
two-order notebook and inferred-parameter comparisons are deferred until that
input is supplied; no order-9 flux is invented, downloaded, edge-filled or
copied. No final inferred real-data equality is claimed, since continuum/noise
models intentionally differ. Deterministic frozen-reference tests remain the
strict numerical regression check. No SpecFit, CCF, iterative clipping,
automatic refitting or GP was implemented.

## Low-level spectral GP likelihood

The first GP implementation step now lives in `jaxstar.specfit.likelihood`.
The existing iid `marginalized_continuum_log_likelihood` moved here without
changing its signature or arithmetic. Package-level imports remain the same;
the former `jaxstar.specfit.continuum` import also resolves to the moved
function. Continuum construction and the iid coefficient posterior stay in
`continuum.py`. The subsequent NumPyro integration is described in the next
section; existing calls without GP arguments retain their iid likelihood.

Install `jaxstar[spectral-inference]` to include `tinygp>=0.3,<0.4` (no smolgp).
GP operations import tinygp lazily, so iid use does not require it. The test
extra includes tinygp for normal CI coverage. Installed tinygp 0.3.0 and the
benchmark environment's 0.3.1 expose compatible `QuasisepSolver`, `DirectSolver`,
`solver.solve_triangular` and `solver.normalization` APIs.

The public functions are:

- `prepare_gp_observation(observation)`;
- `gp_marginalized_continuum_log_likelihood(observation, model_flux, *,
  gp_amplitude, gp_scale, sigma_constant, sigma_continuum, degree=4,
  basis=None, jitter=0.0, gp_solver="auto")`;
- `gp_continuum_posterior(...)`, with the same arguments;
- `gp_conditional_mean(observation, model_flux, *, gp_amplitude, gp_scale,
  jitter=0.0, wavelength=None, gp_solver="auto")`.

The Matérn-3/2 kernel is
`k(d) = amplitude**2 * (1 + sqrt(3)*d/scale) * exp(-sqrt(3)*d/scale)`.
`gp_amplitude` is a finite positive scalar standard deviation in observation
flux units; `gp_scale` is a finite positive scalar correlation length in the
observation wavelength unit. Both are shared across independent regions.
Additive absolute jitter is finite nonnegative, scalar or `(n_region,)`, with
diagonal variance `uncertainty**2 + jitter**2`. No scientific priors or log
parameterization are imposed. Concrete invalid values fail before calculation;
traced values use the existing runtime JAX validation convention.

For each region C is the GP plus diagonal noise covariance. Whiten the residual
`y-X*mu` and prior-scaled continuum columns `X*S` together using tinygp, then
factor the coefficient-space matrix `H=I+Z.T@Z`. The stable quadratic is
`||r_white-Z*delta||**2 + ||delta||**2`; normalization includes C, H and every
usable pixel. No explicit inverse of C is formed. `gp_continuum_posterior`
reuses the same system and returns `ContinuumPosterior(mean, covariance)` with
the existing 1D/2D shape convention.

Exact GP masks require fixed conditioning sizes. `prepare_gp_observation`
computes usable integer indices from the concrete mask once; indices are static
PyTree metadata and all Observation arrays stay dynamic. Pass the returned
object inside JIT/NUTS. Eager calls or JIT closures over a concrete Observation
may pass it directly. An unprepared dynamic Observation inside JIT fails with
setup guidance. Rebuild the prepared object when a fitting mask changes.
Masked flux/error/model values, including NaN/Inf, are excluded before any
arithmetic and are absent from the conditioned covariance. Different usable
counts per region require no ragged public arrays or Observation redesign.

`gp_solver` is static: `"auto"` uses Quasisep on JAX's default CPU backend,
Direct on its GPU backend; explicit `"quasisep"`/`"direct"` override it. Select
explicitly if an executable uses a different backend from the default. This
heuristic comes from the representative float64 spectral benchmark, not a
universal performance claim, and changes no statistical model. Direct can use
substantially more memory. tinygp 0.3 has a mixed-dtype state-space limitation:
float32 inputs require caller-controlled x64=False; in x64 mode use float64.
The package gives setup guidance and never changes JAX precision configuration.

At one fixed nonlinear/GP parameter point, reconstruct the posterior mean by:

```python
data = prepare_gp_observation(obs)  # outside JIT / NUTS
posterior = gp_continuum_posterior(data, physical_flux, **gp_continuum_options)
mean_flux = apply_continuum(physical_flux, basis, posterior.mean)
gp_mean = gp_conditional_mean(data, mean_flux, **gp_noise_options)
data_space_mean = mean_flux + gp_mean
```

Prediction conditions only on usable pixels and defaults to all observation
wavelengths, including masked pixels. Optional prediction wavelengths retain
the region layout and may have a different common pixel count. The helper uses
tinygp triangular solves and the kernel cross-coordinate matrix-vector product
to return the latent residual GP mean. This is the joint linear-Gaussian
posterior-mean decomposition at fixed parameters; a single curve does not
propagate continuum/GP uncertainty or average over nonlinear posterior samples.
Fully masked regions contribute zero density, return the continuum prior and
have zero residual GP mean.

The low-level step added no GP sites to `model_single`, SpecFit, CCF, outlier
handling or explicit continuum sampling. The subsequent NumPyro step below
supplies physical GP values and user-selected priors.

Low-level commit 1 validation on CPU with JAX/jaxlib 0.6.2: the focused
likelihood suite passes all 57 tests with both tinygp 0.3.0 and 0.3.1;
the full suite reports 710 passed,
34 skipped and 2 xfailed. Tests cover exact masks, joint Gaussian dense
references, coefficient/GP means, both solvers, float32/float64, JIT, gradients,
an isolated low-level NUTS smoke test and unchanged iid behavior. Representative
float64 one-/two-region fixtures have maximum absolute dense-reference errors
of 1.42e-14 in log likelihood, 1.73e-17 in coefficient mean, 8.42e-19 in
coefficient covariance and 1.67e-16 in residual GP mean. Direct/Quasisep differ
by at most 1.42e-14 in log likelihood and 4.97e-14 in tested gradients.

A small CPU sanity check of the marginalized GP likelihood's `value_and_grad`
(1984 pixels, 1587 usable, degree 4, float64, auto/Quasisep) takes 0.567 s
for first trace/lowering/compilation and a 4.382 ms median warmed evaluation
over 10 synchronized calls. This measures the likelihood with model-flux and
GP/noise gradients, not a full SpecModel evaluation. GPU execution is not
available here; its auto/Direct selection is policy-tested but not newly timed.

## GP integration in model_single

The NumPyro integration adds `gp_amplitude=None`, `gp_scale=None`,
and `gp_solver="auto"` to `model_single`. Omit both physical GP
arguments for the unchanged iid path; supply both for the GP path. Supplying
only one is an error. Amplitude/scale can each be a fixed positive finite
scalar or a NumPyro distribution producing one. They have native
`gp_amplitude`/`gp_scale` sites, with no lna/lnc parameterization or package
prior. Static solver configuration is passed directly to the low-level density.
No GP predictions or continuum coefficients are stored in the inference trace;
`save_model_flux` and the returned physical parameter dict remain unchanged.

Normal GP use requires no manual mask preparation:

```python
from jaxstar.specfit import model_single

model_single(
    obs, specmodel, **kwargs,
    gp_amplitude=dist.HalfNormal(0.02), gp_scale=0.3,
    gp_solver="auto",  # illustrative flux/wavelength scales above
)
```

In GP mode, `prepare_gp_observation(observation)` runs internally immediately
before the GP likelihood. It excludes exactly the current mask from each GP
covariance. A new inference run with a changed Observation/mask automatically
prepares that mask; there is no user-managed preparation cache. The low-level
preparation helper remains available for custom likelihood/conditioning code.

The GP observation mask must be fixed when the model is traced. Ordinary
SVI/NUTS and JIT closing over concrete Observation data are supported; dynamically
tracing Observation/mask, including `jit_model_args=True`, is intentionally
unsupported and fails clearly. IID dynamic-data behavior is unchanged. A
closure-based JIT may still pass SpecModel's large arrays dynamically.

A real order-8 smoke check reuses `spectral_inference_real_setup.py`, the
existing prepared artifact and frozen IRD sample observation at stride 1.
It uses 1984 pixels (1546 usable, 438 excluded), the saved effective RV
-23.588105813398162 km/s, illustrative GP amplitude 0.015 and scale 1.65
Angstrom, and auto/Quasisep on CPU. JIT `value_and_grad` gives a finite joint
log density of 3871.612990411705 and finite gradients for all 12 latent scalar
inputs, including both GP hyperparameters. This is a parameter-point sanity
check, not a fitted result; no full real-data NUTS run was performed.
The new API needs no prepared-data argument: the supported JIT closes over
the concrete Observation and passes SpecModel's arrays dynamically.

The focused NumPyro suite passes all 36 tests on CPU. It covers ordinary
masked GP trace/eager gradients/concrete-Observation JIT in float32/float64,
both solvers, 20-step SVI and 8-warmup/8-sample ordinary NUTS, native GP site
names, direct low-level factor parity and the intentional dynamic-mask error.
IID density/gradient/site references and iid `jit_model_args=True` NUTS also
pass. The low-level mathematics and other model behavior are unchanged.
Fixed-GP factor differences from direct low-level evaluation are zero for
auto, Quasisep and Direct in the representative masked float64 fixture.
The full suite reports 732 passed, 34 skipped and 2 xfailed in 296.51 s.
