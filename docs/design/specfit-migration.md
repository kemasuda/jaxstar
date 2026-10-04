# Provisional spectral-model migration design

Status: provisional engineering direction, 2026-10-04 (Asia/Tokyo).
This note guides staged migration from frozen `jaxspec` into `jaxstar`; it is
not a frozen public API or final specification. Revisit higher-level choices
as each consumer is migrated and benchmarked. Settle only the decisions needed
for the current milestone. Milestones 1, 2, 2.5, 3a and 3b are implemented.
The deterministic model now supports generic fixed-count SB-N composition and
decomposition. Observation and low-level marginalized Chebyshev continuum
helpers are implemented; SpecFit and inference remain out of scope. See the
[physics note](specmodel-core.md) and [SB-N note](specmodel-sbn.md) for the API and validation.

## 1. Context

Frozen `jaxspec` is the scientific reference and regression oracle, not the
target architecture. Its characterization suite records Coelho/BOSZ/TLUSTY
interpolation, representative IRD and TLUSTY single/SB2 forward models,
physical gradients, dtype behavior, boundaries and a fixed legacy GP density.
The scientific baseline is `3b4ab9134b581e59ccb4f894b01866e8a4297eda`;
`52a255d` adds the characterization material. The review verified all 11
manifest hashes and ran 22 characterization tests successfully on CPU.

Useful spectral functionality is intended to move into `jaxstar.specfit`.
`jaxstar.grid` already supplies `Axis`, `Field`, `RectilinearGrid` and
`GridResult`, with scalar and explicitly named trailing payload fields, named
coordinates, broadcasting and JAX transform support. Milestone 2 supplies
private `_SpectralLibrary` results, common prepared NPZ storage/runtime loading
and legacy jaxspec compatibility adapters; see [the adapter note](specfit-loaders.md).

Reference material under the sibling `../jaxspec/` checkout:
`characterization/README.md`, `tests/characterization/`,
`characterization/reference_outputs/` and `benchmarks/`. Do not modify that
checkout or regenerate its reference outputs to make a migrated implementation
pass.

## 2. Agreed direction

- Keep `jaxstar.grid` independent of spectroscopy, I/O, MIST and NumPyro.
  Atmosphere parameters are interpolation axes; spectral region/wavelength
  samples are trailing payload dimensions. Wavelength resampling remains a
  spectral operation. Do not reproduce the old `SpecGrid` interpolation
  subclasses.
- Permit a specialized payload kernel behind the generic grid API. Reuse
  atmosphere brackets/corner weights across the spectrum, keep arrays as
  dynamic PyTree leaves and avoid explicitly stacking all corner spectra.
  Preserve existing scalar-grid/MIST behavior. Kernel choice, accumulation
  precision and packing should follow numerical and performance evidence.
- Prepare fitting spectra on log-uniform wavelength grids offline. Common
  storage remains able to hold arbitrary wavelength samples. Future `SpecModel`
  setup can validate this fitting contract once; it should not regrid native
  or linear spectra onto log wavelength per likelihood evaluation. Explicit
  `velocity_step` in km/s is the canonical preparation control, with
  `dlnlambda=velocity_step/299792.458`; it is numerical model sampling,
  independent of instrumental resolving power and observed detector pixels.
- Use a deterministic physical boundary compatible with future SB-N components.
  Component count may be static per compiled model. M3a uses a small dynamic
  PyTree for `SpecModel`; numerical library arrays remain ordinary arguments,
  with callable/configuration metadata static.
- Treat Korg as an offline/pre-inference grid builder. Inference consumes its
  prepared numerical grid entirely in JAX, without calling Korg per likelihood
  evaluation. Local-response interpolation is deferred.
- Make custom NumPyro models first-class. Optional default model functions
  provide convenience, while deterministic components remain directly usable.
  Routine usage should be at least as simple as current `jaxspec`.
- Put broadening behind one replaceable callable/operator boundary. Its default
  should retain the efficient combined rotation + radial-tangential
  macroturbulence + Gaussian instrumental broadening treatment. RV shifting is
  conceptually separate; do not add a general IP abstraction yet. Resolving
  power and RV may be scalar or per-region parameters. Detector-pixel
  integration remains separately deferred.
- Target normalized spectra first, while preserving access to useful existing
  unnormalized/intrinsic products and component spectra. General calibrated
  absolute-flux inference is outside the initial scope.
- Prefer Gaussian observational noise plus optional jitter, with a low-order
  multiplicative Chebyshev continuum, Gaussian shrinkage toward unity and
  analytic marginalization. Keep the legacy GP as optional compatibility
  functionality, including useful prediction behavior.
- Define clean domain/coverage behavior instead of adopting accidental legacy
  NaNs or silent clamping. Test intentional differences separately from valid
  interior compatibility; this does not authorize a general MIST invalid-field
  policy change.

## 3. Provisional object responsibilities

These are candidate boundaries, not commitments to exact names or signatures.

| Component | Candidate responsibility |
| --- | --- |
| Loaders/builders | Library schemas, wavelength units/medium, normalization, preparation, validation and provenance; construct the generic numerical grid. |
| `PreparedSpectra` or equivalent | Thin `RectilinearGrid` plus model wavelengths, region identifiers and spectral metadata. Loader return type; advanced construction may be exposed. Name/public visibility remain open. No second interpolation system. |
| `Observation` | Immutable measured wavelength, flux, uncertainty, data-level exclusion mask and optional region/order/exposure labels. No resolving power or other model/instrument parameters. No required pixel edges. |
| `SpecModel` | M3a: intrinsic interpolation, combined broadening/limb darkening/Gaussian IP, relativistic RV and requested-wavelength sampling. M3b: generic component mixing and dilution. No observed flux/errors/masks, continuum or priors. |
| `SpecFit` | Lightweight observation/model bridge, region association, fitting masks, evaluation/likelihood conveniences and diagnostic delegation. Accepts arrays directly. Does not own priors or sampler logic. |
| Continuum helpers (implemented) | Independent Chebyshev basis/correction, two-scale Gaussian prior, normalized analytically marginalized density and conditional coefficient recovery. No fitting object or sampler. |
| Likelihood helpers | Independently callable Gaussian/marginalized densities and optional GP density/prediction. No sampling sites or sampler execution. |
| `numpyro_model.py` | Optional ready-to-use single-star and SB2 functions, including useful fixed/fitted and empirical-prior choices. Probabilistic structures may differ; physical evaluation is shared. |
| Inference/initialization helpers | CCF estimates, initial values, bounds/scaling and useful SVI conveniences; operate on ordinary callable NumPyro models. |

Users must be able to bypass `SpecFit` and combine intrinsic/component spectra,
continuum and likelihood helpers directly, including in joint MIST/spectral
models. Diagnostics and optional preparation/GP dependencies should not be
eager runtime requirements for the deterministic core.

### Observed-data container

The small public container is available independently of fitting:

```python
from jaxstar.specfit import Observation

obs = Observation(
    wavelength=wave,
    flux=flux,
    uncertainty=err,
    mask=mask,                   # True excludes a pixel; None means all usable
    region=("blue", "red"),      # optional labels, one per region
    order=(8, 8),                # two segments may belong to the same order
    exposure="epoch1",           # optional identifying label for this observation
)
prediction = model(params, obs.wavelength)
```

`wavelength`, `flux`, `uncertainty` and the defaulted Boolean mask have identical,
nonempty `(n_pixel,)` or `(n_region, n_pixel)` shapes. No broadcasting, reshaping,
reordering or ragged arrays are introduced. Wavelength is finite, positive and
strictly increasing within every row, including excluded pixels, so it is a
valid evaluation input for `SpecModel`. Units/medium must match the chosen
prepared model library. Flux and uncertainty are finite on usable pixels, with
positive uncertainty. Excluded flux/uncertainty may be nonfinite or invalid and
are preserved without replacement. An entirely excluded observation is allowed;
a later fitting layer can require usable data. Numeric 0/1 masks are safely
converted to Boolean without inversion; other numeric/string casts are rejected.

NumPy/list inputs are copied into read-only arrays without precision promotion
or truncation; JAX arrays retain their dtype/device and are already immutable.
No global x64 setting is changed. The four numerical arrays are dynamic PyTree
leaves. JAX transformations follow the caller's precision setting (including
float64 canonicalization when x64 is disabled). Construction validates concrete
data before JIT. Optional `region` and `order` are independent static tuples of
string/integer identifiers, one per row (also length one for 1D data); duplicates
are allowed. `exposure` is one nonempty static string or None. Convenience
properties are `shape`, `ndim`, `n_regions`, `n_pixels` and `valid = ~mask`.

The responsibility split is:

- Prepared spectral grids: model-library data and model wavelengths.
- `SpecModel`: deterministic physical forward model, still callable without an
  `Observation` as `model(params, wavelength)`.
- `Observation`: measured arrays and their data-level exclusion mask/labels.
- Future `SpecFit`: observation/model association, a separate adjustable fitting
  mask, continuum and likelihood. This layer remains unimplemented.

`Observation` intentionally does not store resolving power. Resolving power is
a forward-model/instrument parameter and may be fixed or inferred independently
of the observed spectral arrays. No model parameters, plotting, file I/O,
likelihood, continuum, inference or multi-exposure framework is added here.

### Milestone 3 assumptions and the deterministic boundary

These assumptions guide the deterministic implementation without authorizing
later fitting layers:

- Fitting grids are prepared on log-uniform wavelength sampling. Common NPZ
  storage still supports arbitrary valid arrays. Offline raw preparation or a
  one-time prepared-grid conversion supplies the fitting grid. Use an explicit
  `velocity_step`; `pixels` is retained only as a secondary compatibility/test/
  special-purpose count control and is mutually exclusive with velocity spacing.
- `SpecModel` is deterministic only. Evaluation conceptually has the form
  `model(params, wav_obs)`, where `params` is a nested dict/PyTree and `wav_obs`
  is an evaluation-time dynamic argument.
- Observed flux, uncertainty, masks, continuum, likelihood and NumPyro do not
  belong in `SpecModel`.
- RV and resolving power may be scalar or per-region arrays. Component
  representation should permit future SB-N; the number of components may be
  static per compiled model.
- Broadening uses one replaceable callable/operator boundary. The default
  preserves combined rotation, radial-tangential macroturbulence and Gaussian
  instrumental broadening. RV shifting remains separate. No separate general
  IP abstraction is selected yet.
- Future custom broadening, including numerical/HEALPix implementations, should
  be possible without rewriting the whole model. No such operator is added now.
- M3a uses a small model PyTree containing the existing dynamic grid/carrier
  leaves and velocity arrays. Pass it explicitly to JIT to avoid capturing
  large library arrays as compile-time constants. Higher-level fitting object
  compilation choices remain deferred.

The intended offline workflow is:

```text
requested fitting wavelength range
    -> choose numerical model velocity sampling (e.g. dv ~ 1 km/s)
    -> prepare log-uniform spectral grid offline
    -> save common spectral-grid artifact
    -> future SpecModel
```

About 1 km/s is a useful historical IRD/BOSZ reference: roughly 4000 model
samples over a typical order gave a numerical sampling resolution of order
`3e5`. It is neither a package default nor universally optimal. Choose dv from
the velocity scales needed for broadening/RV calculations and validate
accuracy/performance in Milestone 3. Never infer it from detector pixel counts,
and do not use `R`/`resolving_power` as an alias for numerical sampling. The
instrumental resolving power remains a separate future model parameter.

### Explicit parity inventory after Milestones 3a and 3b

Milestone 3b implements dilution, generic SB-N combination (including current
SB2 through the same component path), component spectra and relative flux weights.
It reuses the M3a component physics; no `SpecModel2` or duplicate pipeline exists.
Weights are nonnegative relative stellar continuum fluxes, normalized among stars;
dilution remains the featureless fraction of total light. Both can be region-specific.
All components use one library; heterogeneous libraries and pixel-dependent light
ratios are deferred. The tuple length is static under JIT and numerical values dynamic.

Later SpecFit/fitting layers must preserve:

- Observed/model region association and coverage validation, initial observation
  masks, iterative fitting/outlier masks.
- Current linear continuum compatibility and the new Chebyshev path.
- The actual fitted data-space model including continuum: separately expose
  `fit.physical_model(...)`, `fit.continuum(...)` and `fit.model(...)`, or an
  equivalent API. Residual/model plotting must show the fitted data-space model.
  Analytically marginalized Chebyshev coefficients must still permit an
  appropriate conditional continuum reconstruction for visualization.
- Single-star CCF and binary CCF behavior; residual/model plotting and diagnostics.
- Gaussian noise with optional jitter; optional GP likelihood and GP predictions.
- Useful empirical vmic/vmacro relations, q1/q2 limb-darkening convenience and
  physical-logg constraints.
- Default NumPyro single/binary models, custom user-written NumPyro models and
  SVI/initialization helpers.

None of these fitting capabilities is implemented in M3a/M3b. The intentional omission of
legacy norm/slope from the physical model does not remove the requirement to
recover and plot continuum-corrected fitted spectra in later SpecFit.

## 4. Capability-preservation principle

> Internal redesign is welcome; existing useful scientific capability should
> not disappear silently.

The first migration should account for these groups, including capabilities
outside the existing characterization suite:

- Coelho/BOSZ/TLUSTY schemas and preparation; optional offline iSpec synthesis
  and conversion; access to useful intrinsic/unnormalized products.
- Single/SB2/SB-N deterministic modeling; rigid rotation, radial-tangential
  macroturbulence, limb darkening, RV and instrumental broadening; fixed,
  order-dependent or inferred resolving power; dilution and component light
  fractions.
- Multi-order behavior and custom parameter sharing across observations;
  separate observation/fitting masks, iterative masking and mask extension;
  single/SB2 CCF initialization.
- Custom NumPyro composition and optional default single/SB2 models, including
  useful empirical relations and fixed/fitted options; explicit continuum
  fitting and optional GP likelihood/prediction compatibility.
- Useful initialization, SVI, diagnostics, posterior predictions and preparation
  helpers. Preserve the workflows rather than stale/broken wrapper names.

Characterization gaps, such as SB-N, CCF, iterative masking, GP prediction and
inference options, require additional migration tests. They do not imply that
those capabilities are unimportant. New preparation must make missing-model
substitution explicit rather than silently copying another atmosphere node.

Differential rotation is outside current migration requirements and acceptance
tests, following the user's clarification. It may be reconsidered after its
planned changes as a separately validated later addition; no diffrot-specific
extension design or tests are required now.

## 5. Proposed simple usage philosophy

Illustrative only: imports, names and arguments below are not frozen API and
do not describe currently implemented functionality.

```python
grid = load_spectral_grid("prepared_log.npz")
fit = SpecFit(
    grid,
    wavelength=wave,
    flux=flux,
    uncertainty=err,
    mask=mask,
)
mu = fit.flux(params, resolving_power=R)
logp = fit.log_likelihood(mu, jitter=jitter)
```

The constructor may internally create the observation container, physical
model, evaluation plan and continuum basis. Resolving power can instead have
an optional fixed fallback. Custom models can sample independent resolutions
or compute per-order values from a smooth parameterization; a nonstationary
LSF within an order remains a separately validated extension.

A custom NumPyro model can use the same deterministic pieces without `SpecFit`:

```python
# Once, outside inference: model and basis are prepared from grid/geometry.
def research_model(model, data, basis):
    teff = numpyro.sample("teff", dist.Uniform(5500, 6250))
    R = numpyro.sample("R", dist.Uniform(65000, 75000))
    component = {**fixed_component, "atmosphere": {**fixed_atmosphere, "teff": teff}}
    params = {"components": (component,), "instrument": {"resolving_power": R}}
    mu = model(params, data.wavelength)
    logp = marginalized_continuum_log_likelihood(
        data, mu, basis=basis, degree=4,
        sigma_constant=0.1, sigma_continuum=0.02,
    )
    numpyro.factor("spectrum", logp)
```

Here `fixed_component`/`fixed_atmosphere` are user-supplied physical parameters;
the coefficient scales are illustrative. Default NumPyro functions may call
these same helpers. A package inference wrapper is optional.

## 6. Regularized multiplicative Chebyshev continuum (implemented)

The subsequent minimal single-star NumPyro milestone is documented in
[spectral-inference.md](spectral-inference.md). It uses the same marginalized
continuum and additive jitter with `model_single` for SVI and NUTS, without
SpecFit, CCF, iterative masking or GP. GP and empirical/physical convenience
constraints remain explicit later compatibility work.

After stellar component composition, apply one continuum per observed region:
`f_pred = f_model * C(wavelength)`, with `C(x) = sum(a_k*T_k(x), k=0..degree)`.
The free constant term is included once; there is no additional normalization.
Default degree is **4**, giving five coefficients per region. This is a modest
default whose mild overparameterization is controlled by the prior, rather
than automatic degree selection. Any nonnegative static integer degree works.
The prior is `a0 ~ Normal(1, sigma_constant)`, and every `ak>0` independently
has `Normal(0, sigma_continuum)`. Thus `mu=[1,0,...]` and covariance is
`diag([sigma_constant**2, sigma_continuum**2, ...])`. This is equivalent to
the earlier provisional `1 + sum(delta_a_k*T_k)` centered at zero, but expresses
the actual constant coefficient directly. Positivity is deliberately not
enforced: clipping or constrained coefficients would destroy the linear
Gaussian model and its analytic marginalization.

Public low-level helpers are exported from `jaxstar.specfit`:

```python
basis = chebyshev_basis(obs.wavelength, degree=4)  # cache once for fixed geometry
model_flux = model(params, obs.wavelength)         # unchanged physical API
design = continuum_design_matrix(model_flux, basis)
prior = continuum_prior(sigma_constant=s0, sigma_continuum=sc,
                        degree=4, n_regions=obs.n_regions if obs.ndim == 2 else None)
logp = marginalized_continuum_log_likelihood(
    obs, model_flux, basis=basis, degree=4,
    sigma_constant=s0, sigma_continuum=sc, jitter=jitter,
)
posterior = continuum_posterior(
    obs, model_flux, basis=basis, degree=4,
    sigma_constant=s0, sigma_continuum=sc, jitter=jitter,
)
continuum = evaluate_continuum(basis, posterior.mean)
prediction = apply_continuum(model_flux, basis, posterior.mean)
```

Both prior scales must be supplied explicitly; no scientific scale default is
chosen. Each is scalar or `(n_region,)`, permitting later fixed or inferred
hyperparameters. Scales must be finite positive, including the nonconstant
scale for degree 0. Concrete invalid inputs raise ValueError; traced invalid
inputs raise a JAX runtime error with the same validation message. There is
no arbitrary covariance API, degree-dependent shrinkage, or numerical ridge.
`ContinuumPrior` is a small NamedTuple with `mean`, diagonal `scale` and a
`covariance` property; `ContinuumPosterior` is a NamedTuple with `mean` and
`covariance`. Both are ordinary JAX PyTrees containing numerical arrays.

For each row separately, the coordinate is exactly
`x=2*(wavelength-first)/(last-first)-1`. Endpoints map to -1 and +1, including
masked endpoint wavelengths; a one-pixel region uses x=0. Wavelengths are
finite, positive, strictly increasing as in Observation. The basis uses the
recurrence `T0=1`, `T1=x`, `Tk=2*x*T(k-1)-T(k-2)` and has shape
`obs.shape+(degree+1,)`. Explicit coefficient shapes are `(degree+1,)` for
1D data or `(n_region,degree+1)` for 2D data. The latter retains the row axis
even for one region. Ambiguous coefficient/flux broadcasting is rejected.
Independent coefficients belong to observed regions, never individual stars.

Both density and posterior functions accept optional `jitter=0.0`, scalar or
`(n_region,)` (including a length-one region vector for 1D data). It is additive
absolute Gaussian noise in the same flux units as observation uncertainty:
`D[i,i]=sigma_obs[i]**2+jitter[region]**2`. Finite nonnegative values are required;
per-pixel arrays, fractional/model-scaled jitter and arbitrary broadcasting are
not supported. Jitter is a likelihood nuisance argument, never stored in
Observation or SpecModel. `jaxstar.specfit` imposes no prior on it; a later
user-written NumPyro model can choose an appropriate positive prior.

For one region, let `X=f_model[:,None]*basis`, `D=diag(error**2)` on usable
pixels, and `S=diag(prior.scale)`. Standardize to `Z=D^(-1/2)*X*S`,
`r=D^(-1/2)*(y-X*mu)` and `H=I+Z.T*Z`. Cholesky solves yield
`delta=H^(-1)*Z.T*r`. The full normalized density is

```text
q = ||r-Z*delta||^2 + ||delta||^2
logdet = 2*sum(log(error)) + 2*sum(log(diag(chol(H))))
logp_region = -0.5*(q + logdet + N_usable*log(2*pi))
logp = sum(logp_region)
```

The non-subtractive expression for q avoids Woodbury cancellation. The
`error` above is always the effective `sqrt(sigma_obs**2+jitter[region]**2)`,
evaluated with stable `hypot`. It enters both whitening and the Gaussian
variance determinant, so increasing jitter incurs the full normalization
penalty. The conditional posterior uses exactly the same effective covariance.
Default zero jitter reproduces the previous formulation. The
standardized determinant includes the full continuum-prior determinant
contribution, equivalent to `logdet(D)+logdet(Lambda)+logdet(Lambda^-1+X.T*D^-1*X)`.
No normalization terms are dropped, so prior-scale gradients are meaningful.
The conditional mean is `mu+S*delta`, and covariance is `S*solve(H,I)*S`, using
Cholesky solves rather than an explicit matrix inverse. Cost is approximately
`O(n_pixel*(degree+1)^2)` plus the small coefficient solve. The dense system is
only `(degree+1,degree+1)` per region; no pixel-by-pixel covariance is built.
For ten regions and degree 4, marginalization removes 50 continuum coefficients
from a future nonlinear sampler. Conditional recovery still allows the actual
data-space spectrum to be reconstructed and plotted after fitting.

Masked entries are neutralized **before** any division, multiplication, square
or log: excluded y/model flux become zero, and excluded uncertainty becomes
one, with their jitter contribution set to zero before computing effective
errors. They have zero design/residual weight and do not count toward N or the
variance determinant. Masked NaN/Inf flux/error values therefore cannot poison
values or gradients. Fully masked regions contribute zero likelihood and
return the prior. This does not introduce a separate adjustable fitting mask.

Numerical helpers preserve normal JAX float32/float64 promotion without a
global x64 change and support JIT, grad/value_and_grad and vmap at fixed shapes.
Degree and optional region-count setup are static; flux, observations and
scales remain numerical inputs. Tests compare against dense Gaussian references,
posterior linear algebra, finite differences, masked-NaN gradients and the SB-N
physical path. `benchmarks/benchmark_continuum.py` provides a synchronized CPU/GPU
sanity check for ten regions and ~2000 pixels with coefficient/HLO shape checks;
run with `--jitter 0` and `--jitter 0.01` to compare noise choices and include
jitter differentiation. There is no SpecFit, GP, inference, or change to
SpecModel/Observation.

The self-contained [synthetic continuum notebook](../tutorials/specfit/continuum.ipynb)
uses a fixed seed and `model_flux=1`: degree-2 truth, degree-4 analysis, masked
NaN/Inf data and known additive jitter. Three figures show conditional continuum
recovery, all coefficient means/uncertainties (including unused a3/a4), and
separate marginal-likelihood curves for prior scale and jitter. Higher-order
coefficients may be compatible with zero without being exactly zero. The
curves retain Gaussian normalization and integrate all five coefficients
analytically, illustrating possible future hyperparameter inference without
adding NumPyro or an optimization workflow. Generated figures and executed
notebook outputs stay in ignored `benchmark-results/continuum-demo/`.

## 7. Milestones

- **Milestone 1 — generic payload interpolation (complete):** explicit trailing
   payload support in `jaxstar.grid`, preserving scalar/MIST behavior and testing
   shapes, dtype, broadcasting, JIT, gradients, singleton axes and boundaries.
- **Milestone 2 — library adapters/data contract (complete):** Coelho/BOSZ/TLUSTY preparation,
   common storage/runtime loading, metadata and intrinsic evaluation; compare
   against frozen grid references.
- **Milestone 2.5 — log-uniform fitting-grid preparation (complete):** direct raw
   preparation and one-time library-agnostic conversion, numerical sampling
   validation, common artifact round trips and synthetic accuracy tests.
   Actual full raw-library integration/smoke testing remains a later validation
   TODO on a machine with those libraries, not a blocker for this merge.
- **Milestone 3a — deterministic single-component model (complete):** symmetric
   intrinsic/broadened/full outputs, combined broadening, relativistic RV,
   dynamic requested wavelength, JIT/gradients, coverage and frozen parity.
- **Milestone 3b — deterministic composition (implemented):** generic fixed-count SB-N,
   existing SB2 capability, dilution, component/decomposition spectra and relative
   stellar flux weights; shared physics, frozen parity and N-scaling benchmark.
- **Observed-data container (implemented):** immutable Observation with validated
   measured arrays, masks, identifiers and dynamic PyTree leaves.
- **Low-level continuum milestone (implemented):** regularized multiplicative
   degree-4 Chebyshev model, full Gaussian marginalization and conditional
   coefficient recovery, independently of any SpecFit/inference layer.
- **Milestone 4 — fit bridge/probabilistic compatibility:** simple construction, masks,
   custom composition, optional GP and default model functions.
- **Milestone 5 — intentional statistical defaults:** Gaussian/jitter and marginalized
   Chebyshev, validated independently from legacy likelihood/continuum parity.
- **Milestone 6 — capability completion:** CCF, initialization, iterative masks, diagnostics,
   preparation and research composition examples; close characterization gaps.
- **Milestone 7 — CPU/GPU performance/integration gate:** identical inputs and objectives,
   synchronized warm full value-and-gradient timings, compilation/transfer
   costs and peak memory. Performance is measured, not assumed from layout.
- **Milestone 8 — later validated additions:** Korg preparation, possible revised differential
   rotation, and detector/LSF extensions when separately scoped.

Physical compatibility initially preserves interior interpolation, library
parameter meanings, stored values/dtypes, characterized kernels/convolution,
Doppler convention, point sampling and single/SB2 gradients. Use the frozen
README's tolerances and scaled derivative comparisons. A deterministic legacy
continuum adapter can isolate physical parity. Clean domain behavior, new
likelihood/continuum defaults and ownership changes have separate acceptance
tests. Keep existing MIST first-use download behavior and numerical contracts.

## 8. Current open/deferred decisions

These are not yet frozen:

- Final `SpecFit` API and exact spectral-container name/public visibility.
- Multiple-observation/region association and observation-construction conveniences
  in the future `SpecFit` layer; the data-only `Observation` API is implemented above.
- Future fitting-layer continuum configuration and scientifically appropriate
  prior scales; the low-level independent-region/two-scale API is implemented above.
- Final diagnostics/inference-helper organization and optional GP public API.
- Higher-level fitting orchestration under JAX, preserving the model's dynamic
  numerical library arrays rather than baking them into compile-time constants.
- Future detector integration and optical/empirical LSF response convention.
- Later Korg preparation API and full calibrated absolute-flux handling.

Higher-level public APIs remain provisional. Subsequent milestones should
refine their own contracts using migrated code and evidence.
