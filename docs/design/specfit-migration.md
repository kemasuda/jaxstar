# Provisional spectral-model migration design

Status: provisional engineering direction, 2026-10-02 (Asia/Tokyo).
This note guides staged migration from frozen `jaxspec` into `jaxstar`; it is
not a frozen public API or final specification. Revisit higher-level choices
as each consumer is migrated and benchmarked. Settle only the decisions needed
for the current milestone. Milestones 1, 2 and 2.5 are complete. Milestone 3a
implements only the deterministic single-component forward model; Milestone
3b composition and later fitting/inference remain out of scope. See the
[SpecModel note](specmodel-core.md) for the implemented API and validation.

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
  multiplicative Chebyshev continuum, zero-centered Gaussian shrinkage and
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
| `Observation` | Observed wavelength, flux, uncertainty, exclusion mask and labels; optional nominal instrument metadata. Useful for advanced/multiple observations, normally created internally for routine use. No required pixel edges. |
| `SpecModel` | M3a: intrinsic interpolation, combined broadening/limb darkening/Gaussian IP, relativistic RV and requested-wavelength sampling. M3b: generic component mixing and dilution. No observed flux/errors/masks, continuum or priors. |
| `SpecFit` | Lightweight observation/model bridge, region association, fitting masks, evaluation/likelihood conveniences and diagnostic delegation. Accepts arrays directly. Does not own priors or sampler logic. |
| Continuum helpers | Deterministic basis/correction, prior-scale inputs and conditional coefficient recovery; usable with sampled or marginalized coefficients. |
| Likelihood helpers | Independently callable Gaussian/marginalized densities and optional GP density/prediction. No sampling sites or sampler execution. |
| `numpyro_model.py` | Optional ready-to-use single-star and SB2 functions, including useful fixed/fitted and empirical-prior choices. Probabilistic structures may differ; physical evaluation is shared. |
| Inference/initialization helpers | CCF estimates, initial values, bounds/scaling and useful SVI conveniences; operate on ordinary callable NumPyro models. |

Users must be able to bypass `SpecFit` and combine intrinsic/component spectra,
continuum and likelihood helpers directly, including in joint MIST/spectral
models. Diagnostics and optional preparation/GP dependencies should not be
eager runtime requirements for the deterministic core.

### Milestone 3 assumptions and the M3a boundary

These assumptions guide the deterministic implementation without authorizing
M3b composition or later fitting layers:

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

### Explicit parity inventory after Milestone 3a

Milestone 3b must retain dilution, generic SB-N combination (including current
SB2 through the same component path), component spectra and flux ratios. M3a
requires exactly one component; removing that count restriction later must
reuse the component physics, rather than introduce `SpecModel2` or duplicate it.

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

None of these capabilities is implemented in M3a. The intentional omission of
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
    params = {**fixed_star, "atmosphere": {**fixed_atmosphere, "teff": teff},
              "instrument": {"resolving_power": R}}
    mu = model(params, data.wavelength)
    logp = marginalized_continuum_log_likelihood(
        data.flux, mu, uncertainty=data.uncertainty, mask=data.mask,
        basis=basis, coefficient_scales=(0.02, 0.02, 0.01), jitter=0.0,
    )
    numpyro.factor("spectrum", logp)
```

Here `fixed_star`/`fixed_atmosphere` are user-supplied physical parameters;
the coefficient scales are illustrative. Default NumPyro functions may call
these same helpers. A package inference wrapper is optional.

## 6. Continuum convention

The provisional correction is `C(x) = 1 + sum(a_k * T_k(x), k=0..degree)`
over a fixed wavelength-domain coordinate `x` in `[-1, 1]`.

```text
continuum_degree=None -> no correction
continuum_degree=0    -> T0
continuum_degree=1    -> T0,T1
continuum_degree=2    -> T0,T1,T2
```

Degree 2 therefore means three coefficients: normalization correction, slope
and curvature. All shrink toward zero under proper Gaussian priors; do not
add a duplicate free normalization. Exact configuration and scientifically
appropriate scales remain provisional.

Analytic marginalization is preferred by default; explicit coefficient
sampling remains possible. Use small coefficient-space Cholesky/Woodbury
algebra instead of a dense pixel covariance. Retain the model-dependent
covariance, log determinant and derivatives, apply masks to normalization as
well as residuals, and expose conditional coefficient recovery for predictions.

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
- **Milestone 3a — deterministic single-component model (current scope):** symmetric
   intrinsic/broadened/full outputs, combined broadening, relativistic RV,
   dynamic requested wavelength, JIT/gradients, coverage and frozen parity.
- **Milestone 3b — deterministic composition (not started):** generic SB-N,
   existing SB2 capability, dilution, component spectra and flux ratios.
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
- Exact `Observation` API and multiple-observation/region association surface.
- Continuum configuration, scale defaults and grouping/sharing surface.
- Final diagnostics/inference-helper organization and optional GP public API.
- Higher-level fitting orchestration under JAX, preserving the model's dynamic
  numerical library arrays rather than baking them into compile-time constants.
- Future detector integration and optical/empirical LSF response convention.
- Later Korg preparation API and full calibrated absolute-flux handling.

Higher-level public APIs remain provisional. Subsequent milestones should
refine their own contracts using migrated code and evidence.
