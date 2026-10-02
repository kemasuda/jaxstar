# Milestone 3a: deterministic single-component SpecModel

Scope: atmosphere interpolation, combined broadening, relativistic RV and
evaluation-time wavelength sampling. No observation container, continuum,
likelihood, inference or component combination is implemented here.

Setup API: `SpecModel(spectra, *, vmax=50., broadening_operator=None)`. `vmax`
retains the frozen model's finite velocity-kernel support control (km/s), rather
than introducing automatic padding. The default operator retains the frozen
combined rotation/radial-tangential macroturbulence/Gaussian calculation.

The replaceable callable accepts `(wavelength, flux, *, broadening,
resolving_power, velocity_grid)` and returns `(safe_wavelength, safe_flux)`.
This pair deliberately allows custom disk integration without requiring an
analytic kernel. Default convolution returns only its valid interior; RV and
requested-wavelength interpolation then share the same coverage check. Custom
operators declare their safe output coverage by the returned wavelengths.

The model is a small PyTree: the spectral carrier and velocity arrays are
dynamic leaves, the callable/configuration is static metadata. Prefer
`jax.jit(lambda model, params, wave: model(params, wave))` to keep library arrays
as ordinary arguments. Stage methods are not implicitly jitted with static self.

Data-dependent checks raise ValueError in eager evaluation; under JIT a failing
check calls a host error callback on the failure branch, yielding an XLA runtime
exception containing the same explanation. This preserves dynamic `wav_obs`
and clear coverage failure without requiring a separate checkify invocation
from users. Successful compiled checks do not call Python per pixel.

## Public stage and parameter API

```python
from jaxstar.specfit import SpecModel, load_spectral_grid

model = SpecModel(load_spectral_grid("prepared_log.npz"))
params = {
    "components": ({
        "atmosphere": {"teff": 5812.5, "logg": 4.2, "feh": -0.15, "alpha": 0.12},
        "broadening": {"vsini": 6.3, "vmacro": 3.1, "u1": 0.5, "u2": 0.2},
        "rv": 12.4,
    },),
    "instrument": {"resolving_power": 70000.},
}
intrinsic = model.intrinsic(params, wav_obs)
broadened = model.broadened(params, wav_obs)
full = model.full(params, wav_obs)
assert SpecModel.__call__ is SpecModel.full
```

`intrinsic` performs atmosphere interpolation and requested-wavelength sampling;
`broadened` adds combined stellar/Gaussian broadening without RV; `full` then
applies relativistic RV before sampling. No stage applies continuum or dilution.
The component tuple/list must have length one. Atmosphere keys exactly match the
loaded named axes; no source-library dispatch exists in the physical model.
Atmosphere values must be scalar, finite and inside the library domain. This
avoids a region-query by region-payload cross-product. Stages only require their
used parameter groups: intrinsic templates need no broadening/instrument/RV,
and broadened templates need no RV.

All of vsini, vmacro, u1, u2, effective RV and resolving power accept scalar or
`(n_region,)`; every other shape is rejected. Speeds are km/s; vsini/vmacro must
be finite/nonnegative, and the quadratic limb law must have positive disk
normalization `1-u1/3-u2/6`. There is no q1/q2 conversion. Resolving power is
positive; infinity disables the Gaussian contribution. The Gaussian standard
deviation remains `c/R/2.354820`, using the frozen FWHM coefficient.

RV means the effective line-of-sight shift supplied by the caller. Its finite
value must satisfy `abs(rv)<c`. The frozen relativistic factor is
`sqrt((1+rv/c)/(1-rv/c))`, multiplying source wavelengths; positive RV shifts
features redward. Systemic/orbital RV and wavelength-zero-point interpretations
remain outside this layer.

Requested wavelength is dynamic evaluation input with `(n_region,n_pixel)`
shape, finite positive strictly increasing samples and at least one pixel.
Single-region 1D input returns 1D output. Rows correspond positionally to the
library's region identifiers, with no observed/model association machinery.
Units and medium must already match the stored library. Different requested
pixel counts require separate evaluations; no ragged representation is added.

## Model grid, coverage and deliberate legacy differences

Setup runs M2.5 `is_log_uniform` over every region with its documented dtype-aware
tolerance. Unsuitable grids fail with prepare/one-time-conversion guidance.
Common storage remains arbitrary-wavelength capable; no native-to-log fallback
exists inside any model stage.

Velocity step is `c` times the endpoint-mean natural-log step. Unlike the frozen
median-of-adjacent-steps construction, this is robust to float32 wavelength
quantization; on float64 log grids the difference is at rounding precision.
As in frozen jaxspec, the first region fixes a rounded integer half-support,
shared across regions; individual region velocity steps set their actual
support widths. `vmax=50` is retained as a support setting, not automatic
coverage/resolution selection. TLUSTY's recorded broad case uses `vmax=1500`.

Default broadening ports the validated Bessel approximation and Fourier
radial-tangential/rigid-rotation/Gaussian treatment with `Nt=500`. It combines
the contributions before one convolution per region. `vsini` exceeding actual
finite support fails with guidance to increase vmax; callers must also choose
sufficient support for macroturbulent/Gaussian tails, as in the frozen model.
No automatic broadening/RV parameter-range padding is inferred.

Valid convolution excludes the half-support at both model-grid edges. It agrees
with frozen `same` convolution on that interior, while never exposing its
zero-padded edges. After optional RV, requested wavelengths must lie within
the returned safe coverage. Failure raises clearly; interpolation also uses
NaN outside sentinels rather than clamped endpoints. Intrinsic requires only
stored coverage. No legacy boundary clamping/NaNs are reinstated. Atmosphere
domain errors and nonfinite intrinsic/operator flux also fail explicitly.

Offline node resampling versus old atmosphere-then-wavelength interpolation
changes floating-point accumulation order. Comparisons distinguish those small
differences from physical regressions. Exact old sampling is reproduced offline
only in the reference helper (count mode plus its historical endpoint trim);
routine fitting preparation continues to use explicit velocity_step.

## Validation and repository examples

The executed [SpecModel tutorial](../../examples/tutorials/specmodel.ipynb)
walks through a Coelho common artifact, region/pixel sampling, the full parameter
dictionary, explicit intrinsic/broadened/full calls, the real IRD sample overlay,
individual parameter responses, per-region RV and frozen comparison. Figures
are inline. Its generated artifact/cache lives in the ignored repository-relative
`examples/tutorials/.specmodel-data/`, with no `/private/tmp` dependency. The
sibling frozen characterization data is required for this tutorial's sample
and parity comparison; normal runtime use needs only a prepared artifact.
`specmodel_diagnostic.py` remains the non-interactive three-library batch
alternative, sharing small plotting helpers and the read-only reference helper.

Tests cover stage equality/identities, line-width and RV properties, scalar and
region-specific parameters, invalid shapes/domains, eager/JIT coverage failures,
non-log setup rejection, custom callable outputs, dynamic numerical leaves,
reverse-mode gradients/finite differences, outer query vmap and x64-disabled
execution. Frozen comparisons use Coelho/BOSZ/TLUSTY prepared sample data and
immutable source/input SHA256 checks.

Saved IRD/TLUSTY full spectra and physical gradients are compared against
`frozen_main.npz`, undoing the fixed legacy continuum outside SpecModel. That
test-only reconstruction is not a continuum implementation. Intermediate
stages, generalized per-region broadening and BOSZ full/gradients use read-only
frozen numerical source because matching saved stage/BOSZ forward artifacts do
not exist. No generator or sibling bytecode-cache writes are performed.

```bash
MPLCONFIGDIR=/private/tmp/jaxstar-mpl-cache JAXSPEC_REFERENCE_ROOT=../jaxspec \
  PYTHONPATH=src python -m pytest -q
MPLCONFIGDIR=/private/tmp/jaxstar-mpl-cache PYTHONPATH=src \
  python examples/specmodel_diagnostic.py --reference-root ../jaxspec \
  --output-dir /private/tmp/jaxstar-specmodel-coelho
MPLCONFIGDIR=/private/tmp/jaxstar-mpl-cache PYTHONPATH=src \
  python benchmarks/benchmark_specmodel.py --regions 10 --model-pixels 4000 \
  --output-pixels 2000 --dtype float32
```

The diagnostic prepares/saves/loads a common log artifact once, then plots the
real thinned observed sample snapshot alongside intrinsic/broadened/full,
legacy-versus-new physical prediction/difference, and independent vsini/vmacro/
R/RV variations. CLI controls accept shared or comma-separated region values.
It does not fit, mask outliers or implement continuum. The vertical display
range focuses on physical line profiles, so extreme observed outliers do not
flatten the plots. IRD has two segments of the supplied order 8, not two
independent observed orders. TLUSTY has only the blue arm; true distinct-order
or red/NIR validation needs additional data. Synthetic tests cover independent
regions without inventing observed spectra.

The benchmark separately reports first trace/compile, synchronized warmed
forward and value_and_grad medians, compiled memory estimates and model sizes.
Its nonlinear weighted flux functional is deterministic, with no likelihood.
It uses the active backend and is runnable later on GPU. No CPU-only physical
code, custom VJP/rematerialization or GPU optimization is introduced.
The [benchmark instructions](../../benchmarks/README.md) define the shared
B=1, ten-region float32 CPU/GPU case, matching interpolation-only dimensions,
explicit device placement, synchronized AD timing and final delimited JSON.
CPU uses the existing JAX 0.6.2 environment; the separate GPU environment will
use 0.11. Environment versions are recorded so comparisons retain that context.

## Deferred parity

M3b: dilution, generic SB-N including existing SB2, component spectra and flux
ratios. Later SpecFit: region association, observation and iterative masks,
linear/Chebyshev continuum and fitted data-space model reconstruction, CCF
single/binary support, plotting/diagnostics, Gaussian/jitter and optional GP
likelihood/predictions, empirical turbulence relations, q1/q2 and physical-logg
conveniences, default/custom NumPyro models and SVI/initialization. The full
inventory and fitted physical/continuum/data-space API requirement are in
`specfit-migration.md`. None is implemented here.

## Recorded M3a checks (2026-10-02)

Full repository suite, with the frozen reference available:
**342 passed, 1 skipped, 2 xfailed in 108.42 s**. M3a adds 62 cases, including
15 frozen-reference integration cases. The skip is the opt-in full MIST test;
the two expected failures are unchanged pre-existing tests. Float32, float64,
actual x64-disabled execution, explicit-model JIT, callable-model JIT and
reverse-mode differentiation pass. Reference flux checks use rtol=5e-6,
atol=1e-6; scaled physical gradient checks use rtol/atol=3e-4.

Dense diagnostic comparisons after offline conversion and common NPZ save/load
give the following maximum absolute normalized-flux differences from the frozen
physical stages (fixed representative parameters):

| Library | intrinsic | broadened | full |
| --- | ---: | ---: | ---: |
| Coelho | 3.91e-9 | 9.47e-10 | 9.49e-10 |
| BOSZ | 4.45e-7 | 1.34e-7 | 1.34e-7 |
| TLUSTY | 1.19e-7 | 2.55e-8 | 2.55e-8 |

These diagnostic stage comparisons use immutable numerical source. Separate
tests also pass against the existing saved Coelho/TLUSTY full-spectrum and
gradient oracles. Offline interpolation ordering, dtype quantization and the
endpoint-mean step explain the small numerical differences; no reference output
was regenerated. The safe-interior comparisons do not reinstate legacy edge
clamping or zero-padded convolution behavior.

Earlier pre-standard CPU benchmark on TFRT_CPU_0: 8 regions, 4001 model pixels per region,
1 km/s numerical sampling, 1851 requested pixels per region, 20 warm repetitions.
The atmosphere grid has four axes with two nodes each. First trace/compile and
warmed synchronized execution are measured separately:

| dtype | forward compile [s] | warm forward [ms] | value_and_grad compile [s] | warm value_and_grad [ms] |
| --- | ---: | ---: | ---: | ---: |
| float32 (x64 disabled) | 0.387 | 3.68 | 1.042 | 10.40 |
| float64 | 0.391 | 9.89 | 1.501 | 27.91 |

Compiled temporary-buffer estimates are 2.05/33.95 MB for float32 forward/AD,
and 3.82/67.90 MB for float64 forward/AD; these are compiler estimates, not peak
process RSS. Model numerical leaves are 2.18/4.36 MB respectively. No accidental
region-query by region-payload intermediate, Python pixel loop or per-evaluation
native-to-log regridding is used. These CPU results are a sanity check, not a
GPU performance claim or reason for additional optimization.

All three diagnostic library choices execute successfully. Generated plots and
metrics are outside the repository, under
`/private/tmp/jaxstar-specmodel-{coelho,bosz,tlusty}/`:
`stages.png`, `legacy_comparison.png`, `parameter_response.png`, and
`comparison.json`. The plots show intrinsic line structure, broadening and RV
responses, and nearly overlaid legacy/new predictions with small residuals.
Macroturbulence changes are subtle for the broad TLUSTY lines at its sample
resolving power; no parameter fit is implied by the observed/model overlay.
