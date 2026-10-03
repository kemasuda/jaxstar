# Milestone 3b: generic deterministic SB-N composition

One `SpecModel` evaluates an arbitrary nonempty fixed tuple of stars using the
same prepared spectral library. The [M3a physical path](specmodel-core.md) is
unchanged: scalar atmosphere interpolation, combined rotation/radial-tangential
macroturbulence/Gaussian IP, relativistic effective RV and requested-wavelength
sampling. Each star passes independently through that same helper, including
the shared instrument response. No new kernel or algebraic rearrangement of
broadening is introduced. There are no `SpecModel2`/`SpecModelN` subclasses.

## Parameters and continuum-light accounting

```python
# Every component has the existing M3a structure:
primary = {
    "atmosphere": {"teff": 5812.5, "logg": 4.2, "feh": -.15, "alpha": .12},
    "broadening": {"vsini": 6.3, "vmacro": 3.1, "u1": .5, "u2": .2},
    "rv": 12.4,
}
# Define secondary and third with the same structure and independent values.
params = {
    "components": (primary, secondary, third),
    "instrument": {"resolving_power": 70000.},
    "flux_weights": (1., .3, .1),
    "dilution": .1,
}
```

Atmosphere names come from the library; coordinates remain scalar per component.
Broadening values, effective RV and the shared instrumental resolving power
remain scalar or `(n_region,)`. Coverage/domain/sampling checks apply to every
component, including a zero-weight component. Libraries and wavelengths remain
dynamic model/input PyTree leaves. Tuple length and dict structure are static
under JIT; parameter values stay dynamic.

For normalized component spectra `f_i`, let `a_i` be relative stellar continuum
flux weights. They are finite/nonnegative, with a positive finite sum in each
region. They need not sum to one; `[1,.3,.1]` and `[10,3,1]` are equivalent.
No softmax, clipping, absolute values or epsilon correction is applied.

```text
q_i = a_i / sum_j(a_j)                   stellar_fractions
w_i = (1-d) q_i                         light_fractions
total = d + sum_i(w_i f_i)
sum_i(q_i) = 1;  d + sum_i(w_i) = 1
```

`dilution = d` is the featureless fraction **of total continuum light**,
`F_dil/F_total`, in `[0,1)`. It preserves the frozen single-star definition.
It is not a third relative stellar weight. The overall scale of stellar weights
is mathematically non-identifiable; its gradient direction is null. Future
probabilistic transformations and identification choices belong outside this
deterministic model.

`flux_weights` accepts `(N,)` shared across regions or `(N,n_region)`. A tuple
or list can mix scalar and `(n_region,)` entries per component. There are no
per-pixel weights. `dilution` is scalar or `(n_region,)`. All other shapes fail.
One star defaults to weight one and dilution zero; any positive supplied weight
gives `q_1=1`. With dilution it gives `d+(1-d)f_1`. Multiple stars require explicit
weights, avoiding an implicit equal-light assumption. Invalid weights/dilution
raise eagerly, or through the existing failure-only host callback under JIT.

Composition interprets spectra as normalized to their own stellar continua.
No normalization or flux rescaling of library payloads is added. Ordinary N=1
M3a calls without composition fields keep their original physical path, including
useful unnormalized products. Calibrated absolute-flux composition is deferred.

## Stages and decomposition

```python
total = model(params, wavelength)          # exactly model.full(...), flux only
intrinsic = model.intrinsic(params, wavelength)
broadened = model.broadened(params, wavelength)
parts = model.decompose(params, wavelength)  # default stage="full"
raw_intrinsic = model.decompose(params, wavelength, stage="intrinsic").components
raw_broadened = model.decompose(params, wavelength, stage="broadened").components
```

All three stage methods compose the corresponding component spectra using the
same weights/dilution. Intrinsic omits broadening/RV; broadened omits RV; full
includes both. Unused physical parameter groups are still unnecessary.
`SpectralDecomposition` is a lightweight NamedTuple and ordinary JAX result
PyTree. `decompose` calls the same component helper, with no duplicated physics.
Its `stage` string is static when passed through JIT.

| Field | Shape | Meaning |
| --- | --- | --- |
| `total` | `(region,pixel)` | Composite physical spectrum |
| `components` | `(N,region,pixel)` | Unweighted component spectra `f_i` |
| `flux_weights` | `(N,region)` | Broadcast user relative weights `a_i` |
| `stellar_fractions` | `(N,region)` | Stellar-only normalized fractions `q_i` |
| `light_fractions` | `(N,region)` | Stellar fractions of total light `w_i` |
| `dilution` | `(region,)` | Featureless fraction of total light `d` |
| `weighted_components` | `(N,region,pixel)` | `w_i[:,:,None] * f_i` |

`parts.total = parts.dilution[:,None] + parts.weighted_components.sum(axis=0)`.
The ordinary stage calls preserve the single-region 1D output convenience;
decomposition always retains region axes, so its total in that case is `(1,pixel)`.

## Frozen parity and validation

Frozen `SpecModel2` uses secondary/primary ratio `r`; the mapping is weights
`[1,r]`, fractions `[1/(1+r),r/(1+r)]`. Frozen `SpecModelN` instead specified
secondary stellar fractions and inferred primary as one minus their sum. Those
valid fractions can also be supplied as relative weights. Its class hierarchy
and fit/continuum ownership are not reproduced. The available frozen SB2 notebook
was inspected; no separate SB3 notebook is present in that checkout.

The existing saved TLUSTY SB2 oracle uses distinct stars, RVs -180.82/+180.82,
and ratio .65. Local contiguous atmosphere cells cover both stars and are
converted once offline to the old log samples. The saved continuum is removed
externally for physical-flux comparison. Component stages use immutable numerical
source; saved spectra/objective/physical and ratio gradients use the immutable
NPZ. Neither reference source nor artifacts are rewritten. Old SB2 does not
expose dilution itself, so its saved physical mixture receives the preserved
single-star rule externally for the nonzero-dilution comparison. Single-star
nonzero dilution also compares directly with frozen source for all three libraries.

Tests cover N=1/2/3 eager/JIT stages and decomposition in both benchmark precision
configurations, exact N=1 behavior, hand-calculated SB2 with/without dilution,
SB3 scale invariance, per-region and mixed-entry weights, fraction/decomposition
identities, zero weights, invalid domains/shapes and compiled failure checks.
Every physical parameter has finite AD leaves; weights/dilution also have finite-
difference comparisons and the overall weight-scale derivative is zero. Existing
M3a physical tests remain, with only their obsolete count restriction adapted.
The model has no fixed small component limit; larger benchmark counts are
optional, with the synthetic atmosphere recipe covering N<=8. Float32 runs disable x64 and float64 enables it,
as in the established benchmark; mixed-dtype promotion of the unchanged Fourier
kernel is not altered.

The executed [tutorial](../../examples/tutorials/specmodel.ipynb) adds compact
SB2/SB3 sections using direct model/decomposition calls. It plots raw stars,
weighted stellar contributions, featureless dilution and total; SB3 additionally
shows region-dependent weights/RVs. These are parameter illustrations, not fits
or claims that the IRD sample is binary/triple.

The [scaling benchmark](../../benchmarks/benchmark_specmodel_sbn.py) measures
N=1/2/3 by default on the standard ten-region 4000→2000 sample workload. N=1 reuses the
original inputs/objective/AD parameter leaves exactly. N>1 has distinct atmosphere,
broadening/RV parameters and unequal dynamic weights, with zero dilution. It
reports separate forward/AD trace+compile, synchronized warm median/min/mean,
optional compiled temporary bytes, objective and finiteness in one delimited
JSON block. No vectorization, closure specialization, custom AD or GPU-specific
optimization is introduced. Commands are in [benchmark instructions](../../benchmarks/README.md).

## Deferred work

No SpecFit, observation state, continuum, likelihood, CCF, GP, NumPyro, inference,
heterogeneous component libraries, pixel-dependent weights, Korg, differential
rotation or public compile/specialization API is added. The full later fitting
parity inventory remains in [the migration note](specfit-migration.md), including
fitted physical/continuum/data-space model reconstruction. Actual full raw-library
smoke testing remains a later validation TODO on a machine with those libraries.

## Recorded M3b checks (2026-10-03)

Full repository suite with `JAXSPEC_REFERENCE_ROOT=../jaxspec`: **402 passed,
1 skipped, 2 xfailed in 265.71 s**. The skip is the opt-in full MIST check and
the expected failures are unchanged. M3b adds 60 cases without dropping M3a
physical cases. The tutorial executes all 19 code cells without errors and
contains ten inline figures, including the new SB2 and SB3 plots.

Maximum absolute normalized-flux differences for the saved TLUSTY SB2 oracle
are `2.02e-8` at zero dilution and `1.62e-8` at d=.2. Individual full spectra
versus frozen source differ by at most `2.22e-8`/`3.28e-8`. Saved objective and
all mapped physical/ratio gradients pass the existing tolerances (objective
rtol=2e-5, atol=2e-6; scaled gradients rtol/atol=3e-4). No reference regeneration
occurs. All 3668 sibling files retain their starting sizes/modification times;
the immutable reference SHA256 checks also pass.

CPU sanity uses Python 3.12.11, JAX/jaxlib 0.6.2, TFRT_CPU_0; ten regions,
4000→2000 samples, 3 warmups and 20 synchronized rounds. Following review,
the routine benchmark targets N=1,2,3; N=8 is optional, not a default target.

| dtype | N | forward compile [s] | AD compile [s] | warm forward [ms] | warm AD [ms] | forward temp [MB] | AD temp [MB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 1 | 0.449 | 1.099 | 5.75 | 14.34 | 2.58 | 42.44 |
| float32 | 2 | 0.533 | 1.832 | 11.37 | 26.07 | 5.22 | 76.78 |
| float32 | 3 | 0.720 | 2.648 | 15.36 | 40.09 | 7.79 | 113.14 |
| float64 | 1 | 0.392 | 1.052 | 17.87 | 47.36 | 4.80 | 84.87 |
| float64 | 2 | 0.525 | 1.767 | 34.15 | 87.54 | 9.63 | 153.55 |
| float64 | 3 | 0.738 | 2.607 | 50.63 | 129.23 | 14.47 | 226.28 |

Warm entries are medians; JSON also reports minimum/mean and lowering/XLA splits.
Every forward/objective and all 10/20/29 parameter-gradient leaves are finite.
The approximate component scaling is a CPU sanity observation. Temporary MB
are public compiled-buffer estimates, not measured peak memory. No optimization
or GPU performance conclusion follows; A100/JAX 0.11.2 validation is pending.
