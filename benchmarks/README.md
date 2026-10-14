# Spectral CPU/GPU benchmark comparison

Run from the repository root using a Python environment with JAX/jaxlib for the
target hardware and the repository's runtime dependencies. No spectral data,
frozen sibling repository, notebook execution or fitting/inference is required.

The standard case is **float32, B=1, ten regions, four two-node atmosphere axes**.
SpecModel uses 4000 log-uniform model pixels per region (1 km/s) and 2000 output
pixels per region. Even model counts are valid; the velocity kernel remains odd.
The interpolation-only benchmark returns a `(10, 4000)` payload from one scalar
atmosphere query. Each script's defaults are this standard case; explicit options
below make the intended workload clear.

```bash
PYTHONPATH=src python benchmarks/benchmark_specmodel.py --regions 10 --model-pixels 4000 --output-pixels 2000 --dtype float32
PYTHONPATH=src python benchmarks/benchmark_grid_payload.py --axes 4 --nodes 2 --regions 10 --pixels 4000 --batch-size 1 --dtype float32
```

Both scripts use the first device of the active JAX backend. Prefix the same
commands with `JAX_PLATFORMS=cpu` for CPU, or `JAX_PLATFORMS=cuda` on a CUDA JAX
machine to require GPU rather than permit CPU fallback. Inspect the reported
backend/device before comparing. There is no device-specific numerical branch.

## Workloads and timings

SpecModel runs atmosphere interpolation, combined rigid rotation / radial-
tangential macroturbulence / Gaussian IP, relativistic RV and requested-wavelength
sampling. Its existing synthetic absorption spectra and representative nonzero
physical parameters are retained and reported. Gradients are with respect to
the full parameter dict through a nonlinear weighted mean of squared physical
flux, not a likelihood. Inputs, including requested wavelengths, are explicitly
placed on the selected device and synchronized before timing.

Grid payload retains its seeded synthetic fields, coordinate interpolation and
mean squared deviation objective. Its value_and_grad differentiates only the
atmosphere coordinates. It performs no broadening, RV or wavelength resampling.
The two grids have the same axis/node count and payload dimensions, but their
flux values and scalar objectives differ; compare the two costs accordingly.
Existing optional multi-axis/multi-batch and mixed-coordinate-dtype cases remain
available, with multiple cases in one final JSON block. They are not required
for the initial GPU comparison.

`compile_seconds` is trace/lowering plus XLA compilation; both parts are also
reported. This excludes device execution. Three synchronized warmup calls
precede twenty synchronized timed calls by default (`--warmup`, `--rounds`).
`warm_median_seconds` is primary; minimum and mean are also reported. All result
PyTree leaves, including every gradient array, finish before each timer stops.
Setup, loading, transfer and compilation are outside warm execution timing.
Optional public compiled-memory estimates are per forward/AD executable, not
runtime peak memory; unavailable analysis is `null` and does not fail the run.

## Copying results

Each invocation ends with exactly one block:

```text
BENCHMARK_RESULT_BEGIN
{ ... valid JSON ... }
BENCHMARK_RESULT_END
```

Copy the complete block from each command. It contains environment/device,
dimensions, parameters, timings, memory estimates and finite-output checks.
`--json-out benchmark-results/name.json` optionally saves the same JSON;
`benchmark-results/` is ignored. No machine-specific results are committed.
The old SpecModel `--pixels` and grid `--output` options remain aliases.

The CPU validation environment uses JAX/jaxlib 0.6.2; the planned GPU environment
uses 0.11. Scripts use public configuration APIs supported by both. Reported
speed ratios therefore include compiler/version differences as well as hardware
differences. The GPU run itself is still to be performed on that machine.

## Controlled legacy full-iid rerun

`benchmark_legacy_jaxspec.py` reruns **only** the frozen jaxspec
`full_iid_value_and_grad` row from
`../jaxspec/benchmarks/benchmark_best_gp_backend.py`. It does not use the M3a
synthetic workload or jaxstar's physical implementation. It needs the sibling
source, JAX/jaxlib, NumPy, SciPy and psutil; **tinygp is not required or imported**.
No GP entry point, package initializer, inference or plotting module is loaded.
Frozen source bytes are executed in an isolated module namespace without
writing bytecode in the sibling. No numerical compatibility shim was needed
on Mac JAX 0.6.2; the A100 JAX 0.11.2 execution remains to be checked there.
Unsupported frozen-source APIs should fail rather than be silently rewritten.

The default reproduces the documented Linux/A100 inputs:

- CSV: `../jaxspec/data/IRDA00042313_H.csv`; orders 8–17 in that order.
- Grid directory: `/home/masuda/specgrid_irdh_coelho`, recovered from the frozen
  benchmark README. No Mac-directory or synthetic fallback is used.
- CSV `lam` is multiplied by 10 (nm to Angstrom); the frozen CSV helper preserves
  all 2048 rows per order. `all_mask` and nonfinite/error<=0 masks are combined;
  masked flux/error are replaced with 1, as historically.
- The frozen matcher requires `grid_min + 3 < obs_min` and
  `grid_max - 3 > obs_max`, selecting the narrowest covering NPZ (historical
  sorting/ties and filename range parsing are retained).
- Float64/x64; `vmax=50`, `vsini=5`, `zeta=2` km/s, `wavres=70000`,
  `u1=0.5`, `u2=0.2`, RV=0, norm=1, slope=0, dilution=0.
- Scalar atmosphere coordinates are the intersection midpoints of the selected
  grids' bounds. Values and per-file bounds are reported rather than guessed.
- The frozen grid loader/interpolator and `SpecModel.fluxmodel_multiorder` run
  unchanged, including runtime native-to-log interpolation, combined broadening,
  relativistic RV, continuum and dilution. Its working wavelength is
  `np.logspace(log10(wavmin), log10(wavmax), native_count)[1:-1]`, with
  endpoint-trimmed native count, median `dlnlambda`, shared kernel length chosen
  from the first order, `Nt=500`, and historical `same` convolution/interp edges.

The objective is exactly the historical non-GP row:

```python
0.5 * sum(where(~mask_obs, ((flux_obs - flux_model) / error_obs)**2, 0))
```

No Gaussian normalization term or jitter is added. All **13 dict leaves** are
differentiated: scalar `teff`, `logg`, `feh`, `alpha`, `vsini`, `zeta`, `u1`,
`u2`, `dilution`, and per-order arrays `norm`, `slope`, `wavres`, `rv`.
The ten-order case therefore has 49 scalar parameter entries. `mask_fit` is
not used. The benchmark reports model/grid dimensions, matched paths, observed
and valid counts, exact parameter values/shapes, objective, gradient finiteness,
source/CSV SHA256, backend/device and compatibility shims (currently none).

The default observation counts are 2048 per order (20480 total), with unmasked
counts `[1546,1754,1408,1520,1302,1629,1553,1643,1622,1589]` (15566 total).
The saved ten-order artifact's filename inventory reproduces the CSV matching
below. These are the expected basenames under the Linux default directory;
the actual Linux files/contents are unavailable on this Mac and must be
confirmed by the A100 report's `grid_files` and `grid_shapes`.

| Order | Matched basename | Unmasked pixels |
| --- | --- | --- |
| 8 | `15239-15449_normed.npz` | 1546 |
| 9 | `15399-15611_normed.npz` | 1754 |
| 10 | `15563-15777_normed.npz` | 1408 |
| 11 | `15731-15947_normed.npz` | 1520 |
| 12 | `15902-16120_normed.npz` | 1302 |
| 13 | `16077-16297_normed.npz` | 1629 |
| 14 | `16256-16478_normed.npz` | 1553 |
| 15 | `16439-16663_normed.npz` | 1643 |
| 16 | `16626-16853_normed.npz` | 1622 |
| 17 | `16817-17046_normed.npz` | 1589 |

The Linux/A100 historical iid objective is **not available** in the frozen local
artifacts. The saved Mac value `94153.52017616687` belongs to different grids
(`15×3×4×2×5000` per order, versus five gravity nodes in the Linux record), so
it is not used as an A100 oracle. Before timing, this wrapper compares the full
AD objective to an independent host reduction of the frozen forward spectrum
and checks every gradient leaf/shape for finiteness; it rechecks the timed result.

Timing uses the same synchronized utility as M3a. The full objective and every
gradient leaf are ready before a timer stops. Loading, validation and explicit
parameter/observation transfers are outside warm timing. The legacy model/grid
NumPy arrays remain captured by frozen `static self` JIT, preserving that path.
In-memory JIT caches are cleared after preflight; lowering and XLA compilation,
one first execution (including materialization of legacy constants), **3 further
warmup calls** and **30 timed calls** are separated. Reported compile time can
still be affected by a configured persistent disk cache; relevant environment
variables are included in JSON. First execution is an opt-in addition to the
shared utility; existing M3a benchmark timings are unchanged.

On the A100 with JAX/jaxlib 0.11.2, from the jaxstar repository root:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_legacy_jaxspec.py \
  --require-gpu \
  --json-out benchmark-results/legacy-jaxspec-a100-f64-jax011.json
```

For CPU on the same machine, change `cuda` to `cpu` and `--require-gpu` to
`--require-cpu`, using a different result filename. The historical Linux grids
must be available; `--grid-dir` can specify their actual location. To select a
different frozen checkout use `--jaxspec-root`; `--data-csv` and `--orders` may
also be overridden explicitly. An alternate grid or order set is a different
workload and should not be interpreted as the historical A100 rerun.

This Mac has only the immutable Coelho order-8 sample grid, not the historical
Linux ten-order directory. On JAX/jaxlib 0.6.2, the order-8 sanity run returned
objective `6027.030123935277`, finite model/all 13 gradient leaves, and an
independent host-objective difference of `1.09e-11`. Grid shape was
`(15,5,4,2,5000)`, working wavelengths `(1,4998)`, output `(1,2048)` and kernel
`(1,123)`; the grid-midpoint atmosphere was `(5250,3,-0.25,0.2)`.
No tinygp import or compatibility shim was used. Its import/CPU sanity command is:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python benchmarks/benchmark_legacy_jaxspec.py \
  --require-cpu --grid-dir ../jaxspec/characterization/sample_grid_coelho \
  --orders 8 --warmup 1 --repeat 2 \
  --json-out benchmark-results/legacy-jaxspec-mac-order8-sanity.json
```

Copy the **complete** `BENCHMARK_RESULT_BEGIN ... BENCHMARK_RESULT_END` block
from the A100 run. Compare its warm median to the historical **1.685 ms**
(A100/JAX 0.4.33), then consider the new M3a **~2.1–2.2 ms**. The latter uses
different spectra, sampling and objective/gradient leaves; this first experiment
tests the old implementation under the newer JAX stack, not old/new workload
equivalence. Mac sanity timings are not evidence for the A100 comparison.

## SpecModel model/library closure specialization

`benchmark_specmodel_specialization.py` measures execution strategy only.
It imports the existing `benchmark_specmodel.make_inputs`, builds/places one
model/parameter/wavelength tuple once, and reuses those exact arrays for:

- `dynamic_model`: `jit(lambda model, params, wavelength: model(params, wavelength))`.
- `specialized_model`: `jit(lambda params, wavelength: model(params, wavelength))`,
  capturing the same fixed model/library in the closure.

Requested wavelengths and parameters remain **dynamic in both modes**. The
primary experiment captures only the model/library; no optional wavelength
capture case, `static_argnums`, public compile/specialize API, physics change or
SB-N implementation is included. The compiler's internal constant representation
is not prescribed by closure capture. The relevant constant/cache environment
flags are reported along with the actual compiled memory metadata.

The standard workload remains B=1, ten regions, 4000 model pixels and 2000
output pixels per region, 1 km/s numerical model spacing, four two-node
atmosphere axes, vsini=6.3, vmacro=3.1 km/s, u1=0.5/u2=0.2, region RVs from
-15 to +15 km/s and resolving powers from 68000 to 72000. The scalar objective
is the existing `mean(flux**2 * linspace(0.5,1.5,n_output))`, differentiating
only the same complete parameter PyTree, not model/library or wavelength.
This is not the legacy iid likelihood workload.

Both modes use the unchanged shared synchronized timer: separate lowering,
XLA compile and total compile, then 3 warmup and 20 timed calls for forward and
value_and_grad. The entire result PyTree is ready before each timer stops.
Model construction, preparation, transfer and compilation are excluded from
warm timing. Public compiled-memory estimates (`temporary_bytes`,
`argument_bytes`, `output_bytes`) are reported for each stage/mode, or null
when unsupported. No private-XLA inspection is used.

After timing, forward flux, scalar objective and every gradient leaf must
agree: float32 `rtol=5e-5, atol=2e-6`; float64 `rtol=1e-10, atol=1e-12`.
Shapes/dtypes/PyTree structure and finiteness are also checked. JSON includes
maximum absolute differences, including each gradient leaf. A failed comparison
raises instead of reporting a successful benchmark.

From the jaxstar root on the A100 with JAX/jaxlib 0.11.2:

```bash
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_specmodel_specialization.py \
  --require-gpu --dtype float32 \
  --json-out benchmark-results/specmodel-specialization-a100-f32-jax011.json
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_specmodel_specialization.py \
  --require-gpu --dtype float64 \
  --json-out benchmark-results/specmodel-specialization-a100-f64-jax011.json
```

For Mac sanity or same-machine CPU comparison use `JAX_PLATFORMS=cpu` and
`--require-cpu` with separate JSON filenames. Defaults already specify the
standard workload, 3 warmup and 20 rounds; size/round options remain available.
Float64 enables x64 explicitly; float32 disables it, as in the existing benchmark.

Each run emits one `BENCHMARK_RESULT_BEGIN ... BENCHMARK_RESULT_END` block.
Paste **both complete blocks**, including `environment`, `problem`, `parameters`,
`timing`, `modes`, `comparison` and `validation`. Warm speedup is
`dynamic_median / specialized_median` (>1 means faster); warm improvement is
`100 * (1 - specialized_median / dynamic_median)`. Compile ratio is
`specialized_compile / dynamic_compile` (>1 means more compile cost).

Interpret the A100 result before any architectural decision: below roughly 10%
improvement is likely too small to justify complexity, 10–20% may be useful,
and above 20% warrants considering a future fitting specialization path.
Approaching 1.5–1.7 ms in float64 would support specialization explaining much
of the reported gap (controlled legacy ~1.522 ms versus dynamic M3a ~2.1–2.2 ms),
but the legacy spectra/objective/gradient leaves differ. Mac timings are only
sanity checks; this experiment adds no specialization API or Milestone 3b work.

Mac CPU sanity (JAX/jaxlib 0.6.2) passed for the full standard workload in both
precisions with 3 warmup/20 rounds. Maximum absolute dynamic/closure differences:

| Dtype | Forward | Objective | Parameter gradients |
| --- | --- | --- | --- |
| float32 | `2.38e-7` | `1.19e-7` | `1.64e-10` |
| float64 | `4.44e-16` | `0` | `5.94e-19` |

All ten gradient leaves were finite and matched. Additional checks confirmed
that changing requested wavelengths and parameters affects the compiled closure,
that both modes reuse the same device-resident parameter/wavelength arrays,
and that the dynamic objective reproduces the existing benchmark. These are
correctness checks, not an interpretation of A100 performance.

## Generic SB-N scaling

`benchmark_specmodel_sbn.py` measures N=1,2,3 by default using the same standard ten-region,
4000 model / 2000 output pixel workload and deterministic weighted squared-flux
objective. N=1 uses the existing M3a input tree exactly (no added weight/dilution
AD leaves). N>1 varies every component's atmosphere, broadening and nonzero RV
and uses unequal relative weights `1/(i+1)` as dynamic AD inputs. Resolving power
is shared by all stars and region-dependent as before; dilution is zero.
Larger counts are optional through `--components`; the synthetic atmosphere recipe
supports N<=8, while SpecModel has no such limit. N=8 is not a routine validation target.

The model/library, parameters and evaluation wavelengths are device-resident
dynamic inputs. The existing synchronized timer reports separate forward and
value_and_grad lowering/XLA/total compile time, then three warmups and twenty
timed calls, each waiting for every output leaf. JSON records per-N dtype,
median/min/mean, optional compiled temporary/argument/output bytes, objective
and gradient finiteness. Compiled memory estimates are not runtime peak RSS.
Each invocation emits exactly one `BENCHMARK_RESULT_BEGIN ... BENCHMARK_RESULT_END`
block containing all N cases. Results/cache files stay ignored. No closure
specialization, physical change, explicit component vmap or GPU optimization
is introduced; interpret GPU scaling before choosing any execution restructuring.

From the repository root, exact CPU commands (select the intended Python environment):

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python benchmarks/benchmark_specmodel_sbn.py \
  --require-cpu --components 1 2 3 --regions 10 --model-pixels 4000 \
  --output-pixels 2000 --warmup 3 --rounds 20 --dtype float32 \
  --json-out benchmark-results/specmodel-sbn-cpu-f32.json
JAX_PLATFORMS=cpu PYTHONPATH=src python benchmarks/benchmark_specmodel_sbn.py \
  --require-cpu --components 1 2 3 --regions 10 --model-pixels 4000 \
  --output-pixels 2000 --warmup 3 --rounds 20 --dtype float64 \
  --json-out benchmark-results/specmodel-sbn-cpu-f64.json
```

On the A100 JAX/jaxlib 0.11.2 machine:

```bash
unset LD_LIBRARY_PATH
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_specmodel_sbn.py \
  --require-gpu --components 1 2 3 --regions 10 --model-pixels 4000 \
  --output-pixels 2000 --warmup 3 --rounds 20 --dtype float32 \
  --json-out benchmark-results/specmodel-sbn-a100-f32-jax011.json
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_specmodel_sbn.py \
  --require-gpu --components 1 2 3 --regions 10 --model-pixels 4000 \
  --output-pixels 2000 --warmup 3 --rounds 20 --dtype float64 \
  --json-out benchmark-results/specmodel-sbn-a100-f64-jax011.json
```

The library-path unset retains the workaround that resolved cuSPARSE discovery
on this machine. For CPU on the same GPU machine, use the CPU commands above.
Float64 explicitly enables x64; float32 disables it. No GPU is available on this
Mac, so its timings are sanity measurements only. Paste both complete result
blocks from the A100 for later scaling interpretation.

## MIST packaged bolometric-correction validation

`validate_mist_bolometric_corrections.py` is the first feasibility check for
building a photometric evolutionary-track grid from **packaged EEP tracks plus
packaged bolometric corrections**. It recomputes magnitudes at the stellar
parameters already present in official CMD files. This isolates BC conversion
from track interpolation, age/EEP inversion, priors and inference.

Use the MIST v1.2, `v/vcrit=0.4` data from the
[MIST packaged-grid page](https://mist.science/model_grids.html):

- Synthetic photometry: `MIST_v1.2_vvcrit0.4_UBVRIplus.txz` (existing jaxstar cache).
- Theoretical isochrones: `MIST_v1.2_vvcrit0.4_basic_isos.txz` (the matching `.iso` files).
- Bolometric corrections: `BC_tables/v1/UBVRIplus.txz` (read directly as an archive).

The script requires NumPy, SciPy and Matplotlib. It performs no downloads or
archive extraction and does not import or alter `MistFit`. Run from the repo root:

```bash
python benchmarks/validate_mist_bolometric_corrections.py \
  --bc /path/to/UBVRIplus.txz \
  --cmd /path/to/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd \
  --theory /path/to/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso \
  --output-dir benchmark-results/mist-bc-validation
```

Repeat `--cmd` and `--theory` for multiple metallicities, listing both sets in
the same order. Defaults select initial masses 0.7–1.5 solar masses,
`log10(age/yr)=8.0–10.14`, and EEP 202–550. The conversion uses `Av=0`,
`Rv=3.1`, and `M_band = 4.74 - 2.5*log_L - BC_band`.

**Composition convention matters.** In the public MIST color code, the BC
table is queried at `log10(Zsurf/0.0134)` (capped at initial `[Fe/H]+0.1`).
The CMD's displayed `[Fe/H]` additionally includes the surface hydrogen
correction. To recover the BC coordinate from a published CMD, use
`CMD_[Fe/H] + log10(surface_h1/0.7381)`. The script gets `surface_h1` from the
theory file at the same age and EEP and verifies agreement in age, initial/current
mass, temperature, gravity and luminosity. It never estimates a metallicity
offset by fitting the photometry. For a future actual track loader, obtain
the BC coordinate directly from the track's surface composition, keeping it
distinct from both initial metallicity and the spectroscopic surface `[Fe/H]`.

The compared methods are linear interpolation in `(logTeff, logg, BC_metallicity)`
and the interior recipe in the public source: four-point Steffen interpolation
in `logTeff`, linear interpolation in `logg`, then four-point Steffen interpolation
in metallicity. BC evaluation follows this order because the limited cubic
operations are nonlinear. The latter is an experimental NumPy implementation
restricted to interior stencils, not a complete production treatment of edges,
nonzero extinction, other abundance mixtures or MIST versions. The recipe is
documented in
[BC_tables.f90](https://github.com/aarondotter/iso/blob/ed6ef0588664de62df39564fb02e3e0e32d5b2eb/src/BC_tables.f90),
[iso_eep_color.f90](https://github.com/aarondotter/iso/blob/ed6ef0588664de62df39564fb02e3e0e32d5b2eb/src/iso_eep_color.f90),
and [MESA's interpolation routines](https://github.com/MESAHub/mesa/blob/main/interp_1d/public/interp_1d_lib_sg.f90).

On 2026-10-04, initial metallicities `[-0.25, 0.0, +0.25]` supplied
**15,408 selected CMD rows** (5,297 / 5,196 / 4,915). All matched theoretical
stellar parameters agreed exactly, and 420 BC grid-node checks had zero error.
Across all **14 bands** (UBVRI, 2MASS J/H/Ks, Gaia DR2Rev G/BP/RP and Gaia EDR3
G/BP/RP), the largest magnitude error with the source recipe and the correct
BC composition was **4.88e-6 mag**. For comparison, the largest errors were:

| Calculation | Maximum absolute error, all 14 bands |
| --- | ---: |
| Correct composition, public source interpolation recipe | 0.00000488 mag |
| Correct composition, linear interpolation | 0.03546 mag |
| CMD `[Fe/H]` used directly, source recipe | 0.03978 mag |
| Initial `[Fe/H]` used directly, source recipe | 0.02872 mag |

The small remaining discrepancy is compatible with the original single-precision
calculation and six-decimal CMD output; this is not a claim of bitwise equality.
The method was first explored on solar composition, then checked on both adjacent
metallicities without fitting coefficients or zero points to either dataset.
These results establish BC feasibility in the stated domain, not the accuracy
of an interpolated track grid or stellar evolutionary physics.

Results are saved in the ignored `benchmark-results/` directory:
`summary.json` contains per-band residual statistics, input SHA256s, headers,
selection and node checks; `comparison.npz` contains all predictions and
reference values; `residuals.png` compares linear and source-recipe residuals.
The native-track experiment below applies this conversion before constructing
a JAX backend.

## MIST native EEP-track validation

`validate_mist_eep_tracks.py` applies the verified BC conversion to native
evolutionary tracks and audits their age axes. This is an offline NumPy
experiment; it does not yet interpolate across mass or metallicity, implement
a JAX backend, or change `MistFit`.

Inputs are the same packaged `UBVRIplus.txz` BC table and the official
[solar-composition MIST v1.2 EEP archive](https://mist.science/data/tarballs_v1.2/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz).
Extract `00080M.track.eep`, `00100M.track.eep`, and `00120M.track.eep` from that
archive into `benchmark-results/mist-track-inputs/`. They have initial masses
0.8, 1.0, and 1.2 solar masses, initial `[Fe/H]=0`, `[a/Fe]=0`, and `v/vcrit=0.4`.
The script reads EEP numbering and dimensions from the track headers and
validates them against the data rather than assuming a row offset.

```bash
python benchmarks/validate_mist_eep_tracks.py \
  --bc benchmark-results/mist-bc-inputs/UBVRIplus.txz \
  --track benchmark-results/mist-track-inputs/00080M.track.eep \
  --track benchmark-results/mist-track-inputs/00100M.track.eep \
  --track benchmark-results/mist-track-inputs/00120M.track.eep \
  --eep-range 202 550 \
  --output-dir benchmark-results/mist-track-validation
```

The track provides the BC coordinate directly:
`min(log_surf_z - log10(0.0134), initial_feh + 0.1)`. The implementation also
handles the first and last temperature intervals with linear interpolation in
`logTeff`, as in the public `BC_tables.f90`; interior intervals retain the
validated Steffen recipe. Six real-table endpoint/midpoint queries check both
temperature edges across all 14 bands. No extrapolation outside the BC table
is allowed. Metallicity edge stencils and the maximum-gravity boundary remain
unsupported; coverage masks distinguish these implementation limits from
points outside the table. The selected EEP range must have complete photometry.

On 2026-10-04, all three tracks had **1,710 nodes**, spanning EEP 1–1710.
Both age and log-age were strictly increasing in float64, with no missing EEPs
or nonpositive intervals. All **5,130 nodes × 14 bands** had finite absolute
magnitudes at `Av=0`, `Rv=3.1`. The temperature-edge recipe was needed for
4 / 56 / 29 nodes respectively; its independent table checks had zero error.
No composition caps or unsupported BC coordinates occurred in these tracks.

**Age precision is consequential.** Quantizing stored nodes to float32 before
differencing erases many small late-evolution time steps:

| Initial mass | Minimum native age step (yr/EEP) | Zero steps, float32 age | Zero steps, float32 log-age |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.527 | 710 / 1,709 | 900 / 1,709 |
| 1.0 | 1.696 | 621 / 1,709 | 846 / 1,709 |
| 1.2 | 1.218 | 572 / 1,709 | 808 / 1,709 |

These are precision losses, not physical reversals in the native tracks.
Within EEP 202–550, every track has 349 nodes with positive age differences
even after float32 quantization. Nevertheless, the largest relative log-age
step errors there are 2.49%, 1.55%, and 1.04%. Preserve age coordinates and
age-prior Jacobians in float64; promoting already-quantized nodes cannot
recover the lost information.

For this audit, `dt_deep_linear_age_yr = diff(age) / diff(EEP)` is the exact
derivative of piecewise linear-age interpolation inside each cell. If instead
interpolating log-age linearly, the derivative is
`ln(10) * age(EEP) * diff(logage) / diff(EEP)`; the saved
`dt_deep_logage_midpoint_yr` evaluates it at cell midpoints. Neither array
establishes derivative continuity at cell boundaries. Any eventual age prior
must use the Jacobian of the interpolation actually implemented.

All native ages are retained for this diagnostic. In particular, the
0.8-solar-mass track extends to about 24.75 Gyr; an application age limit is
a separate model/prior restriction, not a reason to repair the input track.

The ignored output directory contains `summary.json` (input/script SHA256s,
headers, coverage and precision diagnostics), one `.npz` per track (physical
fields, photometry, masks and cell derivatives), `age_diagnostics.png`, and
`photometry.png` (HR diagram and Gaia EDR3 CMD over the selected EEP range).
Initial mass, current mass, initial metallicity, the BC coordinate, and the
CMD-compatible surface-metallicity proxy are stored separately.

This validates the native-node conversion for these three solar-composition
tracks. The small-grid experiment below adds interpolation across tracks and
gradient checks. Wider phase coverage and sampler behavior remain unvalidated.

## MIST track-grid interpolation and gradients

`validate_mist_track_grid.py` builds an experimental
`(initial_mass, initial_feh, EEP)` grid using the existing `RectilinearGrid`.
It reads selected native tracks directly from the three official MIST v1.2
archives, precomputes the 14-band `Av=0` photometry, and enables JAX float64.
No production `MistFit` code or inference behavior is changed.

Use the `v/vcrit=0.4`, `[a/Fe]=0` EEP archives for initial `[Fe/H]` values
`-0.25`, `0.00`, and `+0.25` from the
[packaged grids](https://mist.science/model_grids.html).
The additional archive URLs are:

- [MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_EEPS.txz](https://mist.science/data/tarballs_v1.2/MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_EEPS.txz)
- [MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_EEPS.txz](https://mist.science/data/tarballs_v1.2/MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_EEPS.txz)

```bash
PYTHONPATH=src python benchmarks/validate_mist_track_grid.py \
  --bc benchmark-results/mist-bc-inputs/UBVRIplus.txz \
  --archive benchmark-results/mist-bc-inputs/MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_EEPS.txz \
  --archive benchmark-results/mist-bc-inputs/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz \
  --archive benchmark-results/mist-bc-inputs/MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_EEPS.txz
```

The full saved grid contains **18 masses × 3 metallicities × 349 EEPs = 18,846
nodes**, from 54 native tracks. Mass spacing is 0.05 solar masses from 0.8 to
0.9 and 0.02 from 0.9 to 1.2; EEP 202–550 is retained without decimation.
Incomplete combinations or unsupported BC coordinates cause an error rather
than padding or extrapolation. No application age cut is applied.

The model interpolates `log10(age/yr)`, `logTeff`, `logg`, `logL`, `logR`, current
mass, composition proxies and precomputed magnitudes linearly in the three
coordinates. Age is `10**interpolated_logage`, and `dt/dEEP` comes from that
same JAX function. A separately interpolated `age_gyr` field is retained only
to compare linear-age and log-age accuracy. The NPZ can be reloaded for use:

```python
# Run with PYTHONPATH=src:benchmarks.
import jax
import jax.numpy as jnp
from validate_mist_track_grid import age_years, load_grid

jax.config.update("jax_enable_x64", True)
grid = load_grid("benchmark-results/mist-track-grid/grid.npz")
point = jnp.array([1.03, -0.08, 410.4])  # initial mass, initial [Fe/H], EEP
fields = grid.interpolate(dict(zip(grid.axis_names, point)))
age_yr = age_years(grid, point)
dt_deep = jax.grad(age_years, argnums=1)(grid, point)[2]
```

**Numerical checks, 2026-10-05:** all 18,846 stored nodes were reproduced,
including after saving and reloading. At 128 random cell-interior queries,
values matched an independent SciPy trilinear interpolator, forward/reverse
autodiff agreed, and the largest gradient discrepancy against central finite
differences was `3.77e-11` after scaling by `1 + abs(reference derivative)`.
All native log-age secants and tested cell-interior `dt/dEEP` values were
positive. Twelve outside-domain/nonfinite queries returned NaN. The script
also records representative derivative jumps at knots: the model is continuous
and piecewise differentiable, not continuously differentiable.

**Held-out tracks:** scientific errors are reported separately from numerical
implementation checks. Coarse mass training uses `[0.8, 0.9, 1.0, 1.1, 1.2]`.
Refined training adds `[0.94, 0.98, 1.02, 1.06, 1.14, 1.18]`. Both use the same
seven mass holdouts `[0.85, 0.92, 0.96, 1.04, 1.08, 1.12, 1.16]` at all three
metallicities (7,329 nodes). The 0.85 holdout retains the same 0.1-wide bracket
in both tests. Metallicity validation withholds all solar-composition tracks
and interpolates between `-0.25` and `+0.25` (6,282 nodes). The joint test
withholds both mass and metallicity (2,443 nodes).

| Held-out case | Age relative error, p95 | Teff error, p95 | Gaia EDR3 G error, p95 / max | dt/dEEP relative error, p95 |
| --- | ---: | ---: | ---: | ---: |
| Coarse mass | 2.55% | 16.52 K | 0.0185 / 0.0624 mag | 9.75% |
| Refined mass | 1.94% | 13.74 K | 0.0185 / 0.0476 mag | 8.93% |
| Metallicity | 1.91% | 15.97 K | 0.0349 / 0.1373 mag | 10.43% |
| Joint, refined mass | 1.94% | 27.13 K | 0.0488 / 0.1339 mag | 9.98% |

These are absolute residual quantiles, weighted uniformly over selected nodes,
not an age prior or a population-weighted measure. Across all 14 bands, the
maximum errors in the refined mass and metallicity tests were 0.1297 and
0.1684 mag respectively. Log-age interpolation reduced the age p95 errors
relative to linear-age interpolation (1.94% versus 2.72% for refined mass;
1.91% versus 3.14% for metallicity), without being better at every point.

**Localized failures matter for priors.** Near mass 1.16, solar metallicity,
EEP 435–437, the native rapid-evolution interval shifts between neighboring
tracks. Refined mass and metallicity interpolation overestimated `dt/dEEP`
by factors of **12.14** and **24.06** at their worst points. In the latter
case the reference was `1.573e6 yr/EEP`, versus `3.784e7 yr/EEP` interpolated.
Positive derivatives alone therefore do not validate an age-prior Jacobian.
The largest temperature/age residuals cluster around EEP 420–440; another
localized photometric mismatch occurs near EEP 510–525.

An additional EEP-resolution test trains on even EEPs and withholds odd EEPs
(including primary EEP 353). Its photometry p95 is 0.00170 mag across all bands,
but the maximum is 0.0700 mag near EEP 511. This deliberately coarsened test
is a reason to retain native EEP sampling, not a measurement of error at
unobserved fractional EEPs.

Outputs in `benchmark-results/mist-track-grid/` include the reusable `grid.npz`,
per-case predictions/references and midpoint Jacobians, `gradient_checks.npz`,
`summary.json` with hashes, exact splits, per-band/per-EEP-range errors and
worst coordinates, `heldout_errors.png`, and `feature_diagnostics.png`.

The final grid retains every available node in the stated domain; validation
uses coarser grids to provide independent references. In particular, the
metallicity holdout spans 0.5 dex, twice the saved grid's 0.25-dex spacing.
Do not interpret these residuals as measured errors at the final grid's
midpoints, or assume refinement removes the localized failures. These holdout
results motivate interpolation-sensitivity studies; they do not establish
that the full saved grid is inadequate for inference or supply its physical
interpolation-error bounds. Sampler behavior and broader mass/metallicity
coverage remain outside this experiment. Production inference uses a separately
interpolated centered age derivative, as described below; these autodiff
derivative comparisons do not measure the error of that prior field.

## Extending the track mass domain from 1.2 to 1.5 solar masses

On 2026-10-05, the track experiment was continued independently of the
MistModel refactor. `build_mist_track_extension.py` reads the existing three
EEP archives and BC archive, adding the 15 native masses 1.22 through 1.50
solar masses at 0.02 spacing. There are **99 tracks**, with a saved grid of
**33 masses x 3 metallicities x 349 EEPs = 34,551 nodes**. Initial FeH remains
`[-0.25, 0, +0.25]` and EEP remains 202–550, with no thinning, extrapolation,
or new download. The older grid and posterior artifacts are preserved.

All scalar fields, all 14 photometric bands, and the centered physical-age
derivatives on the original mass domain are **bitwise unchanged**. The
existing archive reader now accepts an optional `mass_range`; its default
remains `(0.8, 1.2)`. The extended grid passes native-node reproduction,
save/reload, independent SciPy interpolation, and value-gradient checks
(maximum scaled gradient difference `5.44e-11`). The inference-specific
checks also pass over the expanded grid, including the separately
interpolated centered derivative and age-dependent EEP support transform.

After building the baseline grid above, construct the extended NPZ with the
same local BC/EEP archives:

```bash
PYTHONPATH=src python benchmarks/build_mist_track_extension.py
```

The default output is `benchmark-results/mist-track-grid-extended/grid.npz`.
`--inputs`, `--baseline-grid`, `--mass-max`, and `--output-dir` override the
local input paths, baseline NPZ, upper mass, and output directory. The original
baseline is retained. This script performs no downloads; automatic acquisition
and full-grid construction remain future work.

## Production track integration (2026-10-06)

The results in this section describe the initial age-window implementation.
The fixed-EEP/optional-soft-age implementation supersedes its prior and EEP
support rules; see [the update below](#fixed-eep-and-optional-soft-age-bounds-2026-10-07).
The current validation script checks the updated implementation. The original
prototype and saved integration results remain historical references.

The existing complete track NPZ can now be loaded with
`MistModel.from_tracks(path)`. `MistFit(model=..., observations=...)` selects
`inference.model_track` for `kind="track"`. The deterministic input coordinates
are `(mass_init, feh_init, eep)` plus distance for apparent observables; the
NumPyro mass site is named `mass`. Native fields and all 14 photometric bands
remain available through per-call keys. Track Teff/radius/age retain the
prototype's log-interpolation-then-exponentiation order. Centered `dtdeep`
retains the jaxstar convention, including copied interior endpoints and
calculation on the full stored EEP axis before inference bounds.

This integration reads the existing prebuilt grid, requires JAX x64, and
rejects missing cells, nonmonotone ages and nonuniform EEP spacing. It adds no
downloads, larger grid, or higher-order interpolation. The original prototype
and all earlier scientific comparison artifacts are retained unchanged.

Run the integration check from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/jaxstar-mpl-bc PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/k_masuda/miniforge3/envs/jnkepler/bin/python \
  benchmarks/validate_mist_track_integration.py \
  --output-dir benchmark-results/mist-track-integration-final
```

Use a new output directory for a repeat. The default input is
`benchmark-results/mist-track-grid-extended/grid.npz` (0.8–1.5 solar masses,
FeH −0.25/0/+0.25, EEP 202–550). Existing MS/turnoff/RGB observations are read
from `benchmark-results/mist-track-inference/`; no new noise is drawn. Per
case, the numerical comparison uses the truth point, 32 saved posterior
coordinates, and 16 age-supported interior coordinates spanning the wider
mass domain: **147 points total**. Source/grid/input SHA-256 values are in
the saved `summary.json`.

Results against the unchanged `benchmarks/mist_track_inference.py` prototype:

| Quantity | Maximum difference across the three cases |
| --- | ---: |
| Predictions, `abs(new-old)/(1+abs(old))` | 5.58e-15 |
| EEP limits from the age window | 0 (exactly equal) |
| Joint log density, both age-prior choices | 8.74e-11 absolute |
| Constrained log-density gradients, same scaled definition | 7.65e-12 |
| Unconstrained potential at physical starts | 7.47e-14 absolute |
| Unconstrained potential gradients, scaled | 2.97e-12 |

The MultiNestRunner currently constructs independent site transforms from one
trace. Therefore `run_multinest` selects `bound_eep_by_age=False` for the default
track model: EEP has fixed proposal bounds and the age-window rejection remains
in the factor. Its Uniform EEP density is canceled just as for NUTS's conditional
proposal, leaving the **same physical joint density** (maximum scaled difference
2.21e-15 on these points). NUTS retains the age-dependent EEP support from the
prototype. The actual optional Python MultiNest runner is tested with only its
native solve bridge replaced, including its cube transform, log density and
deterministic outputs; no native track MultiNest convergence run was performed.

For each case, the production MistFit entry point also completed a short NUTS
run with two sequential chains, 200 warmup and 100 retained samples per chain,
target_accept_prob=0.99 and max_tree_depth=11. All three had zero divergences,
finite saved outputs, positive dtdeep, and samples inside the age/EEP support.
NetCDF/result metadata save and reload preserve every posterior array exactly.
These are API/sampler integration smoke tests, **not new convergence or coverage
assessments**. ArviZ can warn while summarizing constant saved EEP limits.

The offline track unit tests additionally use independent SciPy values and
densities, check finite-difference gradients, malformed grids, both prior
choices, support normalization, JIT/vmap/PyTree behavior, and result plotting.
See [the public API example](../README.md#evolutionary-track-models)
and [the example notebook](../examples/mistfit/fit_track.ipynb).

Final targeted regression command:

```bash
MPLCONFIGDIR=/private/tmp/jaxstar-mpl-bc PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/k_masuda/miniforge3/envs/jnkepler/bin/python -m pytest \
  tests/unittests/test_mist_tracks.py tests/unittests/test_mistmodel.py \
  tests/unittests/test_mistfit_inference.py tests/unittests/test_mistfit_results.py -q
```

Result: **128 passed, 2 skipped** (the opt-in full isochrone-grid cases),
including **29 track tests**. `git diff --check`, Python AST checks and the
recorded integration source/input hashes also pass.

## Fixed EEP and optional soft age bounds (2026-10-07)

The production track model now samples a fixed EEP interval, defaulting to
the stored grid endpoints (202–550 for the existing extended grid). There is
no hard age cut. `age_max=13.8` applies a soft upper bound by default;
`age_max=None` disables it and `age_min` optionally adds a lower bound.
The independent `smbound` penalties act on the public age in Gyr, with `age_bound_width=0.1` Gyr
and `age_bound_depth=30`. A single bound contributes −15 at its location and
approaches −30 outside, so it suppresses but never excludes an age. The
penalty remains active with `nodata=True`.

The centered `dtdeep` field and all deterministic predictions are unchanged.
The factor is now `log(dtdeep)` for linear age, or
`log(dtdeep)-log(age*ln(10))` for log-age. The fixed Uniform EEP density remains
as a constant. No age-window normalization or EEP-width correction is needed,
and NUTS/MultiNest use the same function and options. Age-to-EEP inversion and
its cached age curves were removed. Track `logamin`, `logamax`, and
`bound_eep_by_age` are replaced by the optional Gyr bounds; isochrone semantics
and arithmetic are unchanged.

This is an intentional prior-support change. Historical no-data corners with
the old hard age window do not describe the new default. Available ages can
exceed 13.8 Gyr, and the mass/age marginals still reflect grid coverage.
`examples/mistfit/fit_track.ipynb` uses the 13.8 Gyr soft bound and documents
how to disable it or add a lower bound. Its earlier executed version is preserved at
`benchmark-results/mist-track-soft-age-bounds/fit_track_before_soft_age_bounds.ipynb`;
the notebook is committed without execution outputs. Local saved results under
`examples/mistfit/results/` are excluded from version control.

Validation command (choose a fresh output directory on subsequent runs):

```bash
MPLCONFIGDIR=/private/tmp/jaxstar-mpl-bc PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/k_masuda/miniforge3/envs/jnkepler/bin/python \
  benchmarks/validate_mist_track_integration.py --age-max 13.8 \
  --output-dir benchmark-results/mist-track-soft-age-bounds/validation
```

The script compares predictions and constrained gradients against the
unchanged prototype on 147 common-support points. On that support, the
expected new-minus-old log-density shift is
`log(tmax-tmin)-log(EEP_width)` for linear age and
`log(logamax-logamin)-log(EEP_width)` for log-age: −3.2345262012 and
−5.0913966507 on this grid. The comparison removes this known constant;
it does not assert equivalence outside the former age window. Soft penalties
are checked against SciPy sigmoids. Unconstrained finite-difference checks
use interior points because some synthetic truth coordinates are exactly
on mass/FeH grid knots, where a centered difference crosses two linear cells.
Short NUTS runs use the 13.8 Gyr soft bound; unit tests cover both the default
and explicitly unbounded-age model. These are integration checks, not a new
posterior convergence or interpolation-accuracy assessment.

Historical results in `benchmark-results/mist-track-soft-age-bounds/validation/summary.json`
were recorded before the default changed from `age_max=None` to `13.8`.
Those sampler runs explicitly selected the 13.8 Gyr bound:

| Check | Result |
| --- | ---: |
| Prediction difference, `abs(new-old)/(1+abs(old))` | ≤5.58e-15 |
| Common-support log-density difference after constant shift | ≤8.74e-11 absolute |
| Constrained-gradient difference, same scaled definition | ≤7.65e-12 |
| Soft penalty versus SciPy | ≤3.66e-12 absolute |
| Unconstrained gradient versus interior finite difference, scaled | ≤7.87e-10 |
| Supported native age extremes (soft bounds do not exclude them) | 0.01393–29.93805 Gyr |
| NUTS, three cases × two chains × 100 retained draws | 0/600 divergences |
| Saved/reloaded posterior arrays | Exactly equal |

Each chain used 200 warmup steps, target acceptance 0.99 and maximum tree
depth 11. At the oldest native point the 13.8 Gyr upper bound adds
a finite −30 log penalty; `age_max=None` removes that factor. The targeted
regressions passed **48 track tests and 99 existing model/inference/result
tests**, with **2 opt-in full-grid tests skipped**. Recorded source/grid/input
hashes, Python syntax, notebook validation and `git diff --check` pass.
