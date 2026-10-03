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
