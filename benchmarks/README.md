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
