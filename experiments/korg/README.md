# Korg experiments

## Internal Julia threading benchmark

`benchmark_threads.py` compares 1, 2 and 4 Julia compute threads through the
current `jaxstar.specfit.korg` synthesis backend. Each thread count runs in a
fresh Python/Julia process, sequentially. It uses all 15 GAOES regions from
[`korg_gaoes.json`](../../scripts/specfit/recipes/korg_gaoes.json), including
its 5 Å margins, 7,500 output pixels per region, 0.0025 Å native vacuum step
and 10 Å line buffer. This does not generate the full atmospheric grid.

Three exact recipe nodes sample different synthesis costs:

| Case | Teff (K) | log g | [Fe/H] | [α/Fe] |
| --- | ---: | ---: | ---: | ---: |
| solar_dwarf | 5950 | 4.0 | 0.0 | 0.0 |
| cool_rich_dwarf | 4600 | 4.5 | +0.5 | +0.25 |
| warm_poor_subgiant | 6700 | 3.5 | −0.5 | −0.25 |

All runs use GES with molecules, MARCS, Bergemann (2025) solar abundances,
Korg's standard alpha group, and the existing empirical-vmic function.
Normalized flux is resampled onto the recipe's air wavelength grids, with no
rotational, macroturbulent or instrumental convolution.

From the repository root, after the optional Korg setup in the
[preparation README](../../scripts/specfit/README.md):

```bash
python experiments/korg/benchmark_threads.py
# Continue after an interruption; completed thread-count runs are retained:
python experiments/korg/benchmark_threads.py --resume
```

The default Julia project is `experiments/korg/.juliacall`; set
`PYTHON_JULIAPKG_PROJECT` to use another project. Each worker sets
`PYTHON_JULIACALL_THREADS` before importing JuliaCall and verifies the actual
Julia thread count. It also sets `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, following
the [JuliaCall multithreading guidance](https://juliapy.github.io/PythonCall.jl/stable/juliacall/#Multi-threading).
BLAS and Julia GC use one thread for every run, to isolate compute-thread
scaling. JuliaCall's signal handling affects Python's Ctrl-C handling; the
parent benchmark terminates its worker if interrupted.

Julia startup, atmosphere preflight and line-list loading are recorded
separately. For each point, one first call is followed by three warm calls;
the reported speed is their median. Each call includes atmosphere interpolation,
synthesis, array transfer, normalization and resampling. Disk I/O and vmic
evaluation are excluded. There is one forced GC after setup, with ordinary
automatic GC during measurements. Peak RSS is the cumulative process maximum,
including setup, compilation and all preceding calls, not memory allocated
by one spectrum.

Ignored `results/benchmark_threads/` stores each worker's `timing.json`, log and
normalized spectra, plus `comparison.json` with speedups and flux differences
relative to one thread. The comparison requires matching source hashes,
physical settings and runtime provenance apart from thread count. It checks
identical output wavelength arrays. Small rounding differences are expected
to be possible with multiple threads, as described in the
[Korg reproducibility FAQ](https://ajwheeler.github.io/Korg.jl/stable/FAQ/#Are-Korg-spectra-bitwise-reproducible?).

### Measured results (2026-10-25)

Local macOS 15.7.4 arm64 machine, 8 logical CPUs; Python 3.13.16,
Julia 1.13.2, Korg 1.3.1 and juliacall 0.9.37. All times below are seconds
per spectrum covering **all 15 regions**, with the median of three warm calls.

| Case | 1 thread | 2 threads | Speedup | 4 threads | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| solar_dwarf | 28.238 | 15.818 | 1.79× | 10.374 | 2.72× |
| cool_rich_dwarf | 53.618 | 29.679 | 1.81× | 20.035 | 2.68× |
| warm_poor_subgiant | 25.147 | 14.437 | 1.74× | 9.507 | 2.64× |

Four threads were 1.48–1.52× faster than two. Their median CPU-time/wall-time
ratios were 2.76–2.99, confirming concurrent CPU use. The cumulative process
peak RSS was **3.02 / 3.50 / 3.38 GiB** for 1 / 2 / 4 threads. These peaks
include automatic GC and initialization; the slightly lower four-thread peak
does not establish that more threads generally need less memory.

All nine saved spectra were finite and had shape `(15, 7500)`, with identical
wavelength arrays across thread counts. Repeated calls within each worker
were bitwise identical in this run. Relative to one thread, the maximum
absolute difference across all cases/configurations was **4.03e-13**, and
the maximum RMS difference was **7.25e-14** in normalized flux. There is no
material spectral change in this test.

Setup remained substantial in each fresh process: Julia/Korg import took
6.8–16.0 s, three-point MARCS preflight including first-use compilation took
146–161 s, and loading the already-downloaded GES list took 35.6–43.7 s.
The first solar-dwarf synthesis took 43.1 / 29.2 / 24.0 s for 1 / 2 / 4 threads.
These costs are separate from the warm medians above.

Four internal threads are a useful starting setting on this machine; no
outer node-parallel scheduler or synthesis-code change is needed. For example,
after configuring the optional Julia environment, the existing CLI can be run
with these thread controls (the benchmark itself did **not** run this full grid):

```bash
PYTHON_JULIACALL_THREADS=4 PYTHON_JULIACALL_HANDLE_SIGNALS=yes \
JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/korg_gaoes.json \
  --output-dir /path/to/prepared/korg_gaoes
```

Set thread variables before JuliaCall's first import; an already-running
notebook needs a kernel restart. Multiplying the three four-thread medians
by 1,575 nodes gives **4.2–8.8 hours of synthesis**. This is a rough
extrapolation, not a measured full-grid duration or a bound: other atmospheric
points can cost more or less, and setup, full-grid preflight and disk I/O add
time. Only these three nodes and these thread counts were benchmarked.

## GALAH / GES timing with the agreed synthesis settings

`benchmark_linelist.py` measures one atmosphere at Teff = 6000 K, log g = 4.25,
[Fe/H] = 0 and [α/Fe] = 0 through juliacall. It uses the current
`jaxstar.specfit.numpyro_model.empirical_vmic(teff, logg, feh)` directly
(1.1776875 km/s here), Bergemann (2025) solar abundances, and Korg's standard
alpha group **including Ar**. These settings also underpin the production
Korg backend; the earlier fixed-vmic experiments below are historical runs.

With jaxstar and juliacall installed as described below, run from the repository
root:

```bash
python experiments/korg/benchmark_linelist.py
```

GES data are downloaded into Julia's artifact cache on first use. Molecular
lines are included for both lists. The script reads wavelength settings from
the existing `bosz_gaoes.json` recipe and `gaoes_rv.csv`, without using that
recipe's BOSZ atmosphere axes or modifying either file. It compares GALAH and
GES on regions 2 and 3, then measures GES on all 15 GAOES regions. Overlapping
regions are merged for synthesis. Native sampling is 0.0025 Å in vacuum;
normalized flux is resampled onto the recipe's logarithmic air grids.

Timing separates Julia startup, line-list loading, the first call for each
case and three subsequent calls (`--repeats 3`). Each synthesis call includes
abundance/atmosphere calculation, synthesis, Python array transfer,
normalization and resampling. No spectrum or chemical-equilibrium result is
reused. The first GALAH call includes compilation; the first GES call runs in
the same Julia process and is **not** an independent cold-start measurement.
The warm median includes normal automatic garbage collection.

Ignored `results/benchmark_ges/` contains `timing.json`, the three normalized
spectra as NPZs and the Julia environment files. The JSON records settings,
versions, the empirical-vmic function source, file hashes, retained line
counts and individual timings. Its `status` is `complete` only after all
cases pass finite-flux, wavelength-coverage and repeated-output checks.
Existing destinations are refused; use `--output-dir DIR` for another run.

Measured on 2026-10-24 with Korg 1.3.1, Julia 1.13.2, juliacall 0.9.37 and
one Julia thread on this machine:

| Case | Unique vacuum span, including margins | Lines retained with 10 Å buffer | Warm median / spectrum | Three warm calls |
| --- | ---: | ---: | ---: | --- |
| GALAH, regions 2 and 3 | 156.92 Å | 71,877 | **1.300 s** | 1.355, 1.117, 1.300 s |
| GES, regions 2 and 3 | 156.92 Å | 1,300,046 | **5.707 s** | 6.165, 5.707, 5.502 s |
| GES, all 15 GAOES regions | 817.12 Å | 5,948,715 | **27.258 s** | 27.545, 27.051, 27.258 s |

GES was **4.39 times slower** for the identical two-region calculation.
Julia/Korg import took 8.70 s; line-list loading took 1.69 s for GALAH and
48.72 s for GES. GES data had been downloaded before this benchmark, so the
48.72 s measures loading the cached artifact, not downloading it. The first
GALAH synthesis took 188.91 s including compilation; subsequent first calls
for GES's two-region and full-range cases took 5.80 s and 29.27 s in that
already-warmed Julia process. The loaded lists contained 307,000 GALAH and
15,609,115 GES lines after Korg's built-in filtering.

One "spectrum" here means one atmosphere over all wavelengths in the named
case, not one echelle region. All saved arrays were finite and all repeated
outputs agreed to absolute tolerance 1e-12. These are single-atmosphere timing
measurements, not a scientific validation of GES or a timing guarantee for
cooler/metal-richer models. The historical 27-node grid has not been
regenerated with the new empirical-vmic settings.

## Historical mini grid through Python / juliacall

`mini_grid.py` exercises the proposed Korg calculation backend using the existing
recipe/CSV reader, logarithmic wavelength sampling and common NPZ storage.
It calls Korg in one persistent Julia session through **juliacall**, loading the
line list once. The original one-point Coelho comparison is documented below.

The [mini recipe](recipes/korg_gaoes_mini.json) has **27 nodes**:
Teff = 5750, 6000, 6250 K; log g = 4.0, 4.25, 4.5;
[Fe/H] = −0.25, 0, +0.25. [α/Fe] = 0 and microturbulence = 1 km/s are fixed
singleton coordinates. `empirical_vmic` is not used. Regions 2 and 3 come
directly from [`gaoes_rv.csv`](../../scripts/specfit/wavelength_ranges/gaoes_rv.csv),
with 5 Å margins and 7,500 logarithmically spaced pixels per region.

From the repository root, in a Python environment for jaxstar:

```bash
python -m pip install -e .  # if this checkout is not already installed
python -m pip install 'juliacall==0.9.37'  # only for generating this experiment
python experiments/korg/mini_grid.py --dry-run
python experiments/korg/mini_grid.py
```

[JuliaCall / JuliaPkg](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)
locates or installs compatible Julia and resolves the adjacent `juliapkg.json`,
which pins Korg 1.3.1. First use may download data and compile packages. The
default Julia project is the ignored `experiments/korg/.juliacall/`; Julia's
shared package cache remains in its usual location. An explicit
`PYTHON_JULIAPKG_PROJECT` overrides that project. See also
[Korg's Python installation guide](https://ajwheeler.github.io/Korg.jl/stable/install/).
The production backend now offers juliacall through the optional `korg`
extra. Neither Julia nor juliacall is required for ordinary jaxstar use.

The synthesis uses MARCS atmospheres, Bergemann (2025) solar abundances and the
[GALAH DR3 line list](https://ajwheeler.github.io/Korg.jl/stable/API/#Korg.get_GALAH_DR3_linelist)
for these optical regions. The abundance mapping is `format_A_X(feh, feh + alpha)`:
the stored `alpha` coordinate means **[α/Fe]**, not [α/M]. It retains the seven
alpha elements specified in the Coelho experiment below; at fixed alpha = 0
this is the same solar-scaled mixture as Korg's default alpha definition.
The continuum-normalized flux is `solution.flux / solution.cntm`, with no
rotational, macroturbulent or instrumental convolution.

GALAH's line list covers separated bands, not the entire interval between its
shortest and longest wavelengths. These two regions fit inside its
[green band](https://www.galah-survey.org/details/facilities/); the experiment
restricts all bounds including margins to the conservative 5660–5860 Å
interior. Extending to all GAOES regions requires a suitable line list first.

The CSV explicitly declares **air** wavelengths. The saved grid retains those
air coordinates; `Korg.air_to_vacuum` converts each evaluation coordinate for
Korg's vacuum synthesis. Overlapping regions are synthesized together at
0.0025 Å spacing, then interpolated back to each target row without
extrapolation. Hydrogen lines and their MHD treatment are enabled for this
optical experiment. These choices are recorded in `manifest.json`.

Results are published to ignored `results/mini_gaoes/` after validation:

- `grid.npz`, `recipe.json`, `wavelength_ranges.csv`, `manifest.json`:
  the common grid and its preparation/synthesis provenance.
- `validation.json`, `validation.npz`, `validation.png`: exact save/load and
  node interpolation checks, comparison with direct Korg synthesis at all
  eight cell centers, and one extra synthesis at 0.00125 Å to check sampling.
- `julia_Project.toml`, `julia_Manifest.toml`: the resolved Julia environment.

Executed on 2026-10-24 with Julia 1.13.2, Korg 1.3.1 and juliacall 0.9.37.
All 27 node checks and save/load comparisons were exact. Across the eight
cell centers, interpolated minus direct normalized flux had **RMS 0.00124747
and maximum absolute error 0.00785513**. Halving the native step at the first
cell center gave RMS 0.0000375894 and maximum absolute difference 0.000601514.
These quantify this small grid's interpolation and sampling errors; they do
not establish a sufficient spacing for parameter inference. The validation
plot was inspected, and loading the NPZ in a separate Python process did not
import juliacall.

`--output-dir DIR` selects a new destination; existing results are never
overwritten. The grid is readable through the ordinary Python API without
importing Julia:

```python
import jax
jax.config.update("jax_enable_x64", True)
from jaxstar.specfit import load_spectral_grid

spectra = load_spectral_grid("experiments/korg/results/mini_gaoes/grid.npz")
flux = spectra.grid.interpolate(
    dict(teff=6000., logg=4.25, feh=0., alpha=0., vmic=1.)
)["flux"]
```

The experiment also checks `SpecFit.from_grid` with synthetic placeholder
observations. It does not run parameter inference or establish observational
accuracy. The 27-node domain and fixed alpha/vmic are deliberately restricted
in this script; broadening, additional abundance axes and larger-grid design
remain outside this experiment.

The original mini-grid experiment was confined to this directory. `mini_grid.py` loads a
private instance of the existing `prepare_grid.py` module and extends its axis
schema in memory to reuse CSV/recipe validation. At that stage, the preparer
and `src/jaxstar` were unchanged. The production Korg backend and recipes are
now documented in the [preparation README](../../scripts/specfit/README.md).

## One Coelho–Korg spectrum comparison

Run one intrinsic-spectrum comparison at **Teff = 6000 K, log g = 4.0,
[Fe/H] = 0.0, [α/Fe] = 0.0**: the CCF template in
[`coelho.ipynb`](../../examples/specfit/coelho.ipynb). All four coordinates
are exact nodes of its `15239-15449_normed.npz` grid (zero-based indices
`[10, 3, 2, 0]`). `compare.py` checks membership and directly reads that flux
slice and all 5,000 stored wavelengths, 15239.00469634–15448.99937692 Å.
There is no interpolation in Coelho atmospheric parameters.

`compare.py` generates one spectrum, independently of the mini grid above.

Executed on 2026-10-24 with Julia 1.13.2 / Korg 1.3.1 at the default sampling:
**Korg − Coelho has mean −0.006149, RMS 0.011903 and maximum absolute
difference 0.164128** (at 15380.86348 Å), in continuum-normalized flux units
over all 5,000 samples. Many line positions coincide, with visible differences
in depths and weak absorption. The generated plot was visually inspected;
stored Coelho samples, Korg continuum division, interpolation and residual sign
were checked numerically. See the interpretation limits below.

### Install and run

Install [Julia](https://julialang.org/downloads/) **1.10 or newer**. This
experiment pins Korg **1.3.1**, whose package compatibility requires Julia
1.10+, even though the general Korg install page still mentions 1.7.
From the repository root:

```bash
# A separate Julia project; does not add Korg to your global project.
julia --project=experiments/korg -e 'using Pkg; Pkg.instantiate()'

# Use an existing Python environment with NumPy and Matplotlib, or:
python -m venv experiments/korg/.venv
source experiments/korg/.venv/bin/activate
python -m pip install numpy matplotlib

python experiments/korg/compare.py
```

The first Julia installation/import can download package/atmosphere data and
compile code. The generated local `Manifest.toml` records resolved dependencies;
retain it with results if exact reproduction is needed. No Python–Julia bridge
is required: Python launches `synthesize.jl` as a subprocess. Use `--julia PATH`
if Julia is not on `PATH`. See also the [Korg install guide](https://ajwheeler.github.io/Korg.jl/stable/install/).

Results go to ignored `experiments/korg/results/` (rerunning replaces files):

- `comparison.png` / `.pdf`: overlay and **Korg − Coelho**, full range plus a zoom.
- `comparison.csv`: the original Coelho wavelengths, both fluxes and difference.
- `korg_native.csv`: Korg's vacuum wavelengths, flux, continuum and flux/continuum.
- `comparison_metadata.json`, `korg_metadata.toml`, `request.toml`: node, input
  checksum, conventions, settings, versions, actual abundance vector and statistics.

`--check-only` checks the Coelho input without Julia. `--output DIR` changes the
output directory. Korg uses 0.01 Å sampling with one sample of endpoint padding;
linear interpolation onto the Coelho grid refuses extrapolation. To inspect
sampling sensitivity, repeat with `--step 0.005 --output experiments/korg/results/finer`.
No continuum fit, velocity shift, rotational, macroturbulent or instrumental
convolution is performed. Microturbulence is explicitly **1 km/s**, matching
the Coelho synthesis prescription at log g ≥ 3; thermal and pressure line
broadening remain part of the synthesis.

### Wavelength convention: checked, not inferred from the filename

The example NPZ has **no medium metadata**. The published native
[`6000_40_p00p00.ms.fits.gz`](https://wwwmpa.mpa-garching.mpg.de/PUBLICATIONS/DATA/SYNTHSTELLIB/solar_scaled/6000_40_p00p00.ms.fits.gz)
has `CRVAL1=2990.015161`, `CD1_1=0.01999596`, `NAXIS1=750901` and a normalized
first row. Its header says Angstroms but does not explicitly label air/vacuum.
The repository's [`prepare_coelho`](../../src/jaxstar/specfit/libraries.py)
and the legacy jaxspec preparer interpret these native coordinates as **air**
and use the Morton air-to-vacuum formula.

On 2026-10-24, reconstructing this segment from the published FITS with that
conversion reproduced **every NPZ wavelength and selected flux value exactly**.
Using the unconverted FITS coordinates instead gave a wavelength-array maximum
discrepancy of 0.00433026 Å and flux RMS discrepancy of 0.0588742. The small
wavelength-array discrepancy occurs because both alternatives are cropped to
the same numerical bounds; it is not the physical air–vacuum offset (~4.16 Å).
This establishes the preparation convention for this specific stored artifact;
it is not a claim that every Coelho-derived file is in vacuum. Every run checks
its wavelengths against the converted native-header reconstruction and fails if
they differ. It therefore uses the NPZ as **vacuum without a second conversion**.

To independently repeat the flux and wavelength provenance check:

```bash
python -m pip install astropy  # only needed for this optional check
mkdir -p experiments/korg/results
curl -L --fail 'https://wwwmpa.mpa-garching.mpg.de/PUBLICATIONS/DATA/SYNTHSTELLIB/solar_scaled/6000_40_p00p00.ms.fits.gz' \
  -o experiments/korg/results/6000_40_p00p00.ms.fits.gz
python experiments/korg/compare.py --check-only \
  --verify-native-fits experiments/korg/results/6000_40_p00p00.ms.fits.gz
```

Omit `--check-only` to save that check alongside a synthesis. The downloaded
gzip SHA256 was `272eb3f4f77f3c4bbbe63b09b9423c7de35d13cbf58bfe68937ef67530cdae68`.
The NPZ SHA256 was `5d4f862ee54af251749f59582ea5219b60a406be7a7910458325e22c3a239b9b`.
[Korg explicitly synthesizes in vacuum](https://ajwheeler.github.io/Korg.jl/stable/generated/tutorials/Basics/#Putting-it-all-together);
its built-in line-list loader handles the line list's own wavelength convention.

### Abundance mapping and interpretation

The current [abundance guide](https://ajwheeler.github.io/Korg.jl/stable/Abundances/)
and [API reference](https://ajwheeler.github.io/Korg.jl/stable/API/#Korg.format_A_X),
checked 2026-10-24 against installed Korg 1.3.1, specify:

```julia
A_X = Korg.format_A_X(feh, feh + alpha_fe;
    solar_abundances=Korg.bergemann_2025_solar_abundances,
    alpha_elements=[8, 10, 12, 14, 16, 20, 22])
atm = Korg.interpolate_marcs(teff, logg, A_X; perturb_at_grid_values=false)
```

The first argument sets the baseline [X/H] for metals, including Fe; the second
sets **[α/H] = [Fe/H] + [α/Fe]**, not [α/Fe]. They are per-element abundance
defaults, not a constraint on total metal mass fraction. The code checks Fe
and all seven Coelho alpha elements (O, Ne, Mg, Si, S, Ca, Ti). Korg normally
also includes Ar in its alpha group; excluding it matches Coelho's definition
and has no effect at this solar-scaled point. Passing `A_X` to the atmosphere
interpolator lets Korg translate to MARCS's solar abundance reference.

The [APOGEE DR17 line list](https://ajwheeler.github.io/Korg.jl/stable/API/#Korg.get_APOGEE_DR17_linelist)
covers this H-band interval. `synthesize` returns unnormalized flux; we divide
by its returned continuum (`solution.flux ./ solution.cntm`). Hydrogen lines
are enabled, with the documented infrared setting `use_MHD_for_hydrogen_lines=false`.

Equal parameter labels do not imply identical physics: [Coelho (2005)](https://arxiv.org/pdf/astro-ph/0505511)
uses ATLAS9/PFANT and Grevesse–Sauval (1998) solar abundances; this Korg run
uses MARCS and Bergemann (2025) solar abundances. Line lists, atmospheric
structures and line/continuum treatments also differ. In particular, Coelho's
paper states that Brackett lines are absent, whereas Korg includes them.
Coelho also already underwent wavelength interpolation when the example NPZ
was prepared. Differences are exploratory library comparisons at identical
labels, not an isolated test of radiative-transfer engines or absolute mixtures.
