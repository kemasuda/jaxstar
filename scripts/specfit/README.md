# Reusable spectral grids

`prepare_grid.py` calls the public `prepare_bosz`, `prepare_coelho`,
`prepare_tlusty` or optional `prepare_korg` API and `save_spectral_grid`.
BOSZ/Coelho/TLUSTY read local raw spectra;
Korg synthesizes spectra through Python/JuliaCall and may download its model
data on first use. The script does not fit observations. Run it from a checkout installed with
`python -m pip install -e .`, or use `PYTHONPATH=src` from the repository root
as below.

Keep the script, shared wavelength tables and reusable JSON recipes in Git.
Keep raw libraries and generated grids in a separate data directory shared by
your analyses. A recipe describes the grid, not an individual target star.

```text
scripts/specfit/
  prepare_grid.py
  recipes/bosz_gaoes.json
  recipes/coelho_gaoes.json
  recipes/bosz_ird_h.json
  recipes/coelho_ird_h.json
  recipes/korg_gaoes.json
  recipes/korg_gaoes_mini.json
  recipes/tlusty_ob2025_optical.json
  wavelength_ranges/*.csv
```

## Build and reuse

Inspect the resolved recipe without reading raw spectra or creating outputs:

```bash
PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/bosz_gaoes.json --dry-run
```

Add `--data-dir` to also check every input path and preview missing-node
replacements without reading spectra or writing files:

```bash
PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/bosz_gaoes.json \
  --data-dir /path/to/bosz/rorig --dry-run
```

Build with local BOSZ-2024 original-resolution files in
`rorig/m+0.00/bosz2024_..._rorig_noresam.txt.gz`, etc.:

```bash
PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/bosz_gaoes.json \
  --data-dir /path/to/bosz/rorig \
  --output-dir /path/to/prepared/bosz_gaoes_v1
```

For Coelho, choose `coelho_gaoes.json` and point `--data-dir` at the directory
containing the native `.ms.fits` files. Relative command-line paths are relative
to the working directory; the recipe's CSV path is relative to the recipe.

The recipes specify reusable atmosphere domains; inspect their axes and region
selection before a production build. `bosz_gaoes.json` uses all GAOES table
regions, Teff 5000--8000 K, logg 3--5, `[C/M]=0` and native `vmic=[0, 1, 2]`.
The IRD recipes illustrate smaller domains and explicit region selection.
Every BOSZ/Coelho/TLUSTY build checks all requested input paths before reading any spectra. Coelho
and BOSZ with the default `missing_nodes="error"` require every Cartesian-product
node. BOSZ can explicitly enable the logg replacement policy below; the GAOES
BOSZ recipe enables it.
The BOSZ preparer averages flux at exactly repeated native wavelengths before
resampling; the original-resolution ASCII files can contain repeated printed
wavelengths. Distinct nearby wavelengths are retained.

Each build creates a **new directory** containing:

```text
bosz_gaoes_v1/
  grid.npz                  # all selected regions in one common-format grid
  recipe.json               # runnable recipe snapshot
  wavelength_ranges.csv    # snapshot of the input wavelength table
  manifest.json             # resolved bounds, versions and SHA-256 hashes
```

An existing output directory is refused before preparation. Change the output
name when changing the grid; a failed build does not publish a partial bundle.
The manifest records SHA-256 hashes of the raw inputs, implementation files,
recipe, table and outputs, plus package versions and resolved settings. Keep
the raw library unchanged while building. Raw BOSZ
files can be revised under the same names, so filenames alone are not version
identifiers. The snapshots can be used to repeat the build from another data
directory without the original recipe location.

The manifest's `source_plan` records every replacement's target coordinates,
method, source coordinates, filenames and weights. `sources` hashes the actual
native files used, including neighbours outside the requested axes. Keep the
manifest alongside the NPZ to retain this replacement history. BOSZ/Coelho
NPZs record the actual source filenames; TLUSTY OB2025 additionally retains
the replacement map in its NPZ provenance.

An analysis needs only the NPZ. Create `obs` from the observation rows you want
to fit, as in `examples/specfit/coelho.ipynb`, then pass the NPZ file to the same
`SpecFit.from_grid` entry point:

```python
import jax
from jaxstar.specfit import Observation, SpecFit

jax.config.update("jax_enable_x64", True)
obs = Observation(wav_obs, flux_obs, error_obs, mask=mask_obs, region=region_labels)
sf = SpecFit.from_grid(
    "/path/to/prepared/bosz_gaoes_v1/grid.npz", obs,
    wav_margin=3., vmax=50.,
)
print(sf.regions)
print(sf.specmodel.spectra.grid.axis_names)
```

Each observation row selects the saved region that covers its entire wavelength
interval plus `wav_margin` Angstrom on each side, including masked pixels.
Only the selected regions enter `spectra`, in observation row order. A saved
region may serve several rows. Observation region labels are retained; they do
not need to match saved labels and do not choose the source regions. Without
observation labels, the selected saved labels are used. Zero or multiple
covering candidates fail before flux loading. Each selected region keeps its
full saved wavelength span; no pixel cropping is performed.

The library, wavelength medium and flux kind come from the NPZ metadata.
Saved sampling is retained unless you explicitly request `pixels=...` or
`velocity_step=...` to resample. You can therefore omit the example's legacy
`library`, `pattern`, `pixels` and `flux_kind` arguments. Explicit library,
medium (other than `"unknown"`) or flux kind (other than `"as_stored"`) must
match the saved metadata. No medium conversion or normalization occurs here.
The full flux array is temporarily read into host memory; selection happens
in NumPy before constructing the JAX grid, so unused regions are not carried
into model evaluation.

The example's bundled observations are IRD data: use a matching IRD grid for
those observations, or your own GAOES observations for a GAOES grid. When
switching from Coelho to BOSZ, update any explicitly named atmosphere parameters
or plotting variables: BOSZ uses `mh` instead of `feh` and also has `carbon`
and `vmic` axes. The subsequent CCF, model-building and fitting workflow uses
the same `sf` interface.

For separate loading/model construction, the equivalent selection is:

```python
from jaxstar.specfit import load_spectral_grid, SpecModel

spectra = load_spectral_grid(
    "/path/to/prepared/bosz_gaoes_v1/grid.npz",
    wavelength=obs.wavelength, regions=obs.region, wav_margin=3.,
)
specmodel = SpecModel(spectra, vmax=50.)
sf = SpecFit(obs, specmodel)
```

`load_spectral_grid(path)` without `wavelength` still restores all saved regions.
Legacy per-region NPZ directories remain supported by `from_grid` with their
existing `library`, `pattern` and explicit sampling settings.

## Recipe and wavelength conventions

Required recipe entries are `library`, `axes`, `wavelength_ranges_file`,
`wav_margin`, `sampling` and `normalized`. `description`, `regions` and
`wavelength_medium`, as well as `missing_nodes`, are optional. Unknown keys fail
to catch misspellings.

- `axes`: increasing arrays or inclusive `min`/`max`/`step` ranges, with the
  names accepted by the selected preparer. No model coordinates are inferred
  from observed stars.
- `regions`: unique integer IDs selected **in the listed order**. Omit it to
  retain CSV row order. IDs are legacy row/order indices, not physical echelle
  diffraction orders. Wave ranges are not sorted by wavelength.
- `wav_margin`: added on each side in Angstrom, for RV and broadening coverage.
  The CSV itself contains the original observed intervals, without this margin.
- `sampling`: logarithmic wavelength sampling, with exactly one of
  `{"pixels": 5000}` or `{"velocity_step": 0.5}` (km/s). The example recipes use
  equal pixel counts so all selected regions fit the current rectangular
  spectral-grid API. Their actual velocity steps differ by region and are
  recorded in the manifest. Fixed velocity steps are useful for a single region
  or ranges yielding identical pixel counts; otherwise the preparer rejects
  the combination. Neither option sets instrumental resolving power.
- `normalized`: `true` for fitting continuum-normalized spectra; `false` keeps
  the preparer's unnormalized flux convention.
- `missing_nodes`: `"error"` (default), `"logg"` for BOSZ,
  or `"lowest_logg"` for TLUSTY OB2025,
  as described below.

For example, these entries can be used within `axes`:

```json
{
  "teff": {"min": 4000, "max": 7000, "step": 250},
  "mh": {"min": -0.5, "max": 0.5, "step": 0.25},
  "vmic": [0, 1, 2, 4]
}
```

Both endpoints are included: the Teff example gives 13 nodes, from 4000 through
7000 K. `step` must be positive and divide `max - min` exactly; nonaligned
endpoints fail rather than changing the spacing. `min == max` gives one node.
The saved recipe retains the range notation; `--dry-run` and the manifest's
`resolved_preparation` show the expanded arrays. JSON does not evaluate Python
expressions such as `np.arange(...)`.

CSV columns are `region,wavmin,wavmax,unit,medium`. Bounds are finite increasing
Angstrom values and `unit` is explicitly `angstrom`. `medium` is `air`,
`vacuum`, or `unknown`. A recipe may resolve an unknown medium through an
explicit `wavelength_medium` declaration. Conflicts with a known CSV medium
fail: changing that declaration does not convert wavelength bounds. The
preparer converts *native spectra* into the selected output medium.

## TLUSTY OB2025

Use `recipes/tlusty_ob2025_optical.json` with the extracted
[TLUSTY OB2025 library](https://stsci.app.box.com/v/tlustyOB2025).
`--data-dir` is the parent of `BSTAR_z100v2`, `OSTAR_z100v2`, etc.; no
download or extraction is performed by the script.

```bash
PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/tlusty_ob2025_optical.json \
  --data-dir /path/to/TLUSTYOB_2025 --dry-run

PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/tlusty_ob2025_optical.json \
  --data-dir /path/to/TLUSTYOB_2025 \
  --output-dir /path/to/prepared/tlusty_ob2025_vt2_optical
```

This is an optical example for observations with R approximately 2500--3000,
not a GAOES recipe or an instrumental convolution. Its 3600--7400 Angstrom
table uses **vacuum** wavelengths, with 10 Angstrom margins. Copy/edit the
table to match the intervals and air/vacuum convention of your observations.
Native spectra are vacuum; choosing air converts the native wavelength scale
after continuum normalization. Air ranges, including margins, must lie above
2000 Angstrom. The recipe uses 10 km/s logarithmic sampling; use 3 km/s to
retain approximately the native sampling. Preparation uses point interpolation
without pre-smoothing, so compare coarser sampling against a finer reference
at the precision needed for your observations. The fitting model applies
instrumental broadening separately.

The example uses 26 Teff nodes from 15000 to 55000 K, logg=3.0--4.75 in 0.25
steps, and Z/Z_sun=[0.1, 0.2, 0.5, 1, 2] (1040 requested atmosphere nodes).
These axes match the old jaxspec example. The Teff list
changes from 1000 K steps below 30000 to 2500 K above it. `logZ` must contain
the logarithms of native metallicities, not arbitrary evenly spaced values.
The complete native set is Z/Z_sun=[0.01, 0.02, 0.032, 0.1, 0.2, 0.5, 1, 2].
The OSTAR `z0033` filename label is explicitly mapped to the nominal 0.032
listed for both families in [Hubeny et al. (2025), Section 2](https://doi.org/10.3847/1538-3881/adb1bf);
this mapping is recorded in provenance.

The axes are **teff/logg/logZ**, with synthesis **vt=2 km/s fixed** throughout.
BSTAR is used below 30000 K, OSTAR at and above 30000 K, following the paper's
overlap recommendation. The underlying atmosphere uses vt=2 for BSTAR and
vt=10 for OSTAR. These settings, the family boundary and source release are
stored in both `grid.npz` and `manifest.json`. They remain available through
`load_spectral_grid(path).provenance` and in saved fit-result metadata, without
introducing a vmic fit parameter. Existing common-format files without this
optional provenance field still load.

Every input `.spec.gz`/`.cont.gz` pair used by the plan must exist and be
nonempty before any spectrum is read. Only exact v2 filenames are used;
v5/v10 and `v2old` files do not participate. This preflight checks paths and
sizes, not compressed file contents.

The recipe enables **`missing_nodes="lowest_logg"`** for the high-Teff,
low-logg boundary, following the old grid's treatment of these lower gaps:

1. Use each requested node whenever its pair exists.
2. If both files are absent and the requested logg is below the lowest
   available logg at the same Teff/Z/family/vt, copy that lowest-logg spectrum.
   Search the raw directory, including logg values outside the requested axis.
3. Do not replace interior holes, upper-edge gaps, partial pairs or empty files.
   An incomplete lowest-logg source pair is an error; it is not skipped in favor
   of the next higher logg. All unresolved inputs are reported before reading.

For example, if the native lower boundary is logg=3.25, only a missing 3.0 node
uses 3.25; existing 3.25, 3.5, etc. retain their own spectra. If the boundary is
4.0, the requested 3.0--3.75 nodes all use 4.0. The boundary is determined
separately for each Teff and metallicity. A missing download of an entire
lower pair is indistinguishable from a native gap: download the intended raw
models first.

The CLI prints `will use lowest available logg=...` for every replacement
before loading spectra. Target/source coordinates and both source paths are
saved in the manifest and NPZ provenance. Copied nodes have **flat spectral
dependence on logg below the native boundary**, rather than new atmosphere
models or a physical extrapolation. Set `missing_nodes="error"` to require a
complete native domain; e.g. 15000--55000 K with logg=4.0--4.75, or
15000--30000 K with logg=3.0--4.75, at the example's five Zs.
`missing_nodes="logg"` remains BOSZ-only.

Normalized flux is the native H_lambda divided by its interpolated continuum.
Continuum samples may differ between models. Exactly identical duplicate
continuum rows are collapsed; duplicate wavelengths with conflicting fluxes
fail instead of being averaged. Missing continuum coverage in a requested
region also fails. `normalized=false` retains native H_lambda values (no
4*pi factor, per-region scaling, or flux-density Jacobian conversion).
Prepared flux is float32; the CLI enables JAX x64 for axes and wavelengths.

Python users can call
`prepare_tlusty(..., source_format="ob2025", missing_nodes="lowest_logg")` with the
same axes, bounds and sampling, using `flux_kind="normalized"`. The default
`source_format="auto"` recognizes either OB2025 or the old
`Z*_vt2/Z*_ostar` layout and refuses ambiguous mixed roots. Explicit
`source_format="legacy"` preserves the old wavelength declarations and
BSTAR-at-30000 behavior. The recipe CLI always selects OB2025 for
`library="tlusty"`.
`check_spectral_inputs("tlusty", ..., axes=..., missing_nodes="lowest_logg")`
previews the same OB2025 input plan. Both APIs default to strict `"error"`
when the policy is omitted. Loading and fitting the resulting NPZ use the same
`SpecFit.from_grid` example above.

## Missing BOSZ nodes

Set `"missing_nodes": "logg"` to enable these rules, keeping Teff, metallicity,
alpha abundance, carbon abundance and vmic fixed:

1. Use the requested native file whenever it exists.
2. If it is missing, use the native files at logg minus/plus 0.5. Resample each
   onto the same output wavelengths, then average their flux with weights 0.5.
   For normalized grids, divide each source's H by its own continuum first.
3. At the **upper end of the requested logg axis only**, if the upper neighbour
   is also absent and the immediate lower neighbour exists, copy the lower
   spectrum. For example, a missing logg=5.0 can use 4.5 when 5.5 is absent.
4. All other cases fail during the input check. There is no lower-edge copy,
   interior one-sided copy, wider-neighbour search or recursive filling.

Neighbours outside the requested axes are allowed and must be present locally.
For example, filling logg=3.0 can require the logg=2.5 source. Geometry follows
each source's own logg (`ms` at <=3, `mp` above 3). This can cross the geometry
boundary. The upper-edge copy makes the flux constant over the copied interval
at those other parameter values; neither replacement is an additional native
atmosphere calculation or an independently validated interpolation accuracy.

Before loading spectra, the script prints English messages such as:

```text
Missing ...g+3.5...: will interpolate linearly in logg from 3 and 4.
Missing ...g+5.0...: upper logg neighbor is also missing; will use logg=4.5 at the requested upper edge.
Input check passed: 4095 requested nodes, 4080 source files, 18 replacements.
```

The counts above illustrate the current GAOES recipe with a complete native
library. All unresolved nodes and missing neighbours are reported together;
failure occurs before spectrum reading or grid allocation. File checks use the
local filesystem, so incomplete downloads can also trigger this policy. They
do not check gzip/FITS contents. Corrupt files, invalid flux and insufficient
wavelength coverage remain errors during reading, and never trigger replacement.

Python users can inspect the same plan with
`check_spectral_inputs("bosz", data_dir, axes=axes, missing_nodes="logg")`, then
call `prepare_bosz(..., missing_nodes="logg")`. The latter repeats the input
check and defaults to strict mode when the option is omitted. The explicit
`source_overrides` API remains available in strict mode and cannot be combined
with `missing_nodes="logg"`.

## Legacy range provenance

The small tables were copied from the corresponding `jaxspec/examples` CSVs;
there is no runtime dependency on a sibling checkout. Original decimal bounds
and row order are preserved. IRD values were converted from nm to Angstrom by
exact decimal multiplication by ten. Missing legacy `order` columns were
replaced by zero-based row indices.

| Table | Legacy file | Medium used in the legacy generation script |
| --- | --- | --- |
| `gaoes_rv.csv` | `wavranges_gaoes-rv.csv` | air, both BOSZ and Coelho |
| `ird_h.csv` | `wavranges_ird_h.csv` | vacuum, both BOSZ and Coelho |
| `hds_red.csv` | `wavranges_hds-red.csv` | air, Coelho |
| `hds_ybred.csv` | `wavranges_hds-Ybred.csv` | air, Coelho |
| `kpf_green.csv` | `wavranges_kpf-green.csv` | vacuum, Coelho |
| `kpf_red.csv` | `wavranges_kpf-red.csv` | ambiguous: both air and vacuum, Coelho |
| `apogee.csv` | `wavranges_apogee.csv` | vacuum, Coelho |

The medium annotations preserve the scripts' declarations; they are not an
independent calibration of the original observed wavelength solutions. These
intervals describe particular legacy reductions, not universal instrument
coverage. KPF-red is marked `unknown` because its original CSV has no medium
metadata and was reused with both settings. Confirm its definition before
adding a recipe with an explicit declaration. The GAOES and IRD examples share
their respective tables between BOSZ and Coelho.

## Korg optical grids

The public Python entry point is `jaxstar.specfit.prepare_korg`. Its synthesis
backend lives in `src/jaxstar/specfit/korg.py`; `libraries.py` supplies the entry
point. Korg does not use the fixed legacy-NPZ `_SCHEMAS` table. Common NPZs store
their own axis names and coordinates, with `library="korg"`.

Install the optional Python extra, then register Korg in JuliaCall's environment:

```bash
python -m pip install -e '.[korg]'
python -c 'import juliapkg; juliapkg.add("Korg", "acafc109-a718-429c-b0e5-afd7f8c7ae46", version="=1.3.1")'
```

JuliaCall manages Julia and PythonCall in the active Python environment. On its
first import it resolves the registered Korg dependency. This backend currently
requires **Korg 1.3.1**. No Julia dependency is registered globally by importing
jaxstar, and no Julia package is needed to load or fit an already generated NPZ.
An existing compatible Julia environment can also be selected using the
[JuliaCall configuration options](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
The initial GES download is about 0.87 GB unpacked; Julia/Korg manage their data
under the Julia depot. Subsequent nodes reuse the parsed line list in one process.

`recipes/korg_gaoes.json` uses all 15 GAOES regions with a 5 Å margin and
7500 log-spaced pixels per region. Its axes are Teff 4000–7000 K in 150 K steps,
logg 3–5 in 0.5 steps, [Fe/H] −0.5–0.5 in 0.25 steps, and [alpha/Fe]
−0.25–0.25 in 0.25 steps (inclusive endpoints): **1575 atmosphere nodes**.
Microturbulence is empirical at each node, with no vmic axis.

The smaller `recipes/korg_gaoes_mini.json` has 27 atmosphere nodes and GAOES
regions 2 and 3:

```bash
PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/korg_gaoes_mini.json --dry-run

JULIA_NUM_THREADS=1 PYTHONPATH=src python scripts/specfit/prepare_grid.py \
  scripts/specfit/recipes/korg_gaoes_mini.json \
  --output-dir /path/to/prepared/korg_gaoes_mini
```

Korg needs no `--data-dir`. `--dry-run` validates and expands the recipe without
starting Julia, downloading data, checking MARCS or writing output. A build
checks MARCS interpolation at **every requested node before synthesis**.
Unsupported points, invalid microturbulence, uncovered wavelengths and synthesis
failures raise errors with context. No missing-node replacement is performed.
Parameter limits and region selection come from the recipe; the mini recipe is
an example, not a backend domain restriction.

The initial supported physics configuration is:

| Quantity | Definition |
| --- | --- |
| Axes | `teff` (K), `logg` (log10 cgs), `feh` = [Fe/H], `alpha` = [alpha/Fe] |
| Solar composition | Korg's explicit `bergemann_2025_solar_abundances` |
| Alpha elements | Korg standard O, Ne, Mg, Si, S, Ar, Ca, Ti |
| Abundance mapping | `format_A_X(feh, feh + alpha, solar_abundances=...)`; second argument is [alpha/H] |
| Atmosphere | `interpolate_marcs(teff, logg, A_X)`, no abundance clamping or node perturbation |
| Line list | GES with molecules, including Korg's built-in line filtering |
| Microturbulence | Existing `empirical_vmic(teff, logg, feh)` at each node; no independent vmic axis |
| Output | Intrinsic normalized `solution.flux / solution.cntm` |
| Hydrogen | Hydrogen lines and MHD occupation probabilities enabled for this optical backend |
| Additional broadening | No rotation, macroturbulence or instrumental convolution |

The abundance and atmosphere calls follow the
[Korg API documentation](https://ajwheeler.github.io/Korg.jl/stable/API/) and
[abundance guide](https://ajwheeler.github.io/Korg.jl/stable/Abundances/).
MARCS uses its own Grevesse 2007 reference mixture; passing `A_X` lets Korg map
onto those atmosphere coordinates. It interpolates existing structures rather
than recomputing the structure for the chosen synthesis microturbulence or
arbitrary elemental mixtures. Korg uses spherical atmospheres for logg < 3.5
and planar ones otherwise. Its very metal-poor MARCS branch fixes the atmosphere
alpha/C mixture; this remains Korg's documented behavior.

The existing empirical vmic relation is used without adding clipping or a new
hard temperature cutoff. Its documented domain is 4000 < Teff < 7000 K,
0 < logg < 5, -4.5 < [Fe/H] < 1. Recipe points outside that domain evaluate the
same prescription and are listed in the manifest's
`korg.vmic_outside_documented_range`; they are extrapolations, not newly validated
microturbulence predictions. Nonpositive/nonfinite values are errors.
The relation is discontinuous at logg=3.5, so interpolation across that boundary
needs separate validation. Individual abundance axes and an independent vmic
axis are not supported in this implementation and are rejected explicitly.

### Wavelengths and sampling

CSV/recipe wavelengths must explicitly specify air or vacuum. Output preserves
that convention. For air output, Korg's `air_to_vacuum` converts each target
coordinate before synthesis/resampling. GES source wavelengths are air; Korg's
GES loader converts the line positions to vacuum. Korg synthesizes in vacuum
at **0.0025 Å** by default, merges overlapping windows and linearly interpolates
onto the requested output rows. Each row must be bracketed by native samples;
there is no extrapolation across a gap or beyond a window.

The optional recipe block `"korg": {"synthesis_step": 0.0025, "line_buffer": 10.0}`
controls native vacuum spacing and the line inclusion buffer (both in Å).
These are distinct from `sampling.pixels` or `sampling.velocity_step`, which
control the saved wavelength grid. Native spacing cannot exceed the smallest
output spacing in vacuum. No resolving power is imposed during generation.

GES has line wavelengths spanning approximately 4748.793–8949.973 Å **in air**.
A request, including synthesis padding and the line buffer, must fit inside
that span after conversion to vacuum. The full GAOES CSV is covered. GES is not
an infrared line list; requests outside its coverage fail. Coverage is a check
of the available wavelength span, not a claim of completeness of atomic data.

### Checkpoints and Python API

The CLI stores one completed node per file in `OUTPUT_DIR.cache/`, or under an
explicit `--cache-dir`. A failed build leaves these checkpoints and publishes no
final bundle. Repeat the same command to resume. Existing completed output
bundles are still refused; to rebuild a bundle from the same cache, choose a new
output directory and pass the old cache path explicitly.

Cache namespaces hash axes, output coordinates/medium, synthesis settings,
empirical-vmic values/source, solar composition, Korg source/artifact definitions,
the actual GES file, Julia environment/version/thread count and backend source.
Changing these settings selects a different namespace. Each node records its
coordinates, vmic, namespace and flux checksum; corrupt checkpoints raise an
error. Keep Julia packages and artifact data unchanged during a build. Save
`manifest.json` alongside `grid.npz` for physics provenance; the NPZ alone is
sufficient for subsequent loading and fitting.

```python
import jax
jax.config.update("jax_enable_x64", True)
from jaxstar.specfit import prepare_korg, save_spectral_grid

spectra, report = prepare_korg(
    axes={"teff": [5750, 6000], "logg": [4.0, 4.5],
          "feh": [-0.25, 0.0], "alpha": [0.0]},
    wavelength_ranges=[(5700, 5800)], wavelength_medium="air",
    sampling="log", pixels=7500, cache_dir="/path/to/korg-cache",
    return_report=True,
)
save_spectral_grid("/path/to/grid.npz", spectra)
```

The saved grid embeds the empirical relation in its fluxes. During fitting,
use its four atmospheric axes; the BOSZ-specific `use_empirical_vmic=True`
inference option is not needed or supported for this grid.

### Optional numerical validation

After building the mini recipe, run:

```bash
JULIA_NUM_THREADS=1 PYTHONPATH=src python scripts/specfit/validate_korg_grid.py \
  /path/to/prepared/korg_gaoes_mini \
  --output-dir /path/to/validation/korg_gaoes_mini
```

This checks exact nodes, `SpecFit.from_grid`, direct Korg spectra at every cell
center, one center at half the native wavelength step, and a nonzero-alpha
synthesis. It writes residual metrics, numerical arrays and an overlay/residual
plot. It launches additional real syntheses and is intended for small grids.
Interpolation and sampling errors depend on the recipe; this is not a test of
parameter-inference accuracy against observations. Ordinary unit tests use a
fake synthesis backend and require no Julia/network access.

Local numerical check (2026-10-25, Korg 1.3.1 / Julia 1.13.2, one Julia thread):

- 27 mini-recipe nodes produced shape `(3, 3, 3, 1, 2, 7500)`.
- All exact-node interpolation errors were zero. Cached regeneration reproduced
  the flux arrays exactly. A separate Python process loaded the grid into
  `SpecFit.from_grid` without importing JuliaCall.
- Eight cell centers: normalized-flux RMS error `0.00128502`, maximum absolute
  error `0.00797217` relative to direct synthesis.
- Native step 0.0025 versus 0.00125 Å at the first center: RMS `3.49665e-5`,
  maximum absolute difference `0.000533065`.
- The alpha=0.2 direct synthesis passed. Empirical vmic ranged from 1.039 to
  1.33575 km/s across the grid. Median warm node time was 5.32 s for the two
  selected regions, excluding setup, preflight and line-list loading.

These numbers describe the checked mini recipe only. Local generated artifacts
are in ignored `benchmark-results/korg-grid/`; they are not package data.
