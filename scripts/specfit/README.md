# Reusable spectral grids

`prepare_grid.py` calls the public `prepare_bosz` or `prepare_coelho` API and
`save_spectral_grid`. It reads local raw spectra; it does not download models or
fit observations. Run it from a checkout installed with
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
The other recipes illustrate smaller domains and explicit region selection.
Every build checks all requested input paths before reading any spectra. Coelho
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
manifest alongside the NPZ to retain this replacement history; the NPZ itself
records the actual source filenames but not the replacement map.

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
- `missing_nodes`: `"error"` (default), or `"logg"` for BOSZ as described below.

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
