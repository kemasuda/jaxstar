# MIST grids and downloads

[README](../README.md) · [MistFit guide](mistfit.md) · [SpecFit](specfit.md)

Track grids are built once from published MIST EEP tracks and bolometric
correction (BC) tables, then loaded with `MistModel.from_tracks(path)`.
Downloaded source archives are cached separately from generated grids, so
rebuilding an NPZ or updating jaxstar reuses existing downloads.

## Build a track grid

The builder supports **MIST v1.2 (default) and v2.5**, with rotation `0.0` or `0.4`
and `afe=0` only. It uses UBVRIplus BCs with Av=0 and Rv=3.1. Choose a rotation rate; the default
includes all published FeH archives and all native masses and EEPs for that
selection.

**First use: create the grid once.** This downloads any missing source archives
and writes the generated grid to the output path you specify:

```python
from jaxstar.mistfit.mistgrid.create_tracks import create_track_grid

path = create_track_grid(
    "mistgrid_tracks.npz", rotation=0.4, afe=0.,
)
```

`"mistgrid_tracks.npz"` is the **output file path**. This relative path saves the
file in the current working directory; use an absolute path to choose another
location. The returned `path` points to the generated file. Keep this NPZ for
subsequent fitting sessions.

A full build includes 15 metallicity archives for v1.2 or 17 for v2.5 and can require several GB for
source data, build memory and the generated grid. Loading it for fitting also
requires memory for the uncompressed arrays. Grids are not bundled with the
package.

To build a smaller region, select published FeH values and native mass/EEP bounds:

```python
path = create_track_grid(
    "mistgrid_tracks_v25_small.npz", version="2.5", afe=0.,
    feh=[-0.25, 0.], mass_range=(0.8, 1.2), eep_range=(202, 550),
)
```

This is an initial validation example, not a built-in domain restriction.
Omit `mass_range`, `eep_range` and `feh` to retain the full native v2.5 selection.
Mass/EEP bounds reduce the generated grid but each selected FeH archive is
downloaded in full. Nonzero v2.5 `afe` is registered in the source catalogue,
but the photometric builder rejects it until its composition conversion has
been validated. See the [v2.5 validation record](../benchmarks/mist_v25/README.md).

For an end-to-end example at `afe=0`, see the
[MIST v2.5 track notebook](../examples/mistfit/fit_track_v25.ipynb). It starts
with a grid-building command using `--include-structure`, then runs NUTS and
plots selected structural quantities from the posterior.

The same builder is available from the command line:

```bash
python -m jaxstar.mistfit.mistgrid.create_tracks mistgrid_tracks.npz --rotation 0.4 --afe 0
```

Optional flags include `--version 2.5`, `--feh -0.25 0 0.25`, `--mass-range 0.8 1.5`,
`--eep-range 202 550`, `--raw-cache-dir /path/to/cache`,
`--local-dir /path/to/archives`, `--offline` and `--overwrite`.

To rebuild an existing output, pass `overwrite=True` (CLI `--overwrite`).
A failed build leaves the previous NPZ intact.

**Subsequent sessions: load the saved grid.** Run the following with the path
to your existing NPZ; grid creation is needed only when you want a new grid.
Enable JAX x64 before loading:

```python
import jax
from jaxstar.mistfit import MistModel

jax.config.update("jax_enable_x64", True)
model = MistModel.from_tracks("mistgrid_tracks.npz")
```

`from_tracks` loads a prebuilt file and never initiates a download. See the
[MistFit guide](mistfit.md) for the next steps.

## Reuse downloads and work offline

Raw-cache selection follows this order:

1. Explicit `raw_cache_dir=...` on a builder (`cache_dir=...` on low-level cache calls).
2. The `JAXSTAR_MIST_RAW_DIR` environment variable.
3. `mist/raw` under the platform-specific jaxstar user cache.

Cache keys include the MIST release, data kind and official archive filename,
which identifies rotation, composition and photometry where applicable. They
are independent of the jaxstar package version and output grid path. A different
MIST release uses separate entries.

The builder reuses valid cached archives before attempting a download. To
import files already downloaded elsewhere, use `local_dir` with their official
archive names. Files are copied into the cache; the originals are unchanged:

```python
path = create_track_grid(
    "mistgrid_tracks_rebuilt.npz",
    local_dir="/path/to/downloaded/archives",
    raw_cache_dir="/path/to/mist-raw-cache",
    offline=True,
)
```

`offline=True` permits cached or local inputs only and fails if any are missing,
including the BC archive. An existing valid cache takes precedence over a local
import. Reuse checks file size and SHA-256 locally. Downloads are streamed and
validated as complete XZ/tar archives before publication. Failed downloads or
refreshes preserve the previous successful entry; partial transfers restart
on retry rather than resume.

### Individual archives

Descriptors list sources without downloading. The low-level cache API lets
you fetch, import or extract individual inputs:

```python
from jaxstar.mistfit.mistgrid.sources import track_archives, bolometric_correction_archive
from jaxstar.mistfit.mistgrid.cache import fetch_archive, extract_archive

sources = track_archives(rotation=0.4, afe=0., feh=None)
source = sources[0]
archive = fetch_archive(source)  # cache hit, or download on a miss
bc_archive = fetch_archive(bolometric_correction_archive())
directory = extract_archive(source, offline=True)
```

`fetch_archive(source, local_path=...)` imports an existing archive.
`force=True` explicitly refreshes a cache entry. Receipts record the source URL,
size, locally computed SHA-256 and import/download origin. The hash is checked
on reuse to detect local corruption. Previous archive contents are retained
when a refresh changes the file.

## Coverage and interpolation support

Mass and EEP axes retain every native node from the selected tracks. Bounds are
inclusive; at least two masses, two FeH values and three EEPs are required. The
builder selects native track files, excluding the `*_INTERP` files supplied in
the archives. Age bounds are set during inference, independently of grid creation.
v2.5 header masses within 1e-8 solar masses of their filename labels are treated
as roundoff and use the nominal file mass; differing header values are recorded
as `header_initial_mass`. Larger discrepancies are errors.

Missing mass/FeH combinations and differing track endpoints remain missing.
By default, the builder writes a partially populated rectangular array with
support tracked per field. `allow_missing=False` (CLI `--strict`) instead
requires a complete, finite rectangle with increasing ages. Malformed source
files always raise an error.

BC values use Steffen interpolation in logT (linear in the first/last intervals),
linear interpolation in logg, then Steffen interpolation in metallicity
with the MESA limited three-point edge recipe. Inclusive table endpoints are
supported. Outside the BC tables, photometry remains missing without clipping
or extrapolation; available physical fields are retained.

For v2.5, the BC v2 temperature coordinate is already `log10(Teff)`; the
matching fixed-alpha slice is read explicitly. Its metallicity coverage is
−3 to +0.5, narrower than the native track catalogue (−4 to +0.5). Thus the
most metal-poor tracks generally retain physical fields without photometry.

Surface `feh` follows the release's CMD convention: the BC metallicity is
`min(log10(Z/Zsun), feh_init + 0.1)`, and the reported surface value subtracts
`log10(X/Xsun)`. For v1.2, `(Zsun, Xsun) = (0.0134, 0.7381)`; for solar-scaled
v2.5, `(0.01685, 0.73465)`. These are conversion constants, distinct from the
initial composition in track headers. v2.5 reads `log_surf_cell_z` and
`surface_h1`; `surface_fe56` is retained by the parser but is not substituted
for the CMD surface-metallicity convention. BC coverage does not constrain
the physical `feh` field. Very H/metal-depleted late phases are outside the
surface-FeH validation interval; consult the validation record before using
that output there.

`MistModel` requires valid native fields at every nonzero interpolation corner
for each requested output. Exact nodes and faces can remain usable beside a
missing cell. Unsupported predictions return `-inf`. Use `model(...)` rather
than raw `model.grid` fields, which use finite placeholders for differentiation.

On loading, `dtdeep` is computed from valid native three-node physical-age
stencils; adjacent interior values are copied to track ends. Missing nodes or
nonpositive increments invalidate the affected stencils. Inference rejects
unsupported required values, and track plots leave gaps.

### Optional v2.5 structural columns

To retain the v2.5 structural diagnostics as well as the fitting fields, add
`include_structure=True` to `create_track_grid`, or `--include-structure` to
the CLI. This opt-in selection stores all 66 native non-isotope columns from
the v2.5 EEP header, with their original names and units; the 102 surface and
central isotope-abundance columns remain excluded. `star_mass` is already
present, so this adds 65 arrays. Original names such as `log_Teff` coexist
with the existing `logTeff` field, preserving both the raw schema and the
existing prediction API. The default field selection is unchanged.

See the [native-column reference](mistgrid-columns.md) for the complete
66-column list, descriptions and units.

For example, from the repository root, using an already populated raw cache:

```bash
PYTHONPATH=src python -m jaxstar.mistfit.mistgrid.create_tracks \
  benchmark-results/mist-track-v25/grid_structure.npz \
  --version 2.5 --rotation 0.4 --afe 0 \
  --mass-range 0.8 1.5 --eep-range 202 550 --feh -0.25 0 0.25 \
  --include-structure --offline
```

Select the original cache with `--raw-cache-dir path/to/cache` when necessary.
The expanded grid exposes these columns through `MistModel` without adding
them to the default inference outputs or likelihood:

```python
model = MistModel.from_tracks("benchmark-results/mist-track-v25/grid_structure.npz")
values = model(
    {"mass_init": 1.0, "feh_init": 0.0, "eep": 350.0},
    keys=("conv_env_top_radius", "conv_env_bot_radius",
          "conv_env_turnover_time_g", "total_moment_of_inertia"),
)
```

The optional fields use the existing per-field missing-value handling.
Finite source zeros/sentinels are retained; for example, zero convection-zone
boundaries/times can indicate no surface convective envelope. Interpret these
before summarizing derived quantities. `phase` is retained numerically and
linearly interpolated like the other columns; fractional predictions are not
categorical evolutionary labels. `--strict` also requires the added columns
to be finite. Storage, units, memory and saved-posterior checks are documented
in the [v2.5 validation record](../benchmarks/mist_v25/README.md).

### Grid metadata

Format 2 NPZs store `track_present`, NaNs for missing native fields and
`metadata_json`, which includes build settings, coverage counts, native members,
archive hashes, primary EEPs, BC conventions and builder/parser hashes:

```python
import json
import numpy as np

with np.load("mistgrid_tracks.npz", allow_pickle=False) as data:
    metadata = json.loads(data["metadata_json"].item())
```

Grids without primary EEP metadata use full-range posterior track plots unless
you supply a drawing interval.

`model.metadata` returns a copy of the release, composition, BC conventions,
input hashes and implementation hashes. This compact provenance also appears
under `grid.metadata` in saved fit run metadata. Older grids without these
entries remain loadable; missing provenance is not inferred from the filename.

## Isochrone grids

`MistModel.from_isochrones()` and `MistFit(observations=...)` use the MIST v1.2
synthetic photometry grid. The NPZ path is resolved in this order:

1. Explicit `path=...`.
2. `JAXSTAR_MISTGRID_PATH`.
3. The versioned user-cache location.

A missing default cache file triggers download and grid generation on first
use. Explicit or environment-specified paths must exist. Later loads reuse
the generated NPZ. The source CMDs are the
[MIST synthetic photometry grids](https://mist.science/model_grids.html#synthetic).

To generate another NPZ from cached sources:

```python
from jaxstar.mistfit.mistgrid.create_grid import create_mistgrid

create_mistgrid("/path/to/new/mistgrid_iso.npz", offline=True)
```

This builder uses the shared raw cache and can also reuse CMD folders or
archives beside the output NPZ.
