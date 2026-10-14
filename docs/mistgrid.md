# MIST grids and downloads

[README](../README.md) · [MistFit guide](mistfit.md) · [SpecFit](specfit.md)

Track grids are built once from published MIST EEP tracks and bolometric
correction (BC) tables, then loaded with `MistModel.from_tracks(path)`.
Downloaded source archives are cached separately from generated grids, so
rebuilding an NPZ or updating jaxstar reuses existing downloads.

## Build a track grid

The builder supports **MIST v1.2 only**, with rotation `0.0` or `0.4`, `afe=0`,
and UBVRIplus BCs with Av=0 and Rv=3.1. Choose a rotation rate; the default
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

A full build includes 15 metallicity archives and can require several GB for
source data, build memory and the generated grid. Loading it for fitting also
requires memory for the uncompressed arrays. Grids are not bundled with the
package.

To build a smaller region, select published FeH values and native mass/EEP bounds:

```python
path = create_track_grid(
    "mistgrid_tracks_small.npz",
    feh=[-0.25, 0., 0.25], mass_range=(0.8, 1.5), eep_range=(202, 550),
)
```

The same builder is available from the command line:

```bash
python -m jaxstar.mistfit.mistgrid.create_tracks mistgrid_tracks.npz --rotation 0.4 --afe 0
```

Optional flags include `--feh -0.25 0 0.25`, `--mass-range 0.8 1.5`,
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

`MistModel` requires valid native fields at every nonzero interpolation corner
for each requested output. Exact nodes and faces can remain usable beside a
missing cell. Unsupported predictions return `-inf`. Use `model(...)` rather
than raw `model.grid` fields, which use finite placeholders for differentiation.

On loading, `dtdeep` is computed from valid native three-node physical-age
stencils; adjacent interior values are copied to track ends. Missing nodes or
nonpositive increments invalidate the affected stencils. Inference rejects
unsupported required values, and track plots leave gaps.

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
