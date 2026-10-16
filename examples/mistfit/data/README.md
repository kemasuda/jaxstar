# Example MIST track grid

`mistgrid_tracks_v1.2_small.npz` is the grid used by
[`fit_track.ipynb`](../fit_track.ipynb). It is included in the repository so the
notebook can load it without downloading or building a grid. The example data
are not included in the Python wheel or source distribution.

The grid uses MIST v1.2, rotation `v/vcrit=0.4`, `[alpha/Fe]=0`, and UBVRIplus
bolometric corrections at `Av=0`, `Rv=3.1`:

| Coordinate | Selection |
| --- | --- |
| Initial mass | All 33 native masses from 0.8 to 1.5 solar masses |
| Initial FeH | -0.25, 0, +0.25 dex |
| EEP | Every integer from 202 to 550 (349 nodes) |

These are also the default inference bounds in the notebook. This small domain
is chosen for the example observations; select a suitable domain when fitting
other stars. Age is an output, with the notebook's soft upper bound at 13.8 Gyr.

## Regenerate

With jaxstar's dependencies installed, run from `examples/mistfit` (the notebook
directory):

```bash
PYTHONPATH=../../src python -m jaxstar.mistfit.mistgrid.create_tracks \
  data/mistgrid_tracks_v1.2_small.npz \
  --version 1.2 --rotation 0.4 --afe 0 \
  --feh -0.25 0 0.25 \
  --mass-range 0.8 1.5 --eep-range 202 550 \
  --overwrite
```

`--overwrite` explicitly replaces the bundled file. Missing source archives
are downloaded; valid cached archives are reused. The mass/EEP selection
reduces the generated NPZ, but each of the three selected FeH archives is
downloaded in full. Add `--offline` to require cached sources, or combine it
with `--local-dir /path/to/archives` to import existing archives.

To build a full grid, choose a different output path and omit `--feh`,
`--mass-range`, and `--eep-range`. See the
[grid guide](../../../docs/mistgrid.md) for cache settings and other selections.

## Sources and provenance

The current track builder generates this NPZ from the
[official MIST EEP tracks and BC tables](https://mist.science/model_grids.html):

- `MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_EEPS.txz`
- `MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz`
- `MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_EEPS.txz`
- `UBVRIplus.txz` from the MIST v1.2 BC tables

The NPZ's `metadata_json` records the build settings, source URLs and SHA-256
hashes, native track members, field coverage, and builder/parser hashes.
Recorded input paths describe the build machine; they are not needed to load
the grid. Rebuilding elsewhere can change provenance metadata and the NPZ
hash even when the numerical arrays are unchanged.

For scientific use, follow the [MIST citation guidance](https://mist.science/references.html).
The v1 models and EEP method are described by
[Choi et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...823..102C/abstract)
and [Dotter (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJS..222....8D/abstract).
