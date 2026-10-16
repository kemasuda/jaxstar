# MIST v2.5 track validation

Validated on 2026-10-15. The first photometric builder supports `afe=0`, both
rotations, and all published native mass/FeH/EEP nodes. Small regions are a
validation choice, not an implementation limit. v1.2 remains the default;
v2.5 isochrone building and nonzero-alpha photometric grids are not enabled.
This directory records the methods and summarized checks. Generated JSON
reports, grids, posterior arrays and figures stay in the ignored
`benchmark-results/` directory, following the existing benchmark convention.

## Optional 66-column structural grid (2026-10-16)

The opt-in `include_structure=True` / `--include-structure` builder selection
retains all 66 non-isotope columns in the v2.5 EEP files, including surface
convection-zone boundaries, three turnover times, total moment of inertia,
apsidal constant, core masses and seismic diagnostics. It preserves their
original names/units. `star_mass` already exists; the other 65 arrays are added
alongside the original nine scalar fields and 14 photometric bands. This
deliberately keeps raw names such as `log_Teff` alongside the existing
`logTeff`, avoiding changes to either the fitting API or the source schema.
The default compact builder output is unchanged.

The validation uses the same domain as `examples/mistfit/fit_track_v25.ipynb`:
mass 0.8–1.5 solar masses, FeH −0.25/0/+0.25, EEP 202–550, alpha=0,
rotation=0.4. There are 141 native tracks and 49,209 nodes on a
`(47, 3, 349)` grid. The baseline grid and saved inference bundle were not
modified. The expanded grid was built offline from the existing raw cache
in 65.2 seconds.

Reproduce from the repository root (set the cache location if needed):

```bash
PYTHONPATH=src python -m jaxstar.mistfit.mistgrid.create_tracks \
  benchmark-results/mist-track-v25/grid_structure.npz \
  --version 2.5 --rotation 0.4 --afe 0 \
  --mass-range 0.8 1.5 --eep-range 202 550 --feh -0.25 0 0.25 \
  --include-structure --offline

MPLBACKEND=Agg PYTHONPATH=src python benchmarks/check_mist_v25_structure.py \
  --baseline benchmark-results/mist-track-v25/grid.npz \
  --expanded benchmark-results/mist-track-v25/grid_structure.npz \
  --posterior examples/mistfit/results/track_nuts_v25/run_20261015_234346_849469 \
  --output benchmark-results/mist-track-v25/structure_check
```

The check requires the saved local posterior, original archives referenced
by the expanded grid, and `psutil` for process-memory measurements. It never
samples or downloads. Use a new output directory when repeating it. The
machine-readable record is written to
`benchmark-results/mist-track-v25/structure_check/report.json`. The reviewed
report copies are kept locally in `benchmark-results/mist-v25-validation/`.

Validation results:

- All 66 columns at all 49,209 nodes agree exactly with an independent
  header/`numpy.loadtxt` read of the three native archives. All are finite in
  this selected domain. The 102 omitted columns are isotope abundances.
- All original NPZ arrays agree exactly (metadata is expected to differ).
- All 55 pre-existing public predictions agree exactly at all 20,000 saved
  NUTS draws (four chains, 5,000 draws each).
- Log likelihood, constrained log joint and their four latent gradients are
  finite and exactly equal at 32 draws spanning the saved chain/draw array.
  This compares the same saved observations and model arguments.
- Every additional native output is finite at all saved draws. Offline unit
  tests also insert an optional-field NaN and confirm unchanged fitting
  values/gradients with per-field support; the real sample alone cannot test
  the missing-value case.

| Measurement | Baseline | With 66 native columns |
| --- | ---: | ---: |
| Compressed NPZ | 7.92 MiB | 24.58 MiB |
| Loaded model array storage | 10.14 MiB | 37.59 MiB |
| Median load time | 0.098 s | 0.201 s |
| Observed process RSS increase | 54.8–63.6 MiB | 121.9–137.3 MiB |

Array storage includes axes, `dtdeep`, numerical fields and Boolean validity
masks. RSS includes temporary allocations and runtime overhead and is not
equal to array storage. Timing/RSS use three fresh CPU processes per grid,
after JAX device initialization, with a warm filesystem cache. These are
small-grid load measurements, not full-grid or NUTS-throughput benchmarks.

Saved-posterior summaries are median [16th, 84th percentiles]:

| Quantity | Result |
| --- | --- |
| Surface convective envelope thickness | 0.3270 [0.2960, 0.3538] solar radii |
| Thickness / stellar radius | 0.3095 [0.2833, 0.3315] |
| Convective envelope mass | 0.02761 [0.02106, 0.03449] solar masses |
| Envelope mass / stellar mass | 0.02938 [0.02149, 0.03791] |
| Local turnover, one mixing length above base | 19.53 [16.50, 22.24] days |
| Local turnover, half a mixing length above base | 26.67 [22.98, 29.91] days |
| Global turnover | 41.26 [34.45, 47.04] days |
| Total moment of inertia | 6.834 [6.411, 7.288] × 10^53 g cm² |
| Apsidal constant k2 | 0.01558 [0.01462, 0.01656] |

The script saves all 66 raw interpolated columns and derived diagnostics in
`posterior_structure.npz`, preserving `(chain, draw)` shape, and a nine-panel
`posterior_structure.png`. Unsupported model values are represented as NaN
in the postprocessing artifact. The saved fit's inference bundle is read only.
These distributions are conditional on the original observations, priors and
grid; no new constraints or sampling have been introduced.

Units and finite zero semantics were checked in the authors'
[MIST2 work directories](https://zenodo.org/records/15232687), file
`rotating_workdirs.tgz`, member
`feh_m025_afe_m2_vvcrit0.4/00095M_dir/src/run_star_extras.f90`
(SHA256 `26541bb997749a78a972f017d23344e716e708a88725e1bb60ed03b59e459386`).
The inspected member was read from a 1 MiB archive prefix, outside the raw
model cache; this prefix is not a complete archive. The code divides boundary
masses/radii by `msun`/`rsun`, emits turnover times in seconds, and computes
total moment of inertia from cgs mass/radius or rotational inertia arrays.
It initializes convection-zone diagnostics to zero when no qualifying surface
zone exists. Therefore the builder retains finite zeros unchanged; the
postprocessing example requires positive envelope depth/mass and positive
turnover times for its derived summaries. All draws in this example pass.
The [official column descriptions](https://mist.science/README_tables.pdf)
and [MIST2 paper, Section II.10](https://arxiv.org/html/2602.22012v1#S2.SS10)
describe the three turnover-time definitions.

All optional fields currently use linear interpolation in their stored
quantity, including `phase`. A fractional phase value is a numerical
diagnostic, not a categorical evolutionary classification. The checks above
establish faithful storage and unchanged existing fitting behavior in this
domain; they do not validate interpolation accuracy or physical interpretation
through every structural discontinuity or evolutionary phase.

## Independent reference data

The [official catalogue](https://mist.science/model_grids.html) supplies the
native EEP archives and [UBVRIplus BC v2](https://mist.science/BC_tables/v2/UBVRIplus.txz).
The [official track form](https://mist.science/interp_tracks.html), with
photometry selected, returns both full theoretical tracks and CMD tracks in
one ZIP. Their age, logT, logg and logL arrays agree before comparing magnitudes.
The form requests used MIST2, UBVRIplus, Av=0, alpha=p0:

| Reference | Mass (solar) | FeH | Rotation | Official output |
| --- | --- | --- | --- | --- |
| `reference.zip` | 1 | 0 | 0.4 | [ZIP](https://mist.science/output/MIST_eep_6ad0d914c6947.zip) |
| `reference-m025.zip` | 0.1, 1, 10 | −0.25 | 0 | [ZIP](https://mist.science/output/MIST_eep_6ad0dc41dc360.zip) |
| `reference-agb.zip` | 2, 3 | 0 | 0.4 | [ZIP](https://mist.science/output/MIST_eep_6ad0dfcc6f87b.zip) |
| `isoref.zip` | logage=9 isochrone | 0 | 0.4 | [ZIP](https://mist.science/output/MIST_iso_6ad0da4b30201.zip) |

The isochrone is only a paired theoretical/CMD reference for the surface FeH
convention, not an implemented v2.5 isochrone loader. Generated server URLs
may expire. The local `benchmark-results/mist-v25-validation/reference_report.json`
contains the per-track discrepancies. The source and fixture SHA-256 hashes
are retained here for the offline regression data:

| File | SHA-256 |
| --- | --- |
| BC v2 `UBVRIplus.txz` | `e39a5d8871a36b450ee79928d82a9abbe104fe3771c8cb3617da60a31d911176` |
| `reference.zip` | `27de4881d1926de45b1c89b775110114d19834b633df754b8ae4e6ab84d9d4d4` |
| `reference-m025.zip` | `4211845267a007ce7b4130c54f2ea441424808ad6e84406098cded56ca1cb48a` |
| `reference-agb.zip` | `5ebb63449478c7c62a072b3203cbdc1575fb2de5013af6e69dfe3de183487b13` |
| `isoref.zip` | `8dad62e1c2c20435bf53a399b3f728b084927f93fa6037e36d7a10c14f231460` |
| `official_reference.npz` | `3747cb5aaa0fa4d0dea0a0b71eb47e5d2d014d01d6c40333a5f8671343f5c56d` |

A compact offline fixture at
`tests/data/mist_v25/official_reference.npz` freezes 24 evenly spaced supported
nodes per track plus the largest-residual node, their unmodified official
magnitudes, and the local BC table stencils. It also includes sampled official
isochrone FeH values. No online requests occur in unit tests.

Reproduce the full comparison and regenerate the fixture from local archives:

```bash
mkdir -p benchmark-results/mist-v25-validation
PYTHONPATH=src python benchmarks/validate_mist_v25.py \
  --bc /path/to/UBVRIplus.txz \
  --reference /path/to/reference.zip /path/to/reference-m025.zip /path/to/reference-agb.zip \
  --iso-reference /path/to/isoref.zip \
  --fixture tests/data/mist_v25/official_reference.npz \
  --output benchmark-results/mist-v25-validation/reference_report.json
```

## Recipe and comparison limits

v2.5 uses `log_surf_cell_z`; the BC v2 columns begin with
`lgTef logg Fe_H a_Fe Av Rv`. Temperature is already logarithmic. Alpha planes
must be selected explicitly. The existing Steffen logT / linear logg / Steffen
metallicity kernel reproduces the solar-scaled official outputs with updated
normalizations `(Zsun, Xsun) = (0.01685, 0.73465)` and `Mbol_sun=4.74`.
These normalization constants are inferred and checked against the paired
official outputs; they are not the native track's initial X and Z, and are not
claimed to come from a published v2.5 generator source.

The BC coordinate is `min(log10(Z/Zsun), feh_init+0.1)`. The reported surface
FeH is that coordinate minus `log10(surface_h1/Xsun)`, matching the CMD
convention. It is not a direct elemental iron/hydrogen measurement. Retaining
`surface_fe56` in the parser allows a future separately specified definition.

All 7,106 supported reference nodes were compared in 14 bands. Maximum absolute
differences are 1.15e-5 mag (1 solar mass, FeH=0), 1.77e-5 (0.1 solar mass),
4.70e-5 (1 solar mass, FeH=−0.25), 3.29e-4 (10 solar masses), and below 8.40e-6
for the 2/3 solar-mass tracks. The regression
gate is 5e-4 mag. Surface FeH over EEP 202–550 agrees within 6.88e-7 dex
(gate 5e-6). These comparisons do not certify every catalogue metallicity,
mass, evolutionary phase or alpha mixture.

The builder uses float64 and the native named metal fraction, preserves
missing native nodes, and leaves BC coordinates outside the published tables
unsupported. BC metallicities cover −3 to +0.5 while tracks extend to −4.
The very metal/H-depleted late tails do not reproduce all official CMD values;
those values can remain finite despite tiny native surface metal fractions.
They are not used to extend BC support. Surface FeH in these tails has not
been independently validated.

Nonzero-alpha probes of the official generator require different composition
normalizations. Reusing the solar-scaled or v1.2 constants is incorrect, so
`create_track_grid` rejects nonzero `afe` before acquisition. The source
catalogue and BC parser recognize all published alpha selections, including
the absent FeH=+0.5/afe=+0.6 combination, for later validation.

## Native archive build and model smoke check

Two complete rotating, solar-scaled archives (FeH=−0.25 and 0) were imported
through the raw cache and built with no mass/EEP bounds. The result retains
342 native tracks on a `(171, 2, 1721)` rectangular grid, with mass 0.1–300
solar masses and EEP 1–1721. Ragged tails and missing photometry remain masked.
The compressed NPZ is about 68 MB. The local record
`benchmark-results/mist-v25-validation/native_build_report.json` contains
source hashes, field coverage, and the numerical smoke prediction.

Five high-mass headers differ from their nominal file masses by about 1e-9
solar masses. The builder accepts roundoff up to 1e-8, uses the nominal mass
axis, and records the original header value. Larger mismatches still fail.

Reproduce the native build and an interior prediction/derivative check:

```python
import jax
import jax.numpy as jnp
import numpy as np
from jaxstar.mistfit import MistModel
from jaxstar.mistfit.mistgrid.create_tracks import create_track_grid

path = create_track_grid(
    "mistgrid_v25_native.npz", version="2.5", feh=[-0.25, 0.],
    rotation=0.4, afe=0., local_dir="/path/to/official/archives",
    raw_cache_dir="/path/to/raw/cache", offline=True,
)
jax.config.update("jax_enable_x64", True)
model = MistModel.from_tracks(path)
keys = ("age", "teff", "logg", "radius", "jmag_abs", "feh", "dtdeep")

def predict(model, coordinates):
    result = model(dict(zip(("mass_init", "feh_init", "eep"), coordinates)), keys=keys)
    return jnp.stack([result[key] for key in keys])

point = jnp.array([1.01, -0.125, 350.5])
assert np.isfinite(jax.jit(predict)(model, point)).all()
assert np.isfinite(jax.jit(jax.jacfwd(predict, argnums=1))(model, point)).all()
```

An exact native node (1 solar mass, FeH=0, EEP=350) also reproduces the NPZ's
age, temperature and J-band magnitude to relative tolerance 1e-12. This is an
integration check, not a new scientific calibration or a posterior comparison
between MIST releases. The other 15 v2.5 metallicity archives have not been
built in this validation run.

The initial small region was also rebuilt offline from the same cache, using
`mass_range=(0.8, 1.2)` and `eep_range=(202, 550)`. Its 82 tracks populate all
28,618 nodes of the `(41, 2, 349)` grid; every stored field is finite. The NPZ
is about 4.8 MB. Its interior smoke prediction matches the full native grid
to relative tolerance 1e-12. The local record is
`benchmark-results/mist-v25-validation/small_build_report.json`.
