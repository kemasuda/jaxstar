# Spectral preparation: Milestones 2 and 2.5

Implemented scope, 2026-10-02: offline raw-file preparation for Coelho, BOSZ
and TLUSTY, optional log-uniform fitting-grid preparation, common prepared NPZ
storage/runtime loading, and compatibility
loading of existing jaxspec NPZs. Interpolation remains exclusively in
`jaxstar.grid`. No deterministic forward model or inference is implemented.
The frozen reference is the sibling `jaxspec` scientific source at `3b4ab913`
and characterization artifacts added at `52a255d`; that checkout is read-only.

## Representation and API

A bare tuple would leave callers to maintain wavelength/flux correspondence;
a dict would add mutable, unchecked structure. A public spectral-grid class
with its own interpolation methods would duplicate an already sufficient API.
The chosen result is the private, frozen `_SpectralLibrary` carrier, returned
by public loader/preparer functions. Its constructor is not exported and is
not a promised user construction API.

Numerical attributes are `.grid` (`RectilinearGrid`) and `.wavelength`
(`region, pixel`, Angstrom). Metadata is `.regions`, `.library`,
`.wavelength_medium`, `.flux_kind`, `.sources`, and `.wavelength_unit`.
The carrier is a PyTree: grid arrays and wavelengths are dynamic leaves;
labels and declarations are immutable metadata. Wavelength is stored once per
payload element, never per atmosphere node. It has no interpolation methods.

```python
from jaxstar.specfit import load_spectral_grid

library = load_spectral_grid("prepared.npz")
flux = library.grid.interpolate(
    {"teff": 5812.5, "logg": 4.2, "feh": -0.15, "alpha": 0.12}
)["flux"]
# flux.shape == library.wavelength.shape == (n_region, n_pixel)
```

The ten exported functions are:

```text
save_spectral_grid(path, spectra, *, overwrite=False)
load_spectral_grid(path)
resample_spectral_grid(spectra, *, sampling="log", pixels=None, velocity_step=None)
is_log_uniform(spectra, *, rtol=1e-5)

load_coelho / load_bosz / load_tlusty(
    paths, *, pixel_slices=None, regions=None,
    wavelength_medium="unknown", flux_kind="as_stored")

prepare_coelho(data_dir, *, axes, wavelength_ranges, pixels=None,
    regions=None, wavelength_medium="vacuum", normalized=True,
    source_overrides=None, sampling=None, velocity_step=None)
prepare_bosz(data_dir, *, axes, wavelength_ranges, pixels=None,
    regions=None, wavelength_medium="vacuum", normalized=True,
    source_overrides=None, sampling=None, velocity_step=None)
prepare_tlusty(data_dir, *, wavelength_ranges, axes=None, pixels=None,
    regions=None, flux_kind="normalized", wavelength_medium="unknown",
    source_overrides=None, sampling=None, velocity_step=None)
```

For the legacy adapters, `paths` accepts a single jaxspec NPZ path or an ordered
sequence. `pixel_slices` is one slice applied to all paths, or one per path.
Repeated paths and region identifiers are supported, including two windows of
one order. No directory selection based on observed wavelengths, fitting
margins or masks is included.

## Preparation and persistence boundary

The initial Milestone 2 implementation returned in-memory `_SpectralLibrary`
results from `prepare_coelho`, `prepare_bosz` and `prepare_tlusty`. Those results
contained complete grids and metadata, but there was no supported persistence
API: a new process could not load them through the legacy loaders. This gap is
now closed within Milestone 2, rather than deferred to a later milestone.

```python
from jaxstar.specfit import prepare_bosz, save_spectral_grid, load_spectral_grid

# Offline, once: node reading, medium conversion, continuum division and
# log-uniform model sampling belong to this library-specific preparation step.
# These two ranges have the same log span and can share one rectangular payload.
prepared = prepare_bosz(
    raw_directory, axes=atmosphere_axes,
    wavelength_ranges=[(5000, 5100), (6000, 6120)],
    sampling="log", velocity_step=1.0,  # km/s; illustrative, not a default
    regions=[8, 9], wavelength_medium="vacuum", normalized=True,
)
save_spectral_grid("prepared.npz", prepared)

# Later, in a separate process: no raw directory or library choice is needed.
spectra = load_spectral_grid("prepared.npz")
```

Preparation still returns the same private carrier; it does not implicitly
write files. `save_spectral_grid` writes that carrier and returns the exact
`Path` used, without adding a filename suffix. It requires `overwrite=True`
to replace an existing artifact. There is no cache discovery or storage backend
abstraction. `load_spectral_grid` reads only the artifact and constructs a fresh
carrier/grid; raw source paths are provenance strings, never opened or checked.
Neither saving nor runtime loading changes flux/wavelength samples or performs
normalization, conversion or resampling.

The common format is an uncompressed NPZ with these entries:

| Entry | Meaning |
| --- | --- |
| `metadata` | Scalar Unicode JSON; format `jaxstar.spectral_grid`, version `1` |
| `axis_0`, …, `axis_N-1` | Axis coordinate arrays in declared grid order |
| `flux` | Physical dimensions declared by metadata, followed by `(region, pixel)` |
| `wavelength` | `(region, pixel)` samples, once per payload element, in Angstrom |
| `fill_value` | Scalar numerical grid boundary fill |

Metadata stores `axis_names`, `axis_kinds`, `flux_dims`, `payload_dims`,
`boundary`, `regions`, `library`, `wavelength_medium`, `wavelength_unit`,
`flux_kind` and `sources`, in addition to the format/version marker. Region
labels preserve string/integer types and ordering; duplicates remain valid.
Grid order and field dimension order are recorded separately, so no transpose
or library-specific shape inference is required. Effective axis kinds and the
boundary fill are retained to preserve interpolation behavior.

No Python objects/pickle are stored; loading uses `allow_pickle=False` and
rejects unknown format versions or inconsistent arrays/metadata. Uncompressed
NPZ keeps the persistence step simple without adding compression CPU work.
Array dtypes are saved as currently held by the carrier; loading follows the
normal JAX precision setting. Float64 round trips require x64 in both processes;
a writer cannot restore precision already discarded during preparation/loading.

The schema has no library registry, fixed atmosphere rank, or Coelho/BOSZ/TLUSTY
axis-key requirements. `library` is only a provenance label. A future synthesis
preparer, including Korg, can emit the same named rectilinear flux representation
and call this writer without changing the runtime loader. Korg itself is not
implemented. Producers use the existing Angstrom and flux-product metadata
contract; no raw inputs or preparation settings are needed to evaluate the
saved grid.

Legacy jaxspec parsing remains separate in `libraries.py`: `load_coelho`,
`load_bosz` and `load_tlusty` accept legacy formats only, not the new format.
They can be used once to migrate an old file via `save_spectral_grid`; callers
must declare medium/product when the legacy artifact lacks that metadata.
Common runtime loading lives in `storage.py` and does not dispatch on library.

## Native schemas and frozen preparation

Every library has one field named `flux`, with physical `dims` in the order
below and `payload_dims=("region", "pixel")`. Raw/NPZ input flux has physical
axes followed by pixel; preparation/loading packs region immediately before
pixel. Queries return `query_shape + (region, pixel)` through the core.

| Library | Native NPZ axis keys | Physical dimensions and meanings |
| --- | --- | --- |
| Coelho | `tgrid,ggrid,fgrid,agrid` | `teff,logg,feh,alpha`; alpha enhancement |
| BOSZ | `tgrid,ggrid,mgrid,agrid,cgrid,vgrid` | `teff,logg,mh,alpha,carbon,vmic`; [M/H], [alpha/M], [C/M], microturbulence |
| TLUSTY | `tgrid,ggrid,zgrid` | `teff,logg,logZ`; log10(Z/Z_sun) |

All legacy prepared files also contain `wavgrid` and `flux`. Legacy loaders
preserve stored samples, flux values and floating precision, subject to normal JAX dtype
canonicalization. They never normalize or convert wavelengths. The old NPZ
format does not reliably identify medium or flux product; filenames are not
used to guess either. Callers may declare them. Enable JAX x64 before loading
when float64 must remain float64; loaders do not change global configuration.

Frozen supplied samples have these axes:

- Coelho: teff 3500–7000 by 250; logg 1–5 by 1; feh
  `[-1,-0.5,0,0.5]`; alpha `[0,0.4]`; 5000 pixels; float64 flux.
- BOSZ: teff 5000–7000 by 250; logg 3.5–5 by 0.5; mh
  -0.75–0.5 by 0.25; alpha/carbon `[-0.25,0.25]`; vmic `[0,2,4]`;
  4000 pixels; float32 flux.
- TLUSTY: teff 15000–30000 by 1000 then 32500–55000 by 2500;
  logg 3–4.75 by 0.25; Z/Z_sun `[0.1,0.2,0.5,1,2]` represented as
  log10; 30000 pixels; float32 flux.

Frozen raw preparers differ from sample provenance: they all save float32.
The new raw preparers retain that output precision. Coelho's example uses
teff 3500–7000 by 250, logg `[3,4,5]`, feh `[-1,-0.5,0,0.5]`, alpha `[0,0.4]`;
its function requires a model-parameter table. Frozen BOSZ defaults are the
sample axes listed above. New Coelho/BOSZ preparation requires explicit axes
to avoid silently choosing a large atmosphere grid. TLUSTY defaults to discovery.

Raw-file behavior is accounted for as follows:

- Coelho reads `{teff}_{10logg:02d}_{p/m}{10abs(feh):02d}p{10alpha:02d}.ms.fits`.
  The primary HDU has normalized row 0 and unnormalized row 1, with wavelength
  from `CRVAL1`/`CD1_1`. Air output retains samples; vacuum output uses the
  frozen Morton conversion, without subsequently requiring equal spacing.
- BOSZ reads gzip three-column `wavelength,H,continuum` files under signed
  metallicity directories. Filenames retain 2024 `mp` for logg > 3 and `ms`
  otherwise, signed abundance tokens and vmic. Normalized output is
  `H/continuum`; unnormalized output retains H. The documented native medium
  is vacuum below 2000 Angstrom and air above it. New vacuum preparation only
  converts the air portion; UV regions requested in air are rejected rather
  than mislabeled. This corrects the old unconditional conversion.
- TLUSTY matches spectrum `.7.gz` and continuum `.17.gz` by basename in
  `Z*_vt2`/`Z*_ostar` directories, rather than zipping independently listed
  files. It derives temperature/gravity from names, metallicity from directory
  Z, skips OSTAR temperatures <=30000 and retains logg >2.9 by default.
  It supplies normalized (continuum division), `median_scaled` (the old
  `absolute=True`, which is not calibrated absolute flux), and unnormalized
  native products. Median scaling is per selected region before preparation.
  No new microturbulence axis or medium conversion is inferred from files.

## Wavelength regions and preparation choices

Nonuniform atmosphere axes are passed straight to `RectilinearGrid`.
Nonuniform wavelength samples are metadata and are equally valid. No spacing
check, wavelength-as-axis interpolation, or mandatory equal-spacing step is
added to loaders.

When `sampling` is omitted, raw preparation uses open output-medium wavelength
bounds, as frozen code does.
With `pixels=None`, selected native wavelength samples must match across nodes
and regions must have a common pixel count. With an explicit integer `pixels`,
the preparer uses each region's common node-coverage intersection and linear
host interpolation to prepare that many samples. This is offline library
preparation, not resampling a model onto observations. The old preparers always
performed this step (defaults 5000/4000/30000); it is now optional.

Prepared regions must share atmosphere coordinates, pixel count and flux dtype.
Different-length regions or regions on different atmosphere grids can be loaded
separately into multiple carriers. No padding or ragged-array machinery is
needed for the supplied samples or the frozen rectangular usage. The same
constraint holds for native raw preparation when no explicit sampling is chosen.

## Milestone 2.5: log-uniform fitting-grid preparation

The future fitting contract is log-uniform wavelength within each region.
Preparing this offline removes native/linear-to-log resampling from each future
model evaluation. Common storage itself is not restricted to fitting grids.

The canonical fitting-grid control for all three raw preparers and the common
converter is `sampling="log", velocity_step=dv`. Supply a positive finite dv
in km/s explicitly, as a scalar for the same spacing in every region or as one
value per region. The exact numerical convention is
`dlnlambda=velocity_step/299792.458`, using natural logs. Samples start at the
lower bound and retain all complete steps inside the upper bound, leaving less
than one step unused at the upper edge (apart from floating roundoff).

`velocity_step` describes numerical model wavelength sampling. It is independent
of instrumental resolving power and observed detector pixel spacing; no control
is inferred from observations. No `R`, `resolving_power` or `sampling_resolution`
alias is introduced, and no default dv is chosen. The roughly 4000 model samples
of historical IRD/BOSZ orders corresponded to a numerical sampling resolution
of order `3e5`, or about 1 km/s. This is a useful reference, not a universally
optimal value or package default. Milestone 3 can validate accuracy/performance
choices for the intended model calculations.

```text
requested fitting wavelength range
    -> choose numerical model velocity sampling (e.g. dv ~ 1 km/s)
    -> prepare log-uniform spectral grid offline
    -> save common spectral-grid artifact
    -> future SpecModel
```

`pixels=N` remains a secondary option for compatibility, tests and special-purpose
count-controlled preparation. It includes each region's bounds and gives
`dlnlambda=ln(upper/lower)/(N-1)`, which may differ between regions. It should not
be chosen simply to match the number of observed detector pixels. Exactly one
of `velocity_step` or `pixels` is required for log sampling; specifying both is
rejected. Counts must be integers >= 2.

Fixed dv changes sample counts with the requested log span, not wavelength
location. The existing rectangular payload requires common counts. If regions
yield different counts, prepare them separately to retain the same dv. Explicit
per-region dv or the special-purpose `pixels` option can instead make counts
equal when that is the intended numerical choice. No padding, silent change to
dv or truncation of longer regions is added. Sampling must provide at least two
distinct wavelengths per region at the stored dtype precision.

Log raw preparation constructs the requested wavelength samples once, then
reads each atmosphere node once and linearly interpolates its native flux
directly onto those samples. There is no intermediate linear prepared grid.
Each node must cover the full requested range, after the existing medium
conversion. Bracketing source points outside the requested range allow accurate
edge interpolation; missing coverage raises an error rather than shrinking the
request or extrapolating. Native normalized/unnormalized products remain as in
Milestone 2. TLUSTY median scaling uses only the old strict-interior native
samples; added bracketing points do not change that scale.

For backwards compatibility, omitted `sampling` keeps M2's `pixels=None` native
and explicit `pixels` linear preparation. Explicit `sampling="native"` accepts
neither count nor velocity; explicit `sampling="linear"` requires a count.
The previous linear common-coverage intersection remains unchanged. Log fitting
preparation is explicit, rather than changing existing users' saved data.

Existing legacy or common prepared grids can be converted without raw files:

```python
legacy = load_bosz("legacy_order8.npz", regions=[8], wavelength_medium="vacuum",
                   flux_kind="normalized")
fitting = resample_spectral_grid(legacy, sampling="log", velocity_step=1.0)
assert is_log_uniform(fitting)
save_spectral_grid("bosz_log.npz", fitting)
# Later process: load_spectral_grid("bosz_log.npz")
```

The converter is library-agnostic and applies the same sampling choices over
the stored coverage. It performs one linear point interpolation in wavelength,
preserving all atmosphere axes/kinds, field order/flux dtype, region identifiers,
medium/product/provenance and grid boundary settings. No extra normalization,
flux scaling, flux-conserving rebinning or higher-order interpolation is added.
Conversion is an offline operation, not observed-wavelength evaluation.

Generated target wavelengths use the effective JAX dtype before flux
interpolation, so spectra are evaluated at the samples actually stored.
Conversion preserves wavelength dtype subject to canonicalization; raw
preparation retains M2's float32 flux and configured wavelength precision.
Endpoints may move inward by floating roundoff to stay inside source coverage.
Sampling too fine to produce distinct representable wavelengths fails clearly.
Callers choose sufficiently wide requested bounds; no future RV, vsini,
macroturbulence or instrumental padding is inferred here.

`is_log_uniform` is a setup-time host check over all regions, requiring at least
two samples per region. For each row it compares natural logs against the affine
grid defined by its endpoints. Maximum residual must be at most
`rtol*dlnlambda + 2*eps(wavelength_dtype) + 8*eps(float64)*max(abs(lnlambda))`,
with default `rtol=1e-5`. This bounds global departure instead of accumulating
roundoff per pixel. Float32 quantization is accounted for; very short linear and
log ranges can be numerically indistinguishable at that precision. M3 can use
this check once at setup and reject unsuitable fitting grids.

Storage schema/version and carrier metadata are unchanged. Wavelength arrays
retain everything needed to check sampling and infer each region's effective
`dv=c*dlnlambda`; no sampling label or source-specific storage field is needed.
Arbitrary valid wavelength arrays still round-trip as version-1 common NPZs.

Frozen `SpecModel` built `logspace(log10(wavmin),log10(wavmax),Nwav)[1:-1]`.
Thus its working count was `Nwav-2`, with step
`ln(wavmax/wavmin)/(Nwav-1)`; `varr_for_kernels` used `c*dlnlambda`. Our full
endpoint/count mode reproduces that untrimmed grid, and matching its trimmed
bounds plus `pixels=Nwav-2` reproduces the working samples within floating
precision. We deliberately do not discard endpoints automatically or impose
the old dependence on observed arrays. No forward model or broadening code is
implemented in this milestone.

## Intentional differences and later migration items

The core's inclusive endpoints, exact singleton coordinates and constant `-inf`
out-of-domain fill govern all libraries. BOSZ nearest clamping, TLUSTY's
parameter clamping, and old wavelength/upper-node NaNs are not reimplemented.
The missing-Coelho-alpha case is supported only when flux unambiguously has
one alpha node, represented as alpha=0; an invented alpha range is not inferred.

Raw missing models fail by default. Explicit `source_overrides` retain the
ability to make deliberate replacements: Coelho/BOSZ map target atmosphere
tuples to file paths; TLUSTY maps them to (spectrum, continuum) pairs. Relative
paths are resolved below data_dir and actual source paths are recorded.
This replaces Coelho's three hard-coded alpha-file substitutions (4250/5/0.5/0,
4750/*/0.5/0.4, 5250/3/0.2/0), BOSZ's broad-exception logg=4 fallback and
TLUSTY's nearest-logg copying. Existing prepared NPZ values remain untouched;
their past substitutions cannot be recovered without provenance.

Useful preparation still to migrate explicitly:

- iSpec synthesis/range generation (empirical vmic, MARCS/ATLAS, line lists,
  abundances, process count, Turbospectrum/SPECTRUM, 400000 resolving power).
- iSpec FITS/parameters.tsv conversion: nm→Angstrom, edge removal, optional
  air→vacuum conversion and packing the Coelho-like axis schema. Do not
  mislabel iSpec data as physical Coelho models.
- Observed-order/coverage association and fitting margins belong to a later
  consumer; spectral forward-model resampling remains Milestone 3.

Before Milestone 3, consumers must use the declared medium/product and preserve
region identity. Missing declarations in old NPZs and historical substitutions
need explicit user/library provenance rather than guesses. Variable-length or
different-axis regions would require multiple carriers, if that actual need
arises. No higher-level architecture was frozen by this milestone.

## Validation

Small generated unit fixtures exercise prepared schemas, shapes, nodes,
nonuniform samples, dtype, scalar/batch queries, JIT/gradients/PyTrees, native
FITS/gzip naming, continuum and intrinsic products, explicit substitutions,
native versus optional preparation sampling, and malformed inputs.

Persistence tests prepare tiny raw fixtures for all three libraries and all
supported flux products, save a common artifact, move the raw directory out of
its original location, then load and interpolate in a fresh Python process.
They compare exact axes/wavelength/flux, dtypes, mixed-type region identifiers,
interpretation/provenance metadata and JIT interpolation results. Additional
tests cover legacy-to-common conversion, arbitrary producer/axis names, distinct
field/grid dimension orders, axis kinds, boundary fill, version/schema rejection
and JAX dtype canonicalization.

Milestone 2 baseline, verified on 2026-10-02: the full default suite passed with `213 passed,
13 skipped, 2 xfailed`; the separate frozen-reference suite passed all 12 tests.
The default skips are the 12 opt-in spectral reference cases and one full-MIST
case; the two existing legacy xfails are unchanged.

Milestone 2.5 final verification on 2026-10-02: the full suite with frozen
references enabled passed with `280 passed, 1 skipped, 2 xfailed`. The one skip
requires a full MIST grid; the existing legacy xfails remain unchanged. There
are 55 added cases covering direct log raw preparation/products, count/velocity
controls, varying native node samples, scalar/batch/gradient queries, dtypes,
legacy conversion and fresh-process loading, arbitrary sampling storage,
validation, frozen logspace equivalence and an endpoint-roundoff regression.
The canonical velocity API is exercised with one scalar dv across raw regions,
the same spacing over different wavelength intervals, float32/float64 round trips
and legacy conversion followed by fresh-process loading. Positive finite dv and
mutual exclusion with count control are tested. The median-scaled flux gradient
oracle uses float64 differences and a float32 reduction-roundoff allowance.
Smooth and line-like analytic spectra agree within absolute flux errors of
`2e-8` and `1e-5`, respectively, for the tested dense native sampling.

Validation TODO (not a blocker for merging Milestone 2 or 2.5): full integration/smoke
testing against the actual raw Coelho, BOSZ and TLUSTY libraries remains to be
performed on a machine where those raw libraries are available. Exercise raw
preparation, common artifact saving, loading in a fresh process and representative
interpolation. The small raw fixtures and frozen prepared NPZ comparisons do
not replace this check of the actual raw libraries. No new test infrastructure
is introduced for this later validation item.

The opt-in read-only suite runs with `JAXSPEC_REFERENCE_ROOT=/path/to/jaxspec`.
It retains the frozen two 64-pixel windows per library (2100/4100, 1680/3280,
9940/6860), checks source/input hashes, and compares the saved spectra/batches/
scaled atmosphere gradients. Test-only linear pixel algebra evaluates the same
wavelengths as the oracle. No source module is imported and no reference is
regenerated. Default tests use no sibling files or large copied fixtures.

Preparation streams nodes into one packed host buffer; optional common-coverage
sampling uses a bounds-only first pass rather than retaining every raw spectrum.
Prepared loading fills one buffer per library, retaining only the current
source field. RectilinearGrid's existing constructor copy/canonicalization is
unchanged. Wavelength memory scales with region×pixel; compiled calls receive
large flux arrays as ordinary PyTree leaves. No GPU claim or kernel change is
made here.
