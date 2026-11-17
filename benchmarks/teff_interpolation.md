# Teff interpolation: implementation and held-out validation

The spectrum path is `SpecModel._atmosphere` → `interpolate_flux` →
`RectilinearGrid.interpolate`. Previously `_atmosphere` called the grid
directly. `linear` preserves that exact call; `cubic` evaluates the existing
multilinear kernel on at most four native Teff slices and interpolates their
flux with local PCHIP. The slopes are calculated **after** interpolating
logg, metallicity and any additional composition/vmic axes. No generic grid
or MIST behavior changes.

Stored flux has shape `(atmosphere axes..., region, pixel)` with explicit
dimension names; field axes may be permuted. Typical schemas are
`teff/logg/feh/alpha` (Coelho/Korg), `teff/logg/mh/alpha/carbon/vmic` (BOSZ),
and `teff/logg/logZ` (TLUSTY). Grid queries broadcast to `B + (region, pixel)`;
the local cubic stencil has shape `B + (4, region, pixel)`. A SpecModel star
has scalar atmosphere coordinates, producing `(region, pixel)` before
broadening, RV, sampling and component combination. Those stages feed the
existing likelihood/NumPyro/SVI/NUTS paths and remain JAX-compatible.

The [PCHIP algorithm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
uses weighted harmonic-mean slopes internally and limited one-sided endpoint
slopes. Its first Teff derivative is continuous; its second derivative and
derivatives with respect to other coordinates need not be. Finite-input
branches avoid reciprocal singularities at flat pixels or extrema. The mode
is static PyTree metadata on SpecModel, defaults to `linear`, and is included
in saved fit metadata. No interpolation dependency is added.

## Reproduce

Use an environment with this checkout's dependencies and `PYTHONPATH=src`.
The script enables x64 and leaves each grid's stored flux dtype unchanged.
It accepts common-format files or legacy Coelho/BOSZ/TLUSTY NPZs. It reads
existing spectra without resynthesis, wavelength resampling or broadening.
Each coordinate supplied below must be a native grid coordinate. All interior
Teff nodes are held out one at a time; the omitted spectrum never enters the
prediction. At fixed native other coordinates, extracting the temperature
line is equivalent to the existing other-axis interpolation.

```bash
PYTHONPATH=src python benchmarks/validate_teff_interpolation.py \
  /path/to/specgrid_gaoes_coelho/5139-5231_normed.npz \
  /path/to/specgrid_gaoes_coelho/5473-5570_normed.npz \
  /path/to/specgrid_gaoes_coelho/5853-5956_normed.npz \
  --format coelho \
  --coordinates '{"logg":[1,3,5],"feh":[-1,0,0.5],"alpha":[0]}' \
  --exclude-nodes /path/to/coelho-excluded-nodes.json \
  --output benchmark-results/teff-interpolation/coelho

PYTHONPATH=src python benchmarks/validate_teff_interpolation.py \
  /path/to/jaxstar-grid_optical_tlusty_vt2/grid.npz \
  --coordinates '{"logg":[3,4,4.75],"logZ":[-1,0,0.3010299956639812]}' \
  --output benchmark-results/teff-interpolation/tlusty

PYTHONPATH=src python -m pytest tests/unittests/test_teff_interpolation.py --run-slow
```

Common-format provenance replacements are excluded from both the held-out
truth and the local training stencil. Legacy grids lack this provenance;
`--exclude-nodes` accepts a JSON list of atmosphere-coordinate dictionaries.
For the Coelho run below the 11 known replacement nodes from the local source
audit were supplied; only `(4250, logg=5, feh=0.5, alpha=0)` affects these
locations. Its five affected holdouts in each of three regions were omitted.
Supplying just this entry reproduces the selected exclusions:

```json
[{"teff":4250,"logg":5,"feh":0.5,"alpha":0}]
```

`cases.csv` contains per-location, per-temperature, per-region RMS, maximum
absolute residual and weighted RMS for both modes. `report.json` records
input SHA256 hashes, dtype, wavelength coverage, skipped cases, coordinates
and summaries by temperature and location. Generated reports remain under
ignored `benchmark-results/`.

Weights are an explicit diagnostic photon-noise proxy for normalized spectra:
`1 / (0.01**2 * max(native_flux, 0.05))`, corresponding to continuum S/N=100
with a flux floor. Weighted RMS is `sqrt(sum(w * residual**2) / sum(w))` in
flux units; these are not observed uncertainties or likelihood scores. Other
flux normalizations require a different noise model.

## Local results, 2026-11-17

Python 3.13 / JAX 0.11.2, CPU, x64 enabled. Coelho uses stored float64 flux,
15 Teff nodes from 3500 to 7000 K, 9 logg/FeH locations and three 5000-pixel
windows covering approximately 5139–5231, 5473–5570 and 5853–5956 Å. There
are 336 accepted cases and 15 exclusions. These are prepared legacy native
node spectra; the old NPZs do not provide full source provenance, so the
known exclusions are not a new exhaustive audit of their original inputs.

Numerical and integration validation passed 215 checks: 214 in the combined
`--run-slow` run of `test_teff_interpolation`, `test_specfit_grid_setup`,
`test_specmodel`, `test_specmodel_sbn`, `test_numpyro_spectral` and
`test_specfit_results`, plus the subsequently added x64-disabled regular-grid
check. This includes both modes' SVI/NUTS smoke runs, scalar projections with
`jax.grad`, native-node reproduction, finite gradients at extrema/flat pixels,
edge/small-grid behavior, broadcasting, field-axis permutations and independent
SciPy PCHIP comparison. It is a focused regression run, not the full repository
test suite.

TLUSTY OB2025 uses stored float32 flux, 26 native temperatures from 15000 to
55000 K, nonuniform Teff spacing, 9 logg/logZ locations and 21726 pixels
covering 3590–7409.9 Å. There are 185 accepted cases and 31 exclusions from
recorded missing-node replacements. The existing library switches BSTAR to
OSTAR at 30000 K; its provenance records different atmosphere vmic values
for those families. This experiment leaves that grid and its physics intact.

| Metric | Coelho linear | Coelho cubic | TLUSTY linear | TLUSTY cubic |
| --- | ---: | ---: | ---: | ---: |
| Mean per-case flux RMS | 0.0130130 | 0.00737336 | 0.00143823 | 0.00104393 |
| Pooled flux RMS | 0.0211552 | 0.0136665 | 0.00198483 | 0.00167972 |
| Mean per-case maximum absolute residual | 0.0571992 | 0.0379707 | 0.0513519 | 0.0424891 |
| Worst absolute residual | 0.327195 | 0.327195 | 0.467216 | 0.392335 |
| Mean per-case weighted RMS | 0.0124373 | 0.00739200 | 0.00157306 | 0.00111949 |

Pooled RMS weights all tested pixels equally within each library; windows
can overlap. Counts below are diagnostic cases, not independent stars or
statistical confidence levels.

- **Coelho:** cubic improves RMS in **329/336 (97.9%)** cases; median
  cubic/linear RMS ratio is **0.390**. All nine atmosphere locations have
  median improvement. All seven worse cases occur at **3750 K**, near the
  lower grid edge. The worst RMS ratio is **1.799**, at logg=3, FeH=0 in the
  5853–5956 Å window (0.02990 → 0.05379). Maximum absolute residual improves
  in 312 cases, worsens in 18 and ties in 6. Weighted RMS improves in 327.
- **TLUSTY:** cubic improves RMS in **179/185 (96.8%)** cases; median ratio
  is **0.629**. All nine locations have median improvement. All six worse
  cases occur at **28000–30000 K**, around the stored family transition.
  The largest relative degradation is **1.179** at 28000 K, logg=4.75,
  logZ=-1. At 30000 K, logg=4, logZ=log10(2), RMS rises from 0.008695 to
  0.009777. Maximum absolute residual improves in 141 cases, worsens in 32
  and ties in 12. Weighted RMS improves in 178.
- Bracketing-spectrum overshoot is zero for Coelho and at most **5.96e-8**
  for TLUSTY (float32 roundoff). Worse prediction can therefore occur even
  without oscillatory overshoot.

Cubic gives broad improvement in these two samples, but is not uniformly
better, particularly near boundaries or changes in the native physics.
`linear` remains the default. This test omits a node and thus widens its local
interval; it does not establish errors at arbitrary sub-grid temperatures,
posterior calibration, performance on every library or inference speed.
No real-star inference settings or fits were changed. BOSZ/Coelho/TLUSTY
schemas and off-node other-coordinate interpolation are also checked against
an independent SciPy PCHIP reference in the synthetic numerical tests.
