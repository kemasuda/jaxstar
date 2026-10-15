"""Load evolutionary-track NPZ grids, with precomputed photometry.

This reads the schema produced by create_tracks.create_track_grid and the
earlier benchmarks/build_mist_track_extension.py.
It does not download tracks or build bolometric corrections on first use.
"""

from pathlib import Path
import json

import jax
import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from .._data import _MistGridData


TRACK_AXES = ("initial_mass", "initial_feh", "eep")
_BAND_KEYS = dict(zip(
    ("Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I",
     "2MASS_J", "2MASS_H", "2MASS_Ks", "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev",
     "Gaia_RP_DR2Rev", "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"),
    ("umag", "bmag", "vmag", "rmag", "imag", "jmag", "hmag", "kmag",
     "gmag2", "bpmag2", "rpmag2", "gmag3", "bpmag3", "rpmag3"),
))
_REQUIRED_FIELDS = ("logage", "age_gyr", "logTeff", "logg", "logL", "logR",
                    "star_mass", "cmd_surface_feh")


def centered_age_derivative(age_gyr, eep):
    """Central dt/dEEP (Gyr/EEP), copying the adjacent interior endpoints.

    Compute on the full native uniform EEP axis before inference bounds are
    applied. This estimates the smooth relation independently of the linear
    logage interpolant, following jaxstar's stored dmdeep convention.
    """
    age_gyr, eep = np.asarray(age_gyr, dtype=np.float64), np.asarray(eep, dtype=np.float64)
    if eep.ndim != 1 or len(eep) < 3 or age_gyr.ndim == 0 or age_gyr.shape[-1] != len(eep):
        raise ValueError("At least three EEPs matching the final age dimension are required")
    spacing = np.diff(eep)
    if (not np.all(np.isfinite(eep)) or not np.all(spacing > 0)
            or not np.allclose(spacing, spacing[0], rtol=1e-12, atol=0)):
        raise ValueError("Centered age differences require uniform EEP spacing")
    if not np.all(np.isfinite(age_gyr) & (age_gyr > 0)) or not np.all(np.diff(age_gyr, axis=-1) > 0):
        raise ValueError("Native ages must be finite, positive, and increasing in EEP")
    middle = (age_gyr[..., 2:] - age_gyr[..., :-2]) / (2 * spacing[0])
    return np.concatenate((middle[..., :1], middle, middle[..., -1:]), axis=-1)


def masked_age_derivative(age_gyr, eep, present):
    """Centered differences within native segments, never across missing EEPs.

    A center needs two positive adjacent age increments. Copy its derivative
    to a segment endpoint only when that endpoint has a full three-node
    stencil. Invalid ages/increments inside a native segment remain holes.
    """
    # Validate spacing through the complete-grid routine, using a dummy curve.
    centered_age_derivative(np.arange(1., len(eep)+1), eep)
    result = np.full_like(age_gyr, np.nan, dtype=np.float64)
    step = np.diff(eep)[0]
    for age, exists, out in zip(age_gyr.reshape(-1, len(eep)),
                                present.reshape(-1, len(eep)), result.reshape(-1, len(eep))):
        changes = np.diff(np.r_[False, exists, False].astype(int))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if end-start < 3:
                continue
            a = age[start:end]
            valid = np.isfinite(a) & (a > 0)
            increasing = np.diff(a) > 0
            good = valid[:-2] & valid[1:-1] & valid[2:] & increasing[:-1] & increasing[1:]
            middle = np.where(good, (a[2:]-a[:-2])/(2*step), np.nan)
            out[start+1:end-1] = middle
            out[start], out[end-1] = middle[0], middle[-1]
    return result


def load_tracks(path):
    """Read an explicit NPZ once; require x64.

    All scalar native fields are retained. Named photometric bands are split
    into scalar absolute-magnitude fields. dtdeep is recomputed from native
    age_gyr on the full EEP axis. Primary EEP labels from optional metadata_json
    are retained for plotting, not as an interpolated field. Format 2 uses
    track_present plus NaN field values to
    represent ragged tracks and unsupported photometry. Legacy grids remain
    strict: missing/nonfinite values without this explicit schema are errors.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before loading a track grid")
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        required = {*TRACK_AXES, *_REQUIRED_FIELDS, "absolute_magnitudes", "bands"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"track NPZ is missing fields: {sorted(missing)}")
        axes = {name: np.array(data[name], dtype=np.float64) for name in TRACK_AXES}
        for name, axis in axes.items():
            if (axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis))
                    or not np.all(np.diff(axis) > 0)):
                raise ValueError(f"track axis {name} must be finite, increasing, and have at least two nodes")
        if axes["initial_mass"][0] <= 0:
            raise ValueError("initial masses must be positive")
        shape = tuple(len(axis) for axis in axes.values())
        if data["bands"].ndim != 1:
            raise ValueError("track bands must be a one-dimensional array of names")
        bands = tuple(data["bands"].tolist())
        if len(bands) != len(_BAND_KEYS) or set(bands) != set(_BAND_KEYS):
            raise ValueError("track bands must contain the 14 supported UBVRIplus bands exactly once")
        magnitudes = np.array(data["absolute_magnitudes"], dtype=np.float64)
        metadata = None
        if magnitudes.shape != (*shape, len(bands)):
            raise ValueError("Incomplete or nonfinite absolute_magnitudes")
        if "metadata_json" in data.files:
            if data["metadata_json"].shape != () or data["metadata_json"].dtype.kind != "U":
                raise ValueError("track metadata_json must be a scalar JSON string")
            try:
                metadata = json.loads(data["metadata_json"].item())
            except ValueError as exc:
                raise ValueError("Invalid track metadata_json") from exc
            if (not isinstance(metadata, dict) or metadata.get("kind") != "track"
                    or metadata.get("format_version") not in (1, 2)):
                raise ValueError("Unsupported track metadata format")
        masked = metadata is not None and metadata["format_version"] == 2
        present = None
        if masked:
            if "track_present" not in data.files:
                raise ValueError("Masked track NPZ requires track_present")
            present = np.array(data["track_present"])
            if present.dtype.kind != "b" or present.shape != shape or not present.any():
                raise ValueError("track_present must be a nonempty Boolean mask matching the grid")
        elif "track_present" in data.files:
            raise ValueError("track_present requires track metadata format 2")
        excluded = {*TRACK_AXES, "absolute_magnitudes", "bands", "metadata_json", "track_present"}
        native = {name: np.array(data[name], dtype=np.float64) for name in data.files if name not in excluded}
    for name, values in native.items():
        if values.shape != shape or np.any(np.isinf(values)) or (not masked and np.any(np.isnan(values))):
            raise ValueError(f"Incomplete or nonfinite native field: {name}")
        if masked and np.any(np.isfinite(values[~present])):
            raise ValueError(f"Native field {name} contains values outside track_present")
    if (np.any(np.isinf(magnitudes)) or (not masked and np.any(np.isnan(magnitudes)))
            or (masked and np.any(np.isfinite(magnitudes[~present])))):
        raise ValueError("Incomplete or nonfinite absolute_magnitudes")
    if not np.allclose(10 ** (native["logage"] - 9), native["age_gyr"],
                       rtol=2e-13, atol=0, equal_nan=masked):
        raise ValueError("Native physical ages and logage disagree")
    if masked and np.any(np.isfinite(native["age_gyr"]) & (native["age_gyr"] <= 0)):
        raise ValueError("Native physical ages must be positive or missing")
    native["dtdeep"] = (masked_age_derivative(native["age_gyr"], axes["eep"], present)
                        if masked else centered_age_derivative(native["age_gyr"], axes["eep"]))
    for i, band in enumerate(bands):
        name = _BAND_KEYS[band]
        if name in native:
            raise ValueError(f"Duplicate scalar and band field: {name}")
        native[name] = magnitudes[..., i]
    masks = {name: np.isfinite(values) for name, values in native.items()} if masked else None
    if masked:
        native = {name: np.where(masks[name], values, 0.) for name, values in native.items()}
    grid = RectilinearGrid(axes=axes,
        fields={name: Field(values, TRACK_AXES) for name, values in native.items()},
        fill_value=0. if masked else np.nan)
    try:
        primary = np.asarray(sorted({eep for record in (metadata or {}).get("tracks", ())
                                     for eep in record.get("primary_eeps", ())}), dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid primary EEP metadata") from exc
    if (primary.ndim != 1 or not np.all(np.isfinite(primary))
            or np.any(primary <= 0) or np.any(primary != np.floor(primary))):
        raise ValueError("Primary EEP metadata must contain positive integer labels")
    provenance = {key: metadata[key] for key in (
        "format_version", "mist_version", "rotation", "afe", "photometry", "Av", "Rv",
        "BC_recipe", "chemistry", "Zsun", "Xsun", "Mbol_sun", "implementation_sha256", "inputs",
        "include_structure", "native_columns",
    ) if metadata is not None and key in metadata}
    return _MistGridData(grid, kind="track", sources=(str(path),),
                         validity=None if masks is None else tuple(masks[k] for k in grid.field_names),
                         primary_eeps=tuple(int(eep) for eep in primary),
                         metadata_json=json.dumps(provenance, sort_keys=True, allow_nan=False))
