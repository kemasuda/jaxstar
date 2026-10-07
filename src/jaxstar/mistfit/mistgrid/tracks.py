"""Load complete evolutionary-track NPZ grids, with precomputed photometry.

This reads the schema produced by benchmarks/build_mist_track_extension.py.
It does not download tracks or build bolometric corrections on first use.
"""

from pathlib import Path

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


def load_tracks(path):
    """Read an explicit NPZ once; require x64 and a complete rectangular grid.

    All scalar native fields are retained. Named photometric bands are split
    into scalar absolute-magnitude fields. dtdeep is recomputed from native
    age_gyr on the full EEP axis. Missing cells/partial tracks are not supported.
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
        if magnitudes.shape != (*shape, len(bands)) or not np.all(np.isfinite(magnitudes)):
            raise ValueError("Incomplete or nonfinite absolute_magnitudes")
        excluded = {*TRACK_AXES, "absolute_magnitudes", "bands"}
        native = {name: np.array(data[name], dtype=np.float64) for name in data.files if name not in excluded}
    for name, values in native.items():
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Incomplete or nonfinite native field: {name}")
    if not np.allclose(10 ** (native["logage"] - 9), native["age_gyr"], rtol=2e-13, atol=0):
        raise ValueError("Native physical ages and logage disagree")
    native["dtdeep"] = centered_age_derivative(native["age_gyr"], axes["eep"])
    for i, band in enumerate(bands):
        name = _BAND_KEYS[band]
        if name in native:
            raise ValueError(f"Duplicate scalar and band field: {name}")
        native[name] = magnitudes[..., i]
    grid = RectilinearGrid(axes=axes,
        fields={name: Field(values, TRACK_AXES) for name, values in native.items()},
        fill_value=np.nan)
    return _MistGridData(grid, kind="track", sources=(str(path),))
