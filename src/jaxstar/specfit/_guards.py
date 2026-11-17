"""Fixed trusted-pixel selections, resolved before inference."""

import numpy as np

from .sampling import _C_KMS


MG_CENTERS_AIR = (5167.322, 5172.684, 5183.604)
MG_HALF_WIDTH = 1.


def guard_array(value, shape, *, name="custom_guard"):
    if value is None:
        return None
    array = np.array(value, copy=True)
    if array.shape != shape or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    array.setflags(write=False)
    return array


def guard_velocity(value):
    array = np.asarray(value)
    if (array.ndim != 0 or array.dtype.kind not in "fiu"
            or not np.isfinite(array) or abs(array) >= _C_KMS):
        raise ValueError("guard_rv must be a finite scalar in km/s with abs(guard_rv) < c")
    return float(array)


def magnesium_guard(observation, medium, rv):
    """The original air ±1 Å windows, in the initial relativistic RV frame."""
    if medium not in {"air", "vacuum"}:
        raise ValueError("mg_guard requires wavelength_medium='air' or 'vacuum'")
    beta = guard_velocity(rv) / _C_KMS
    wave = np.asarray(observation.wavelength) / np.sqrt((1 + beta) / (1 - beta))
    centers = np.asarray(MG_CENTERS_AIR)
    if medium == "air":
        # Match the manually configured GAOES experiments exactly.
        guard = np.any(abs(wave[..., None] - centers) <= MG_HALF_WIDTH, axis=-1)
    else:
        from .libraries import _air_to_vacuum

        low = _air_to_vacuum(centers - MG_HALF_WIDTH)
        high = _air_to_vacuum(centers + MG_HALF_WIDTH)
        guard = np.any((wave[..., None] >= low) & (wave[..., None] <= high), axis=-1)
    return guard & ~np.asarray(observation.mask)
