"""Setup-time wavelength coverage selection shared by spectral loaders."""

import numpy as np

from ._data import _validate_wavelength
from .observation import _identifiers


def _wavelength_request(wavelength, regions, wav_margin):
    wave = np.asarray(wavelength)
    if wave.dtype.kind not in "fiu":
        raise TypeError("wavelength must contain real numeric values")
    if wave.ndim not in (1, 2):
        raise ValueError("wavelength must have shape (pixel,) or (region, pixel)")
    wave = np.atleast_2d(wave)
    _validate_wavelength(wave, ndim=2)
    labels = _identifiers(regions, "regions", len(wave))
    if labels is None:
        labels = tuple(range(len(wave)))
    margin = np.asarray(wav_margin)
    if (margin.ndim != 0 or margin.dtype.kind not in "fiu"
            or not np.isfinite(margin) or margin < 0):
        raise ValueError("wav_margin must be finite nonnegative Angstrom")
    return wave, labels, float(margin)


def _covering_indices(wave, labels, margin, candidates):
    """Choose one (description, lower, upper) candidate per observation row."""
    selected = []
    for label, row in zip(labels, wave):
        low, high = float(row[0]) - margin, float(row[-1]) + margin
        matches = [i for i, (_, left, right) in enumerate(candidates)
                   if left <= low and right >= high]
        if len(matches) != 1:
            raise ValueError(
                f"Region {label!r}: expected one covering grid, found {len(matches)} "
                f"for [{low:g}, {high:g}] Angstrom including wav_margin={margin:g}. "
                f"Matches: {[candidates[i][0] for i in matches]}. "
                f"Candidates: {candidates}")
        selected.append(matches[0])
    return selected
