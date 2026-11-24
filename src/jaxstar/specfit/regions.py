"""Host-side extraction of native observed pixels and their source mapping."""

from dataclasses import dataclass

import numpy as np

from .observation import Observation, _array
from ._selection import _covering_indices, _wavelength_request


@dataclass(frozen=True, eq=False)
class RegionMapping:
    """Source rows and native pixel indices for an extracted Observation.

    Arrays are read-only. ``source_indices`` indexes the original rows (also
    useful for per-order IP parameters); ``source_regions`` retains their labels.
    ``pixel_indices`` is -1 at rectangular padding. ``take`` remaps an original
    observation-shaped array, e.g. ``mapping.take(sf.fit_mask, fill_value=True)``.
    """

    source_indices: object
    source_regions: tuple
    pixel_indices: object
    source_shape: tuple

    def __post_init__(self):
        for name in ("source_indices", "pixel_indices"):
            object.__setattr__(self, name, _array(getattr(self, name)))

    def take(self, values, *, fill_value):
        values = np.asarray(values)
        if values.shape != self.source_shape:
            raise ValueError("values must have the original Observation.shape")
        selected = np.atleast_2d(values)[self.source_indices[:, None],
                                        np.maximum(self.pixel_indices, 0)]
        return np.where(self.pixel_indices >= 0, selected, fill_value)


def extract_regions(observation, *, wavelength_ranges, regions=None, source_regions=None):
    """Extract line windows without resampling measured pixels.

    Ranges are finite increasing [lower, upper] pairs in the observation's
    wavelength frame/unit/medium, including both endpoints. A range must fit
    wholly inside exactly one source row. ``source_regions`` selects original
    labels explicitly when orders overlap; for an unlabeled Observation the
    labels are zero-based row indices. ``regions`` labels the new line windows
    (default: zero-based line indices), independently of source-order labels.

    Returns ``(Observation, RegionMapping)``. Output is always 2D. All selected
    fluxes, errors and original masks are retained. Unequal pixel counts are
    padded to the longest window using adjacent native wavelengths within the
    same source row; padding has NaN flux/error, mask=True and pixel index -1.
    No wavelength is fabricated, and the measured arrays are never modified.
    Model coverage must include these masked wavelengths too. Model-grid
    padding for RV/convolution is separate from these observation windows.
    """
    if not isinstance(observation, Observation):
        raise TypeError("extract_regions requires an Observation")
    bounds = np.asarray(wavelength_ranges)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds):
        raise ValueError("wavelength_ranges must have shape (n_region, 2)")
    bounds, labels, margin = _wavelength_request(bounds, regions, 0.)
    wave = np.atleast_2d(np.asarray(observation.wavelength))
    source_labels = observation.region or tuple(range(observation.n_regions))
    candidates = [(str(label), row[0], row[-1]) for label, row in zip(source_labels, wave)]
    indices = _covering_indices(bounds, labels, margin, candidates,
                               source_regions=source_regions, candidate_regions=source_labels)
    intervals = [(np.searchsorted(wave[i], lo, side="left"),
                  np.searchsorted(wave[i], hi, side="right"))
                 for i, (lo, hi) in zip(indices, bounds)]
    if any(end == start for start, end in intervals):
        raise ValueError("each wavelength range must contain at least one observed pixel")
    count = max(end - start for start, end in intervals)
    waves, pixels = [], []
    for i, (start, end) in zip(indices, intervals):
        left = max(0, min(start - (count - (end - start)) // 2, wave.shape[1] - count))
        columns = np.arange(left, left + count)
        waves.append(wave[i, columns])
        pixels.append(np.where((columns >= start) & (columns < end), columns, -1))
    mapping = RegionMapping(np.asarray(indices), tuple(source_labels[i] for i in indices),
                            np.asarray(pixels), observation.shape)
    extracted = Observation(np.asarray(waves), mapping.take(observation.flux, fill_value=np.nan),
                            mapping.take(observation.uncertainty, fill_value=np.nan),
                            mapping.take(observation.mask, fill_value=True),
                            region=labels, exposure=observation.exposure)
    return extracted, mapping
