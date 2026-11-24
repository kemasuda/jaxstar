"""Crop wavelength payloads on the host before creating per-line JAX grids."""

import numpy as np

from .sampling import _C_KMS, _sampling_mode, _target_wavelengths


def _crop_payload(wave, flux, indices, request, *, pixels=None, velocity_step=None):
    observed, _, margin = request
    bounds = observed[:, [0, -1]].astype(float) + np.array([-margin, margin])
    source = wave[indices]
    spans = [(max(0, np.searchsorted(row, lo, side="right") - 1),
              min(len(row), np.searchsorted(row, hi, side="left") + 1))
             for row, (lo, hi) in zip(source, bounds)]
    if pixels is None and velocity_step is None:
        # Equal-length native slices preserve log spacing without inventing
        # continuum values. Shorter windows receive extra real grid samples.
        count = max(max(end - start, 2) for start, end in spans)
        starts = [max(0, min(start - (count - (end - start)) // 2, wave.shape[1] - count))
                  for start, end in spans]
        return (np.stack([wave[i, start:start + count] for i, start in zip(indices, starts)]),
                np.stack([flux[..., i, start:start + count] for i, start in zip(indices, starts)], axis=-2))
    _sampling_mode("log", pixels, velocity_step)
    if velocity_step is not None:
        dv = np.asarray(velocity_step)
        if (dv.dtype.kind not in "fiu" or dv.shape not in ((), (len(indices),))
                or not np.all(np.isfinite(dv) & (dv > 0))):
            raise ValueError("velocity_step must be positive finite scalar or one value per region")
        step = np.broadcast_to(dv, (len(indices),)) / _C_KMS
        logs = np.log(bounds)
        intervals = np.ceil((logs[:, 1] - logs[:, 0]) / step)
        if np.any(~np.isfinite(intervals)) or np.max(intervals) >= np.iinfo(np.intp).max - 2:
            raise ValueError("velocity_step requests too many samples")
        # One extra interval protects coverage from rounding at endpoints.
        span = (int(np.max(intervals)) + 1) * step
        available = np.log(source[:, [0, -1]])
        if np.any(span > available[:, 1] - available[:, 0]):
            raise ValueError("common cropped pixel count exceeds source coverage; use pixels or group similar windows")
        lo = np.clip(np.mean(logs, axis=1) - span / 2,
                     available[:, 0], available[:, 1] - span)
        bounds = np.exp(np.column_stack((lo, lo + span)))
        # Avoid even ULP extrapolation at a source endpoint.
        bounds[:, 0] = np.maximum(bounds[:, 0], source[:, 0])
        bounds[:, 1] = np.minimum(bounds[:, 1], source[:, -1])
    target = _target_wavelengths(bounds, "log", pixels, velocity_step, dtype=wave.dtype)
    output = np.empty(flux.shape[:-2] + target.shape, dtype=flux.dtype)
    for r, i in enumerate(indices):
        lo = max(0, np.searchsorted(wave[i], target[r, 0], side="right") - 1)
        hi = min(wave.shape[1], np.searchsorted(wave[i], target[r, -1], side="left") + 1)
        source_flux = flux[..., i, lo:hi].reshape(-1, hi - lo)
        destination = np.empty((len(source_flux), target.shape[1]), dtype=flux.dtype)
        for n, row in enumerate(source_flux):
            destination[n] = np.interp(target[r], wave[i, lo:hi], row)
        output[..., r, :] = destination.reshape(output.shape[:-2] + (target.shape[1],))
    return target, output
