"""Offline wavelength preparation and setup-time fitting-grid validation."""

from dataclasses import replace

import jax
import numpy as np

from jaxstar.grid import Axis, Field, RectilinearGrid
from ._data import _SpectralLibrary, _validate_wavelength


_C_KMS = 299792.458  # Frozen jaxspec's numerical velocity-grid convention.


def resample_spectral_grid(spectra, *, sampling="log", pixels=None, velocity_step=None):
    """Resample a common spectral library once, offline, retaining its metadata.

    For fitting use ``sampling='log', velocity_step=dv`` with an explicit positive
    finite dv in km/s (scalar or one per region). This is numerical model sampling,
    independent of instrumental resolving power and observed detector pixels.
    The convention is dlnlambda = velocity_step / 299792.458, with natural logs;
    it does not define RV physics. Fixed-step grids start at each stored lower
    bound and stop at the last sample within its upper bound. Regions must yield
    a common pixel count; otherwise resample them separately to retain dv.

    ``pixels`` remains an alternative for compatibility/tests/special-purpose
    count-controlled preparation, including both endpoints. It is mutually
    exclusive with ``velocity_step`` and is not tied to observed pixel counts.
    Neither control is inferred and there is no default log velocity spacing.

    Linear sampling with an explicit ``pixels`` count is also supported. Flux
    uses piecewise-linear point interpolation in wavelength, without extra
    normalization/scaling. Atmosphere axes, field order/dtype, boundary policy,
    region identity, medium, flux product and provenance are preserved. Wavelength
    dtype is preserved, subject to JAX canonicalization. No raw files are opened.
    """
    if not isinstance(spectra, _SpectralLibrary):
        raise TypeError("spectra must be a spectral-library loader/preparer result")
    mode = _sampling_mode(sampling, pixels, velocity_step)
    if mode == "native":
        raise ValueError("resampling requires 'log' or 'linear' sampling")
    original = np.asarray(spectra.wavelength)
    bounds = original[:, [0, -1]]
    wavelength = _target_wavelengths(bounds, mode, pixels, velocity_step, dtype=original.dtype)
    field = spectra.grid.field("flux")
    values = np.asarray(field.values)
    flux = np.empty(values.shape[:-1] + (wavelength.shape[-1],), dtype=values.dtype)
    # Offline work streams one atmosphere/region spectrum into the output.
    for index in np.ndindex(values.shape[:-2]):
        for region in range(original.shape[0]):
            flux[index + (region,)] = np.interp(wavelength[region], original[region],
                                              values[index + (region,)])
    grid = RectilinearGrid(
        axes={name: Axis(spectra.grid.axis(name), kind)
              for name, kind in zip(spectra.grid.axis_names, spectra.grid.axis_kinds)},
        fields={"flux": Field(flux, field.dims, payload_dims=field.payload_dims)},
        boundary=spectra.grid.boundary, fill_value=spectra.grid.fill_value,
    )
    return replace(spectra, grid=grid, wavelength=wavelength)


def is_log_uniform(spectra, *, rtol=1e-5):
    """Check all regions numerically, once at setup time (not a JIT operation).

    Each region must have at least two increasing positive finite samples.
    Compare ln(wavelength) with the affine grid implied by its endpoints. Maximum
    absolute log-coordinate residual must be <= rtol*dlnlambda plus roundoff:
    twice the stored wavelength dtype epsilon and eight float64 epsilons times
    the largest absolute log-wavelength. This accounts for float32 wavelength
    quantization without allowing error to accumulate with the pixel count.
    Very short ranges may be indistinguishable from linear sampling at that
    precision. The check uses arrays, never source labels or sampling metadata.
    """
    if not isinstance(spectra, _SpectralLibrary):
        raise TypeError("spectra must be a spectral-library loader/preparer result")
    if not np.isscalar(rtol) or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be a finite nonnegative scalar")
    return _is_log_uniform(np.asarray(spectra.wavelength), rtol)


def _is_log_uniform(wavelength, rtol=1e-5):
    try:
        _validate_wavelength(wavelength, ndim=2)
    except ValueError:
        return False
    if wavelength.shape[-1] < 2:
        return False
    logs = np.log(wavelength.astype(np.float64))
    step = (logs[:, -1] - logs[:, 0]) / (logs.shape[-1] - 1)
    expected = logs[:, :1] + step[:, None] * np.arange(logs.shape[-1])
    roundoff = (2 * np.finfo(wavelength.dtype).eps
                + 8 * np.finfo(np.float64).eps * np.max(np.abs(logs), axis=1))
    return bool(np.all(np.max(np.abs(logs - expected), axis=1) <= rtol * step + roundoff))


def _sampling_mode(sampling, pixels, velocity_step):
    # Preserve M2's omitted-pixels=native and explicit-pixels=linear behavior.
    mode = ("native" if pixels is None else "linear") if sampling is None else sampling
    if mode not in {"native", "linear", "log"}:
        raise ValueError("sampling must be 'native', 'linear' or 'log'")
    if pixels is not None and (not isinstance(pixels, (int, np.integer))
                               or isinstance(pixels, bool) or pixels < 2):
        raise ValueError("pixels must be an integer >= 2")
    if velocity_step is not None and mode != "log":
        raise ValueError("velocity_step requires sampling='log'")
    if mode == "native" and pixels is not None:
        raise ValueError("native sampling does not accept pixels")
    if mode == "linear" and pixels is None:
        raise ValueError("linear sampling requires pixels")
    if mode == "log" and (pixels is None) == (velocity_step is None):
        raise ValueError("log sampling requires exactly one of pixels or velocity_step")
    return mode


def _target_wavelengths(bounds, sampling, pixels, velocity_step, *, dtype=np.float64):
    bounds = np.asarray(bounds, dtype=np.float64)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
            or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] <= 0)
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("resampling requires a positive finite wavelength interval per region")
    if velocity_step is not None:
        velocity = np.asarray(velocity_step)
        if velocity.dtype.kind not in "iuf" or np.any(~np.isfinite(velocity)) or np.any(velocity <= 0):
            raise ValueError("velocity_step must be positive finite km/s")
        if velocity.ndim == 0:
            velocity = np.full(len(bounds), velocity, dtype=np.float64)
        elif velocity.shape != (len(bounds),):
            raise ValueError("velocity_step must be a scalar or one value per region")
        step = velocity.astype(np.float64) / _C_KMS
        spans = np.log(bounds[:, 1] / bounds[:, 0])
        intervals = spans / step
        if not np.all(np.isfinite(intervals)) or np.any(intervals >= np.iinfo(np.intp).max - 1):
            raise ValueError("velocity_step requests too many samples")
        # Allow endpoint roundoff when a requested interval is exactly integral.
        counts = np.floor(intervals + 8 * np.finfo(np.float64).eps / step).astype(np.intp) + 1
        if np.any(counts < 2):
            raise ValueError("velocity_step must provide at least two samples per region")
        if not np.all(counts == counts[0]):
            raise ValueError("velocity_step gives different region pixel counts; prepare separately or use pixels")
        rows = [lo * np.exp(delta * np.arange(count))
                for (lo, _), delta, count in zip(bounds, step, counts)]
    else:
        rows = [np.geomspace(lo, hi, pixels) if sampling == "log" else np.linspace(lo, hi, pixels)
                for lo, hi in bounds]
    dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
    wavelength = np.asarray(rows, dtype=dtype)
    # Keep representable endpoints inside coverage; never extrapolate by an ULP.
    for row, (lo, hi) in zip(wavelength, bounds):
        lower, upper = dtype.type(lo), dtype.type(hi)
        if lower < lo:
            lower = np.nextafter(lower, dtype.type(np.inf))
        if upper > hi:
            upper = np.nextafter(upper, dtype.type(-np.inf))
        row[0] = max(row[0], lower)
        row[-1] = min(row[-1], upper)
    try:
        _validate_wavelength(wavelength, ndim=2)
    except ValueError as error:
        raise ValueError("requested sampling is too fine for the wavelength dtype") from error
    if sampling == "log" and not _is_log_uniform(wavelength):
        raise ValueError("log-uniform sampling cannot be represented at the wavelength dtype precision")
    return wavelength
