"""Log fitting-grid conventions, precision, validation and offline accuracy."""

import jax
import numpy as np
import pytest

from jaxstar.grid import Axis, Field, RectilinearGrid
from jaxstar.specfit import is_log_uniform, load_spectral_grid, resample_spectral_grid, save_spectral_grid
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit.sampling import _C_KMS, _sampling_mode, _target_wavelengths


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def library(wavelength, *, dtype=np.float64, function=None):
    wavelength = np.asarray(wavelength, dtype=dtype)
    if wavelength.ndim == 1:
        wavelength = wavelength[None]
    if function is None:
        function = lambda w: 3 + 0.02 * (w - 5000)
    flux = np.stack([function(wavelength) + offset for offset in (0., 0.2)]).astype(dtype)
    grid = RectilinearGrid(axes={"custom_abundance": Axis([0., 1.], "nonuniform")},
                           fields={"flux": Field(flux, ("custom_abundance",), payload_dims=("region", "pixel"))},
                           fill_value=-17.)
    return _SpectralLibrary(grid, wavelength, tuple(f"region-{i}" for i in range(len(wavelength))),
                            "independent-producer", "vacuum", "unnormalized", ("original-input",))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("control", ["velocity_step", "pixels"])
def test_sampling_exact_nodes_affine_accuracy_and_metadata(tmp_path, dtype, control):
    upper = 6120 if control == "velocity_step" else 6150
    original = library([np.linspace(5000, 5100, 501), np.linspace(6000, upper, 501)], dtype=dtype)
    options = {"velocity_step": 3.0} if control == "velocity_step" else {"pixels": 701}
    converted = resample_spectral_grid(original, **options)
    assert is_log_uniform(converted)
    assert converted.wavelength.dtype == dtype
    assert converted.grid.field("flux").values.dtype == dtype
    assert converted.grid.axis_kinds == original.grid.axis_kinds
    assert converted.grid.boundary == original.grid.boundary
    np.testing.assert_array_equal(converted.grid.axis("custom_abundance"), original.grid.axis("custom_abundance"))
    np.testing.assert_array_equal(converted.grid.fill_value, -17.)
    for attribute in ("regions", "library", "wavelength_medium", "wavelength_unit", "flux_kind", "sources"):
        assert getattr(converted, attribute) == getattr(original, attribute)
    for node in (0., 1.):
        expected = 3 + 0.02 * (converted.wavelength - 5000) + 0.2 * node
        np.testing.assert_allclose(converted.grid.interpolate({"custom_abundance": node})["flux"],
                                   expected, rtol=2e-7, atol=2e-7)
    result = jax.jit(lambda s, q: s.grid.interpolate({"custom_abundance": q})["flux"])(converted, np.array([0., .4, 1.]))
    np.testing.assert_allclose(result[1], .6 * result[0] + .4 * result[2], rtol=2e-7, atol=2e-7)
    assert result.shape == (3, 2, converted.wavelength.shape[-1])
    if control == "velocity_step":
        # Quantized float32 wavelengths allow their explicit representational
        # error; log differences themselves are evaluated at float64 precision.
        spacing = _C_KMS * np.diff(np.log(np.asarray(converted.wavelength, dtype=np.float64)), axis=1)
        np.testing.assert_allclose(spacing, 3.0, rtol=1e-8, atol=2 * np.finfo(dtype).eps * _C_KMS)
    restored = load_spectral_grid(save_spectral_grid(tmp_path / "sampling.npz", converted))
    assert is_log_uniform(restored)
    np.testing.assert_array_equal(restored.wavelength, converted.wavelength)
    np.testing.assert_array_equal(restored.grid.field("flux").values, converted.grid.field("flux").values)


@pytest.mark.parametrize("profile,tolerance", [("smooth", 2e-8), ("line", 1e-5)])
def test_synthetic_profile_accuracy_without_extra_normalization(profile, tolerance):
    # Dense linear input resolves the line; linear point interpolation has its
    # expected O(h^2) error against the analytic profile on target samples.
    wav = np.linspace(4995, 5005, 10001)
    function = (lambda w: 2 + 0.04 * np.sin((w - 5000) / 3)) if profile == "smooth" else (
        lambda w: 2.3 - 0.7 * np.exp(-0.5 * ((w - 5000.37) / 0.1) ** 2))
    original = library(wav, function=function)
    converted = resample_spectral_grid(original, velocity_step=0.09)
    actual = converted.grid.interpolate({"custom_abundance": 0.})["flux"][0]
    expected = function(np.asarray(converted.wavelength[0]))
    assert np.max(np.abs(actual - expected)) < tolerance
    np.testing.assert_array_equal(actual, np.interp(converted.wavelength[0], wav, function(wav)))
    assert np.median(actual) > 1.9  # unnormalized continuum remains unnormalized


def test_velocity_step_convention_endpoints_and_multi_region_counts():
    original = library([np.linspace(5000, 5010, 21), np.linspace(6000, 6012, 21)])
    converted = resample_spectral_grid(original, velocity_step=3.)
    assert is_log_uniform(converted)
    dln = np.diff(np.log(converted.wavelength), axis=1)
    np.testing.assert_allclose(dln * _C_KMS, 3., rtol=1e-9)
    np.testing.assert_array_equal(converted.wavelength[:, 0], original.wavelength[:, 0])
    upper_gap = np.log(original.wavelength[:, -1] / converted.wavelength[:, -1])
    assert np.all(upper_gap >= 0)
    assert np.all(upper_gap < 3 / _C_KMS)
    different = library([np.linspace(5000, 5010, 21), np.linspace(6000, 6024, 21)])
    with pytest.raises(ValueError, match="different region pixel counts"):
        resample_spectral_grid(different, velocity_step=3.)
    assert is_log_uniform(resample_spectral_grid(different, pixels=201))


@pytest.mark.parametrize("bounds", [(5000., 5010.), (6000., 6012.), (12000., 12060.)])
def test_canonical_velocity_spacing_is_independent_of_region(bounds):
    original = library(np.linspace(*bounds, 101))
    converted = resample_spectral_grid(original, sampling="log", velocity_step=1.0)
    spacing = _C_KMS * np.diff(np.log(converted.wavelength[0]))
    np.testing.assert_allclose(spacing, 1.0, rtol=1e-8)
    expected_count = int(np.floor(np.log(bounds[1] / bounds[0]) * _C_KMS)) + 1
    assert converted.wavelength.shape == (1, expected_count)
    assert is_log_uniform(converted)


def test_frozen_working_grid_logspace_and_trimmed_edges():
    # Exact construction from frozen specmodel.py; no sibling source import or
    # observed-model operation is needed to compare this numerical convention.
    bounds = np.array([[5000., 5100.], [6000., 6120.]])
    nwav = 1000
    frozen_full = np.array([np.logspace(np.log10(lo), np.log10(hi), nwav) for lo, hi in bounds])
    frozen_working = frozen_full[:, 1:-1]
    count = _target_wavelengths(bounds, "log", nwav, None)
    velocity = _target_wavelengths(bounds, "log", None, _C_KMS * np.log(bounds[:, 1] / bounds[:, 0]) / (nwav - 1))
    for result in (count, velocity):
        np.testing.assert_allclose(result[:, 1:-1], frozen_working, rtol=3e-15)
    # Requesting the trimmed bounds and Nwav-2 count reproduces the old working
    # grid itself. New preparation does not silently discard endpoints.
    trimmed = _target_wavelengths(frozen_working[:, [0, -1]], "log", nwav - 2, None)
    np.testing.assert_allclose(trimmed, frozen_working, rtol=3e-15)
    actual_velocity = np.diff(np.log(count), axis=1) * _C_KMS
    frozen_velocity = np.median(np.diff(np.log(frozen_working)), axis=1)[:, None] * _C_KMS
    np.testing.assert_allclose(actual_velocity, np.broadcast_to(frozen_velocity, actual_velocity.shape), rtol=1e-9)


def test_fixed_step_endpoint_roundoff_stays_inside_requested_coverage():
    step = 1e-5
    upper = 5000 * np.exp(100 * step)
    for _ in range(4):
        upper = np.nextafter(upper, -np.inf)
    wavelength = _target_wavelengths([[5000, upper]], "log", None, _C_KMS * step)
    assert wavelength.shape == (1, 101)
    assert np.all(wavelength >= 5000)
    assert np.all(wavelength <= upper)
    np.testing.assert_allclose(np.diff(np.log(wavelength)), step, rtol=1e-8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_validation_uses_actual_samples_and_precision(dtype):
    assert is_log_uniform(library(np.geomspace(5000, 5500, 1001), dtype=dtype))
    assert not is_log_uniform(library(np.linspace(5000, 5500, 1001), dtype=dtype))
    mixed = library([np.geomspace(5000, 5500, 101), np.linspace(6000, 6600, 101)], dtype=dtype)
    assert not is_log_uniform(mixed)
    assert not is_log_uniform(library([5000.], dtype=dtype))
    assert is_log_uniform(library([5000., 5100.], dtype=dtype))
    # A localized half-cell perturbation is not a uniform numerical grid.
    perturbed = np.geomspace(5000, 5500, 101).astype(dtype)
    perturbed[50] *= np.exp(0.5 * np.log(1.1) / 100)
    assert not is_log_uniform(library(perturbed, dtype=dtype))


def test_log_and_arbitrary_wavelength_storage_remain_supported(tmp_path, x64_context):
    original = library([5000., 5000.2, 5001.4, 5005., 5100.])
    for name, spectra in (("arbitrary", original), ("log", resample_spectral_grid(original, pixels=201))):
        path = save_spectral_grid(tmp_path / f"{name}.npz", spectra)
        restored = load_spectral_grid(path)
        np.testing.assert_array_equal(restored.wavelength, spectra.wavelength)
        np.testing.assert_array_equal(restored.grid.field("flux").values, spectra.grid.field("flux").values)
        assert is_log_uniform(restored) == (name == "log")
    with x64_context(False):
        converted = resample_spectral_grid(load_spectral_grid(path), pixels=151)
        assert converted.wavelength.dtype == np.float32
        assert converted.grid.field("flux").values.dtype == np.float32
        assert is_log_uniform(converted)


@pytest.mark.parametrize("kwargs,message", [
    ({}, "exactly one"), ({"pixels": 7, "velocity_step": 2}, "exactly one"),
    ({"pixels": 1}, "integer >= 2"), ({"pixels": True}, "integer >= 2"),
    ({"pixels": 2.5}, "integer >= 2"), ({"sampling": "bad", "pixels": 7}, "sampling"),
    ({"velocity_step": 0}, "positive finite"), ({"velocity_step": -1}, "positive finite"),
    ({"velocity_step": np.nan}, "positive finite"), ({"velocity_step": np.inf}, "positive finite"),
    ({"velocity_step": True}, "positive finite"),
    ({"velocity_step": [1, 2]}, "one value per region"),
    ({"velocity_step": 1000}, "at least two"),
    ({"sampling": "linear", "velocity_step": 2}, "requires sampling='log'"),
])
def test_ambiguous_or_invalid_sampling_fails(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resample_spectral_grid(library(np.linspace(5000, 5010, 21)), **kwargs)


def test_native_linear_backwards_compatibility_and_too_fine_precision(x64_context):
    assert _sampling_mode(None, None, None) == "native"
    assert _sampling_mode(None, 5, None) == "linear"
    with pytest.raises(ValueError, match="does not accept pixels"):
        _sampling_mode("native", 5, None)
    with pytest.raises(ValueError, match="requires pixels"):
        _sampling_mode("linear", None, None)
    with x64_context(False), pytest.raises(ValueError, match="too fine"):
        resample_spectral_grid(library([5000., 5001.]), pixels=10000)


@pytest.mark.parametrize("rtol", [-1, np.nan, np.inf, [1e-5]])
def test_invalid_validation_tolerance(rtol):
    with pytest.raises(ValueError, match="finite nonnegative"):
        is_log_uniform(library(np.geomspace(5000, 5100, 31)), rtol=rtol)
