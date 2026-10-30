"""Kernel widths from physical limits, calibrated IPs and actual convolution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from jaxstar.specfit import SpecModel
from jaxstar.specfit.model import _kernel_fwhm
from jaxstar.specfit.sampling import _C_KMS
from test_specmodel import make_library, parameters


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def minimal_params(**broadening):
    return {"components": ({"broadening": dict(
        vsini=0., vmacro=0., u1=.5, u2=.2) | broadening},),
        "instrument": {"resolving_power": 60000.}}


@pytest.mark.parametrize("x64", [False, True])
def test_gaussian_limit_region_order_and_jit(x64, x64_context):
    with x64_context(x64):
        dtype = np.float64 if x64 else np.float32
        model = SpecModel(make_library(dtype=dtype, pixels=4097))
        params = minimal_params()
        resolution = np.array([40000., 80000.], dtype=dtype)
        params["instrument"]["resolving_power"] = resolution
        width = model.broadening_fwhm(params)
        assert width.shape == (1, 2)
        assert width.dtype == dtype
        np.testing.assert_allclose(width[0], _C_KMS / resolution, rtol=4e-4)
        compiled = jax.jit(lambda m, p: m.broadening_fwhm(p))(model, params)
        np.testing.assert_allclose(compiled, width, rtol=2e-6)


def test_rotation_limit_is_not_a_gaussian_equivalent_width():
    model = SpecModel(make_library(regions=1, pixels=4097))
    params = minimal_params(vsini=15., u1=0., u2=0.)
    params["instrument"]["resolving_power"] = np.inf
    # A uniformly bright rotating disk has profile sqrt(1 - (v/vsini)**2).
    width = model.broadening_fwhm(params)
    assert width.shape == (1, 1)
    np.testing.assert_allclose(width[0, 0], np.sqrt(3) * 15., rtol=3e-3)


def test_asymmetric_fixed_ip_uses_actual_profile_without_resolving_power():
    library = make_library(pixels=4097)
    ip = dict(mu1_kms=-1.1, mu2_kms=1.8, sigma1_kms=1.4, sigma2_kms=3.2,
              fraction2=.3)
    model = SpecModel(library, fixed_ip=ip)
    params = minimal_params()
    params.pop("instrument")

    def profile(v):
        return sum(weight / ip[f"sigma{i}_kms"] * np.exp(
            -.5 * ((v - ip[f"mu{i}_kms"]) / ip[f"sigma{i}_kms"])**2)
            for i, weight in ((1, .7), (2, .3)))

    peak = minimize_scalar(lambda v: -profile(v), bounds=(-10., 10.), method="bounded").x
    half = profile(peak) / 2
    expected = (brentq(lambda v: profile(v) - half, peak, 25.)
                - brentq(lambda v: profile(v) - half, -25., peak))
    width = model.broadening_fwhm(params)
    np.testing.assert_allclose(width, expected, rtol=1e-3)
    compiled = jax.jit(lambda m, p: m.broadening_fwhm(p))
    np.testing.assert_allclose(compiled(model, params), width, rtol=1e-12)
    # A dynamically supplied model's per-region calibration changes the width.
    changed = SpecModel(library, fixed_ip={**ip, "sigma1_kms": [1.4, 2.5]})
    modified = compiled(changed, params)
    np.testing.assert_allclose(modified[0, 0], width[0, 0], rtol=1e-12)
    assert modified[0, 1] > 1.3 * width[0, 1]


def test_component_region_axes_and_irrelevant_parameters():
    model = SpecModel(make_library())
    params = minimal_params(vsini=[4., 12.], vmacro=[2., 6.])
    params["components"] += ({"broadening": dict(vsini=8., vmacro=3., u1=.3, u2=.1)},)
    width = model.broadening_fwhm(params)
    assert width.shape == (2, 2)
    assert width[0, 1] > width[0, 0]
    np.testing.assert_allclose(width[1, 0], width[1, 1], rtol=1e-12)
    for i, component in enumerate(params["components"]):
        single = {**params, "components": (component,)}
        np.testing.assert_array_equal(model.broadening_fwhm(single)[0], width[i])
    # Neither atmosphere evaluation nor composition/RV validation is needed.
    for component in params["components"]:
        component.update(atmosphere={"unused": np.nan}, rv=np.nan)
    params.update(flux_weights=(-1., -1.), dilution=np.nan)
    np.testing.assert_array_equal(model.broadening_fwhm(params), width)


def test_width_matches_the_kernel_applied_to_a_delta_line():
    library = make_library()
    params = parameters()
    model = SpecModel(library, fixed_ip=dict(
        mu1_kms=-1.1, mu2_kms=1.8, sigma1_kms=1.4, sigma2_kms=3.2, fraction2=.3))
    params.pop("instrument")
    impulse = np.zeros_like(library.wavelength)
    impulse[:, impulse.shape[1] // 2] = 1.
    _, response = model._broaden(params["components"][0], params, jnp.asarray(impulse))
    center = response.shape[1] // 2
    half = model.velocity_grid.shape[1] // 2
    response = response[:, center - half:center + half + 1]
    measured = _kernel_fwhm(model.velocity_grid, response)
    np.testing.assert_allclose(model.broadening_fwhm(params)[0], measured, rtol=1e-12)


def test_disabled_broadening_returns_zero_per_region():
    model = SpecModel(make_library())
    params = minimal_params()
    params["instrument"]["resolving_power"] = [np.inf, 60000.]
    width = model.broadening_fwhm(params)
    assert width[0, 0] == 0.
    assert width[0, 1] > 0.


def test_disconnected_half_maximum_intervals_fail():
    model = SpecModel(make_library(), fixed_ip=dict(
        mu1_kms=-8., mu2_kms=8., sigma1_kms=1.5, sigma2_kms=1.5, fraction2=.5))
    params = minimal_params()
    params.pop("instrument")
    with pytest.raises(ValueError, match="ambiguous.*multiple intervals"):
        model.broadening_fwhm(params)


@pytest.mark.parametrize("compiled", [False, True])
def test_unbracketed_half_maximum_fails(compiled):
    model = SpecModel(make_library(), vmax=5.)
    params = minimal_params()
    params["instrument"]["resolving_power"] = 10000.
    evaluate = lambda m, p: m.broadening_fwhm(p)
    if compiled:
        evaluate = jax.jit(evaluate)
    with pytest.raises(Exception, match="increase SpecModel vmax"):
        evaluate(model, params).block_until_ready()


def test_custom_operator_does_not_fall_back_to_an_unrelated_kernel():
    def identity(wavelength, flux, **unused):
        return wavelength, flux
    model = SpecModel(make_library(), broadening_operator=identity)
    with pytest.raises(NotImplementedError, match="custom broadening operators"):
        model.broadening_fwhm(minimal_params())


@pytest.mark.parametrize("field,value,message", [
    ("vsini", -1., "nonnegative"),
    ("vmacro", np.nan, "finite"),
    ("vsini", [1., 2., 3.], "scalar or shape"),
    ("vsini", 100., "increase SpecModel vmax"),
])
def test_width_shares_spectral_broadening_validation(field, value, message):
    params = minimal_params(**{field: value})
    with pytest.raises(ValueError, match=message):
        SpecModel(make_library()).broadening_fwhm(params)


def test_fixed_ip_keeps_instrument_mode_validation():
    model = SpecModel(make_library(), fixed_ip=dict(
        mu1_kms=0., mu2_kms=0., sigma1_kms=2., sigma2_kms=3., fraction2=.3))
    with pytest.raises(ValueError, match="omit instrument"):
        model.broadening_fwhm(minimal_params())
