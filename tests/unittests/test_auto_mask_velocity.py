"""Automatic dilation uses one measured half-FWHM, with explicit overrides."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from jaxstar.specfit import Observation, SpecFit, SpecModel
from jaxstar.specfit.sampling import _C_KMS
from jaxstar.specfit.specfit import _extend_mask
from test_specmodel import make_library, observations, parameters


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def auto_case(single=False, fixed_ip=False):
    library = make_library(regions=1 if single else 2)
    ip = dict(mu1_kms=-1., mu2_kms=2., sigma1_kms=1.5 if single else [1.5, 3.],
              sigma2_kms=2.5, fraction2=.3)
    physical = SpecModel(library, fixed_ip=ip if fixed_ip else None)
    wave = observations(library)
    if single:
        wave = wave[0]
    fixed = np.zeros(wave.shape, bool)
    fixed[..., 3] = True
    fit = SpecFit(Observation(wave, np.ones(wave.shape), np.full(wave.shape, .01), fixed), physical)
    params = parameters()
    params["components"][0]["broadening"].update(vsini=0., vmacro=0.)
    params["instrument"] = {} if fixed_ip else {
        "resolving_power": 40000. if single else np.array([40000., 80000.])}
    probability = np.full(wave.shape, .1)
    probability[..., [0, 10, 12, 50, -1]] = .9
    probability[fixed] = .99  # Fixed-mask pixels must not seed extension.
    prediction = dict(params=params, model_flux=np.ones(wave.shape),
                      normal_residual_variance=np.full(wave.shape, .01**2),
                      mismatch_probability=probability)
    calls = []

    def predict(point, observation, specmodel):
        calls.append(point)
        return prediction

    return fit, SimpleNamespace(predict=predict), prediction, calls


def expected_velocity_mask(fit, velocity):
    probabilities = fit.outlier_diagnostics["mismatch_probability"]
    flags = []
    for wave, q in zip(np.atleast_2d(fit.observation.wavelength), np.atleast_2d(probabilities)):
        distance = _C_KMS * np.abs(np.log(wave[:, None] / wave[q > .5]))
        flags.append(np.any(distance <= velocity, axis=1))
    return np.asarray(flags).reshape(fit.observation.shape) & ~fit.observation.mask


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("fixed_ip", [False, True])
def test_default_auto_global_half_fwhm_and_one_prediction(single, fixed_ip, capsys):
    fit, model, prediction, calls = auto_case(single, fixed_ip)
    widths = np.asarray(fit.specmodel.broadening_fwhm(prediction["params"]))
    largest = float(np.max(widths))
    old = np.zeros(fit.observation.shape, bool)
    old[..., 80] = True
    fit.set_fit_mask(old)
    flags = fit.mask_outliers(point={}, model=model, extension_factor=100.).copy()
    assert len(calls) == 1
    d = fit.outlier_diagnostics
    assert d["mask_velocity"] == .5 * largest
    assert d["mask_velocity_source"] == "auto"
    assert d["max_kernel_fwhm"] == largest
    np.testing.assert_array_equal(flags, expected_velocity_mask(fit, .5 * largest))
    assert not flags[..., 80].any()
    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert f"mask_velocity = {.5 * largest:.6g} km/s" in output[0]
    assert f"max kernel FWHM {largest:.6g} km/s" in output[0]
    # Explicit auto and any extension_factor give the same nonrecursive mask.
    repeated = fit.mask_outliers(point={}, model=model, mask_velocity="auto", extension_factor=0.)
    np.testing.assert_array_equal(repeated, flags)
    assert len(calls) == 2
    # The recorded scalar can reproduce the mask as an explicit override.
    np.testing.assert_array_equal(fit.mask_outliers(point={}, model=model, mask_velocity=d["mask_velocity"]), flags)


def test_sbn_ignores_dark_components_and_zero_weight_regions(capsys):
    fit, model, prediction, _ = auto_case()
    params = prediction["params"]
    first = copy.deepcopy(params["components"][0])
    second = copy.deepcopy(first)
    second["broadening"].update(vsini=[8., 30.], vmacro=3.)
    dark = copy.deepcopy(first)
    dark["broadening"]["vsini"] = 10000.  # No kernel calculation for a completely dark star.
    params.update(components=(first, second, dark), flux_weights=(1., np.array([.25, 0.]), 0.))
    contributing = {"components": (first, second), "instrument": params["instrument"]}
    widths = np.asarray(fit.specmodel.broadening_fwhm(contributing))
    expected = max(widths[0, 0], widths[0, 1], widths[1, 0])
    assert widths[1, 1] > expected
    flags = fit.mask_outliers(point={}, model=model)
    assert fit.outlier_diagnostics["max_kernel_fwhm"] == expected
    assert fit.outlier_diagnostics["mask_velocity"] == .5 * expected
    np.testing.assert_array_equal(flags, expected_velocity_mask(fit, .5 * expected))
    assert capsys.readouterr().out.count("mask_velocity =") == 1


@pytest.mark.parametrize("velocity,extend", [(10., True), (0., True), (None, True),
                                            ("auto", False), (10., False), (None, False)])
def test_explicit_legacy_and_disabled_skip_kernel_calculation(velocity, extend, monkeypatch, capsys):
    fit, model, prediction, calls = auto_case()
    prediction.pop("params")

    def forbidden(*args, **kwargs):
        raise AssertionError("this path must not calculate a kernel width")

    monkeypatch.setattr(SpecModel, "broadening_fwhm", forbidden)
    flags = fit.mask_outliers(point={}, model=model, mask_velocity=velocity,
                              extend_outlier_mask=extend, extension_factor=2.)
    d = fit.outlier_diagnostics
    assert len(calls) == 1 and capsys.readouterr().out == ""
    assert d["max_kernel_fwhm"] is None
    if not extend:
        assert d["mask_velocity"] == 0. and d["mask_velocity_source"] == "disabled"
        expected = d["mismatch_probability"] > .5
    elif velocity is None:
        assert d["mask_velocity"] is None and d["mask_velocity_source"] == "run_length"
        expected = np.stack([_extend_mask(row, 2.) for row in np.atleast_2d(d["mismatch_probability"] > .5)])
    else:
        assert d["mask_velocity"] == velocity and d["mask_velocity_source"] == "explicit"
        expected = expected_velocity_mask(fit, velocity)
    np.testing.assert_array_equal(flags, expected & ~fit.observation.mask)


def test_zero_broadening_gives_zero_padding(capsys):
    fit, model, prediction, _ = auto_case()
    prediction["params"]["instrument"]["resolving_power"] = np.inf
    flags = fit.mask_outliers(point={}, model=model)
    assert fit.outlier_diagnostics["mask_velocity"] == 0.
    np.testing.assert_array_equal(flags, fit.outlier_diagnostics["mismatch_probability"] > .5)
    assert "mask_velocity = 0 km/s" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["missing_params", "custom_operator", "missing_weights",
                                    "invalid_kernel", "bad_shape", "nan", "negative"])
def test_auto_failure_requires_override_and_preserves_state(failure, monkeypatch, capsys):
    fit, model, prediction, _ = auto_case()
    old = np.zeros(fit.observation.shape, bool)
    old[..., 80] = True
    fit.set_fit_mask(old)
    previous_diagnostics = {"previous": True}
    fit.outlier_diagnostics = previous_diagnostics
    if failure == "missing_params":
        prediction.pop("params")
    elif failure == "custom_operator":
        fit.specmodel = SpecModel(fit.specmodel.spectra, broadening_operator=lambda *args, **kwargs: None)
    elif failure == "missing_weights":
        prediction["params"]["components"] *= 2
    elif failure == "invalid_kernel":
        prediction["params"]["instrument"]["resolving_power"] = 1000.
    else:
        widths = np.zeros(2) if failure == "bad_shape" else np.full((1, 2), np.nan if failure == "nan" else -1.)
        monkeypatch.setattr(SpecModel, "broadening_fwhm", lambda *args: widths)
    with pytest.raises(ValueError, match="supply an explicit mask_velocity"):
        fit.mask_outliers(point={}, model=model)
    np.testing.assert_array_equal(fit.fit_mask, old)
    assert fit.outlier_diagnostics is previous_diagnostics
    assert capsys.readouterr().out == ""
    # Even if auto is unavailable, a numerical override remains usable.
    fit.mask_outliers(point={}, model=model, mask_velocity=5.)
    assert fit.outlier_diagnostics["mask_velocity"] == 5.
