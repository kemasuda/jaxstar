"""Observer-frame calibration, constant-IP parity and inference integration."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (
    Observation, SpecFit, SpecModel, load_fit_result, make_model_single, model_single, save_fit_result,
)
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit.sampling import _C_KMS
from test_ip_calibration import ip_values
from test_specmodel import make_library, observations, parameters


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def fixed_params():
    params = parameters()
    params.pop("instrument")
    return params


def varying_ip(wave):
    x = _C_KMS * np.log(wave / np.array([5000., 6000.])[:len(wave), None])
    return ip_values(mu1_kms=-1. + .01*x, sigma2_kms=2.5 + .002*x)


def test_callable_evaluated_once_and_copied_into_dynamic_pytree():
    library = make_library()
    calls, returned = [], []

    def calibration(wave):
        assert isinstance(wave, np.ndarray) and not wave.flags.writeable
        np.testing.assert_array_equal(wave, library.wavelength)
        calls.append(wave.shape)
        values = varying_ip(wave)
        returned.append(values)
        return values

    model = SpecModel(library, fixed_ip=calibration)
    params, wave = fixed_params(), observations(library)
    expected = model(params, wave)
    returned[0]["mu1_kms"][:] = 100.
    exposed = model.fixed_ip
    exposed["mu1_kms"] = 100.
    compiled = jax.jit(lambda m, p, w: m(p, w))
    np.testing.assert_allclose(compiled(model, params, wave), expected, atol=1e-12)
    assert np.all(np.isfinite(model.broadening_fwhm(params)))
    assert calls == [library.wavelength.shape]
    assert model._ip_fourier is None  # stellar broadening runs with beta=0
    assert model.fixed_ip["sigma1_kms"].shape == library.wavelength.shape
    changed = SpecModel(library, fixed_ip=lambda w: ip_values(mu1_kms=[-1., 5.]))
    assert np.max(np.abs(compiled(changed, params, wave) - expected)) > .01


@pytest.mark.parametrize("macro_model", ["rt", "gaussian"])
@pytest.mark.parametrize("rv", [0., .43, 12.])
def test_constant_callable_matches_existing_combined_kernel(macro_model, rv):
    library = make_library()
    params, wave = fixed_params(), observations(library)
    params["components"][0]["rv"] = rv
    constant = SpecModel(library, fixed_ip=ip_values(), macro_model=macro_model)
    variable = SpecModel(library, fixed_ip=lambda w: ip_values(), macro_model=macro_model)
    # Separate finite-support convolutions differ from the combined Fourier
    # kernel only by small tail/discretization errors on this resolved grid.
    np.testing.assert_allclose(variable(params, wave), constant(params, wave), atol=3e-7, rtol=0)
    np.testing.assert_allclose(variable.broadened(params, wave),
                               constant.broadened(params, wave), atol=3e-7, rtol=0)
    np.testing.assert_array_equal(variable.intrinsic(params, wave), constant.intrinsic(params, wave))


@pytest.mark.parametrize("shift", [-31.25, 31.25])
def test_incident_wavelength_shift_field_has_analytic_profile(shift):
    """mu(x)=a*x+b stretches a line and moves it by a*RV in the observer frame.

    This distinguishes input- from output-indexed calibration, and both from
    applying the IP before RV. The flat response is 1/(1+a); normalizing by it
    gives the analytic continuum-normalized mixture below.
    """
    library = make_library(regions=1)
    slope = .1

    def calibration(wave):
        x = _C_KMS * np.log(wave / 5000.)
        return ip_values(mu1_kms=slope*x - 1.1, mu2_kms=slope*x + 1.2)

    model = SpecModel(library, fixed_ip=calibration)
    params, wave = fixed_params(), observations(library)
    params["components"][0]["broadening"].update(vsini=0., vmacro=0.)
    params["components"][0]["rv"] = _C_KMS * np.tanh(shift / _C_KMS)
    x = _C_KMS * np.log(wave[0] / 5000.)
    expected = np.ones_like(x)
    for weight, mu, sigma in ((.65, -1.1, 1.3), (.35, 1.2, 2.1)):
        intrinsic_sigma = (1 + slope) * 3.
        width = np.hypot(intrinsic_sigma, sigma)
        center = (1 + slope) * shift + mu
        expected -= weight * .42 * intrinsic_sigma / width * np.exp(-.5*((x-center)/width)**2)
    np.testing.assert_allclose(model(params, wave[0]), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize("shift,sigma", [(-46.875, 1.2), (46.875, 4.5)])
def test_rv_uses_width_at_shifted_line_location(shift, sigma):
    library = make_library(regions=1)

    def calibration(wave):
        return ip_values(mu1_kms=0., sigma1_kms=np.where(wave < 5000., 1.2, 4.5), fraction2=0.)

    model = SpecModel(library, fixed_ip=calibration)
    params, wave = fixed_params(), observations(library)
    params["components"][0]["broadening"].update(vsini=0., vmacro=0.)
    params["components"][0]["rv"] = _C_KMS * np.tanh(shift / _C_KMS)
    x = _C_KMS * np.log(wave[0] / 5000.)
    width = np.hypot(3., sigma)
    expected = 1 - .42 * 3 / width * np.exp(-.5*((x-shift)/width)**2)
    np.testing.assert_allclose(model(params, wave[0]), expected, atol=2e-6, rtol=0)
    # Calibration changes alone must not put a feature in a flat spectrum.
    flat = model._ip_response.apply(jnp.ones_like(library.wavelength))
    np.testing.assert_allclose(flat, 1., atol=1e-14)


def test_jit_gradients_vmap_and_sbn_composition():
    library = make_library()
    model, params, wave = SpecModel(library, fixed_ip=varying_ip), fixed_params(), observations(library)

    def evaluate(theta):
        original = params["components"][0]
        component = {**original, "rv": theta[0], "broadening": {
            **original["broadening"], "vsini": theta[1], "vmacro": theta[2]}}
        return model({"components": (component,)}, wave)

    theta = jnp.array([.43, 6.3, 3.1])
    objective = lambda theta: jnp.sum(evaluate(theta) * jnp.linspace(.8, 1.3, wave.shape[-1])**2)
    value, gradient = jax.jit(jax.value_and_grad(objective))(theta)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    step = 1e-3
    finite = [(objective(theta + step*row) - objective(theta - step*row))/(2*step) for row in np.eye(3)]
    np.testing.assert_allclose(gradient, finite, rtol=5e-4, atol=1e-8)
    batch = jnp.stack((theta, theta + jnp.array([12., 0., 0.])))
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(batch),
                               jnp.stack([evaluate(t) for t in batch]), atol=1e-12)

    other = deepcopy(params["components"][0])
    other["rv"] = 21.
    other["atmosphere"]["custom_temperature"] = .8
    params.update(components=(*params["components"], other), flux_weights=[[1., 2.], [3., 1.]], dilution=.1)
    parts = jax.jit(lambda m, p, w: m.decompose(p, w))(model, params, wave)
    np.testing.assert_allclose(parts.total, model(params, wave), atol=1e-12)
    for i, component in enumerate(params["components"]):
        np.testing.assert_allclose(parts.components[i], model({"components": (component,)}, wave), atol=1e-12)


def test_float32_and_region_broadcasting(x64_context):
    with x64_context(False):
        library = make_library(dtype=np.float32)
        model = SpecModel(library, fixed_ip=lambda w: ip_values(mu1_kms=[-1., 2.], sigma2_kms=[[2.], [3.]]))
        params, wave = fixed_params(), observations(library)
        assert model._ip_response.weights.dtype == jnp.float32
        np.testing.assert_array_equal(model.fixed_ip["mu1_kms"][:, 0], [-1., 2.])
        eager = model(params, wave)
        np.testing.assert_allclose(jax.jit(lambda m, p, w: m(p, w))(model, params, wave), eager,
                                   rtol=2e-6, atol=2e-7)
        _, gradient = jax.jit(jax.value_and_grad(lambda p: jnp.sum(model(p, wave))))(params)
        assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(gradient))


def test_unused_margin_does_not_change_prediction_and_missing_coverage_fails():
    library = make_library()
    cropped_grid = RectilinearGrid(axes={"custom_temperature": np.array([0., 1.])},
        fields={"flux": Field(library.grid.field("flux").values[..., 80:-80],
                              ("custom_temperature",), payload_dims=("region", "pixel"))})
    cropped = _SpectralLibrary(cropped_grid, library.wavelength[:, 80:-80], library.regions,
                               library.library, library.wavelength_medium, library.flux_kind)
    params, wave = fixed_params(), observations(library)
    full = SpecModel(library, fixed_ip=varying_ip)
    small = SpecModel(cropped, fixed_ip=varying_ip)
    np.testing.assert_allclose(full(params, wave), small(params, wave), atol=2e-12)
    params["components"][0]["rv"] = 190.
    with pytest.raises(ValueError, match="coverage"):
        full(params, wave)
    edge = np.asarray(library.wavelength[:, 1:4])
    with pytest.raises(ValueError, match="coverage for fixed_ip"):
        SpecFit(Observation(edge, np.ones_like(edge), np.full_like(edge, .01)), full)


@pytest.mark.parametrize("change,message", [
    ({"sigma1_kms": 0.}, "positive"),
    ({"sigma2_kms": np.nan}, "finite"),
    ({"fraction2": 1.1}, "fraction2"),
    ({"mu1_kms": np.zeros((3, 4))}, "shape"),
    ({"sigma1_kms": 300.}, "coverage"),
])
def test_invalid_callable_values(change, message):
    with pytest.raises(ValueError, match=message):
        SpecModel(make_library(), fixed_ip=lambda w: ip_values(**change))


def test_callable_modes_and_stage_semantics():
    library = make_library()
    model, params, wave = SpecModel(library, fixed_ip=varying_ip), fixed_params(), observations(library)
    with pytest.raises(ValueError, match="omit instrument"):
        model(parameters(), wave)
    with pytest.raises(ValueError, match="custom broadening_operator"):
        SpecModel(library, fixed_ip=varying_ip, broadening_operator=lambda *args: None)
    params["components"][0]["rv"] = 0.
    expected = model(params, wave)
    del params["components"][0]["rv"]
    np.testing.assert_array_equal(model.broadened(params, wave), expected)


@pytest.mark.parametrize("macro_model", ["rt", "gaussian"])
@pytest.mark.parametrize("pixels", [1024, 1025])
def test_representative_fwhm_uses_central_ip_per_region(macro_model, pixels):
    library = make_library(pixels=pixels)

    def calibration(wave):
        values = varying_ip(wave)
        values["sigma1_kms"] = np.array([1.3, 2.6])[:, None]
        return values

    variable = SpecModel(library, fixed_ip=calibration, macro_model=macro_model)
    center = {name: value[:, pixels // 2] for name, value in variable.fixed_ip.items()}
    constant = SpecModel(library, fixed_ip=center, macro_model=macro_model)
    params = fixed_params()
    other = deepcopy(params["components"][0])
    other["broadening"].update(vsini=12., vmacro=5.)
    params["components"] += (other,)
    expected = constant.broadening_fwhm(params)
    assert expected.shape == (2, 2)
    assert expected[0, 1] > expected[0, 0]
    np.testing.assert_allclose(variable.broadening_fwhm(params), expected, atol=1e-12)
    compiled = jax.jit(lambda m, p: m.broadening_fwhm(p))
    np.testing.assert_allclose(compiled(variable, params), expected, atol=1e-12)


def test_representative_fwhm_checks_central_ip_support():
    model = SpecModel(make_library(), vmax=5., fixed_ip=lambda w: ip_values())
    with pytest.raises(ValueError, match="central fixed_ip.*broadening support"):
        model.broadening_fwhm(fixed_params())


def test_inference_rv_bounds_and_saved_calibration(tmp_path):
    from test_specfit import case
    original, config, point = case()
    physical = SpecModel(original.specmodel.spectra, vmax=30., fixed_ip=varying_ip)
    fit = SpecFit(original.observation, physical)
    assert np.all(fit.rv_bounds[:, 0] > original.rv_bounds[:, 0])
    assert np.all(fit.rv_bounds[:, 1] < original.rv_bounds[:, 1])
    config.pop("resolving_power")
    density = lambda theta: log_density(model_single, (fit.observation, physical), config, theta)[0]
    value, gradient = jax.jit(jax.value_and_grad(density))({"teff": jnp.array(5800.)})
    assert np.isfinite(value) and np.isfinite(gradient["teff"])
    fixed = dict(config["broadening"], rv=12., jitter=.004)
    model = make_model_single(fit, fixed=fixed, use_gp=False, degree=2)
    prediction = model.predict(point, fit.observation, physical)
    output = save_fit_result(tmp_path / "result", sf=fit, model=model, point=point)
    result = load_fit_result(output)
    record = result.metadata["grid"]
    assert record["fixed_ip_mode"] == "wavelength_dependent"
    values = {name: np.asarray(item["values"], dtype=item["dtype"])
              for name, item in record["fixed_ip"].items()}
    restored = SpecModel(physical.spectra, vmax=record["vmax"], fixed_ip=lambda w: values)
    np.testing.assert_array_equal(restored(prediction["params"], fit.observation.wavelength),
                                   prediction["physical_flux"])
    fit.mask_outliers(point=point, model=model, plot=False)
    expected_width = .5 * np.max(physical.broadening_fwhm(prediction["params"]))
    np.testing.assert_allclose(fit.outlier_diagnostics["mask_velocity"], expected_width)
    assert fit.outlier_diagnostics["mask_velocity_source"] == "auto"
    fit.mask_outliers(point=point, model=model, mask_velocity=10., plot=False)
    assert fit.outlier_diagnostics["mask_velocity"] == 10.


def test_factories_forward_callable(tmp_path):
    from test_specfit_grid_setup import write_grids, observation
    _, wave = write_grids(tmp_path)
    obs = observation(wave, regions=("blue", "red"))
    calibration = lambda w: ip_values(mu1_kms=[-1., 2.])
    model = SpecModel.from_grid(tmp_path, wave, regions=obs.region, pixels=501, fixed_ip=calibration)
    fit = SpecFit.from_grid(tmp_path, obs, pixels=501, fixed_ip=calibration)
    np.testing.assert_array_equal(model._ip_response.weights, fit.specmodel._ip_response.weights)
