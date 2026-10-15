"""Fixed ThAr IP loading, physical shifts, inference and result reconstruction."""

import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

from jaxstar.specfit import (
    SpecFit, SpecModel, load_fit_result, make_model_single, model_single,
    read_ip_calibration, save_fit_result, single_star_params,
)
from jaxstar.specfit.broadening import combined_kernel, double_gaussian_fourier, gaussian_sigma
from test_specmodel import make_library, observations, parameters


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def ip_values(**changes):
    return dict(mu1_kms=-1.1, mu2_kms=1.2, sigma1_kms=1.3, sigma2_kms=2.1,
                fraction2=.35) | changes


def write_calibration(path):
    rows = [[order, name, value] for order in (104, 105)
            for name, value in ip_values(mu1_kms=order - 106.).items()]
    # Unused derived rows and quantiles are deliberately irrelevant to fixed IP.
    rows.extend([[104, "R", "unused"], [104, "centroid_kms", "unused"]])
    with (path / "parameters.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["order", "parameter", "median"])
        writer.writerows(rows[::-1])


def test_csv_directory_lookup_order_alignment_and_repeated_orders(tmp_path):
    write_calibration(tmp_path)
    calibration = read_ip_calibration(tmp_path, orders=[105, "104", 105])
    assert calibration.orders == (105, "104", 105)
    assert calibration.source == tmp_path / "parameters.csv"
    np.testing.assert_array_equal(calibration.median["mu1_kms"], [-1., -2., -1.])
    np.testing.assert_array_equal(calibration.median["fraction2"], [.35] * 3)
    assert len(calibration.median) == 5
    with pytest.raises(TypeError):
        calibration.median["fraction2"] = 0.
    with pytest.raises(ValueError):
        calibration.median["fraction2"][0] = 0.


def test_csv_missing_file_header_order_and_duplicate(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_ip_calibration(tmp_path, orders=[104])
    (tmp_path / "parameters.csv").write_text("order,parameter,p16\n")
    with pytest.raises(ValueError, match="required CSV columns"):
        read_ip_calibration(tmp_path, orders=[104])
    write_calibration(tmp_path)
    with pytest.raises(ValueError, match="missing.*106"):
        read_ip_calibration(tmp_path, orders=[106])
    with (tmp_path / "parameters.csv").open("a") as stream:
        stream.write("104,sigma1_kms,1.3\n")
    with pytest.raises(ValueError, match="duplicate.*104.*sigma1_kms"):
        read_ip_calibration(tmp_path, orders=[104])
    for orders in ([], "104", [True], [104.5]):
        with pytest.raises(ValueError, match="orders must"):
            read_ip_calibration(tmp_path, orders=orders)


@pytest.mark.parametrize("fraction", [0., .27, 1.])
def test_kernel_matches_analytic_shifted_mixture(fraction):
    velocity = np.linspace(-40., 40., 641)
    values = ip_values(mu1_kms=-2.37, mu2_kms=3.21, fraction2=fraction)
    ip = double_gaussian_fourier(velocity, **values)
    kernel = combined_kernel(velocity, 0., 0., .5, .2, 0., ip_fourier=ip)
    expected = sum(weight / values[f"sigma{i}_kms"] * np.exp(
        -.5 * ((velocity - values[f"mu{i}_kms"]) / values[f"sigma{i}_kms"])**2)
        for i, weight in ((1, 1-fraction), (2, fraction)))
    expected /= expected.sum()
    np.testing.assert_allclose(kernel, expected, rtol=1e-11, atol=2e-16)
    np.testing.assert_allclose(kernel.sum(), 1., atol=2e-15)
    np.testing.assert_allclose(np.sum(kernel * velocity),
        (1-fraction) * values["mu1_kms"] + fraction * values["mu2_kms"], atol=2e-13)


def test_combined_kernel_matches_shifted_legacy_kernels():
    velocity = np.linspace(-60., 60., 481)
    values = ip_values(mu1_kms=-1.5, mu2_kms=2.75)
    first = combined_kernel(velocity, 3.1, 6.3, .5, .2, values["sigma1_kms"])
    second = combined_kernel(velocity, 3.1, 6.3, .5, .2, values["sigma2_kms"])
    expected = .65 * np.roll(first, -6) + .35 * np.roll(second, 11)
    actual = combined_kernel(velocity, 3.1, 6.3, .5, .2, 0.,
        ip_fourier=double_gaussian_fourier(velocity, **values))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-16)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_gaussian_limit_jit_and_region_isolation(dtype):
    library = make_library(dtype=dtype)
    params, wave = parameters(), observations(library)
    expected = SpecModel(library)(params, wave)
    params.pop("instrument")
    fixed = ip_values(mu1_kms=0., sigma1_kms=float(gaussian_sigma(70000.)), fraction2=0.)
    model = SpecModel(library, fixed_ip=fixed)
    compiled = jax.jit(lambda m, p, w: m(p, w))
    tolerance = 3e-7 if dtype == np.float32 else 3e-13
    np.testing.assert_allclose(compiled(model, params, wave), expected, rtol=0, atol=tolerance)
    # Fixed values and the precomputed factor are dynamic PyTree leaves.
    changed = SpecModel(library, fixed_ip={**fixed, "mu1_kms": [0., 4.]})
    changed_flux = compiled(changed, params, wave)
    np.testing.assert_allclose(changed_flux[0], expected[0], rtol=0, atol=tolerance)
    assert np.max(np.abs(changed_flux[1] - expected[1])) > .02
    fixed["mu1_kms"] = 100.
    exposed = model.fixed_ip
    exposed["mu1_kms"] = 200.
    np.testing.assert_array_equal(model.fixed_ip["mu1_kms"], [0., 0.])


def test_model_analytic_line_shift_and_single_region():
    library = make_library(regions=1)
    wave, params = observations(library), parameters()
    params.pop("instrument")
    params["components"][0]["broadening"].update(vsini=0., vmacro=0.)
    params["components"][0]["rv"] = 0.
    values = ip_values(mu1_kms=-2.37, mu2_kms=3.21)
    model = SpecModel(library, fixed_ip=values)
    velocity = 299792.458 * np.log(wave[0] / 5000.)
    expected = np.ones_like(velocity)
    for i, weight in ((1, .65), (2, .35)):
        sigma = np.hypot(3., values[f"sigma{i}_kms"])
        expected -= weight * .42 * 3 / sigma * np.exp(
            -.5 * ((velocity - values[f"mu{i}_kms"]) / sigma)**2)
    np.testing.assert_allclose(model(params, wave[0]), expected, rtol=0, atol=2e-11)
    np.testing.assert_array_equal(model(params, wave)[0], model(params, wave[0]))


def test_model_stellar_gradients_and_vmap():
    library = make_library()
    model, params, wave = SpecModel(library, fixed_ip=ip_values()), parameters(), observations(library)
    params.pop("instrument")
    def evaluate(theta):
        component = params["components"][0]
        component = {**component, "rv": theta[0], "broadening": {
            **component["broadening"], "vsini": theta[1], "vmacro": theta[2]}}
        return model({"components": (component,)}, wave)
    objective = lambda theta: jnp.sum(evaluate(theta) * jnp.linspace(.8, 1.3, wave.shape[-1])**2)
    theta = jnp.array([.43, 6.3, 3.1])
    _, derivative = jax.jit(jax.value_and_grad(objective))(theta)
    assert np.isfinite(derivative).all()
    step = 1e-3
    finite = [(objective(theta + step * row) - objective(theta - step * row)) / (2 * step)
              for row in np.eye(3)]
    np.testing.assert_allclose(derivative, finite, rtol=5e-4, atol=1e-8)
    batch = jnp.stack((theta, theta + jnp.array([1., 0., 0.])))
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(batch),
                               jnp.stack([evaluate(t) for t in batch]), atol=1e-12)


def test_fixed_ip_with_x64_disabled(x64_context):
    with x64_context(False):
        library = make_library(dtype=np.float32)
        model = SpecModel(library, fixed_ip=ip_values())
        params, wave = parameters(), observations(library)
        params.pop("instrument")
        assert model._ip_fourier.dtype == jnp.complex64
        eager = model(params, wave)
        compiled = jax.jit(lambda m, p, w: m(p, w))(model, params, wave)
        np.testing.assert_allclose(compiled, eager, rtol=2e-6, atol=2e-7)
        value, gradient = jax.jit(jax.value_and_grad(
            lambda p: jnp.sum(model(p, wave) * jnp.linspace(.8, 1.3, wave.shape[-1]))))(params)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(gradient))


@pytest.mark.parametrize("change,message", [
    ({"sigma1_kms": 0.}, "positive"),
    ({"sigma2_kms": np.nan}, "finite"),
    ({"fraction2": 1.1}, "fraction2"),
    ({"mu1_kms": [0., 1., 2.]}, "shape"),
    ({"mu2_kms": np.inf}, "finite"),
    ({"mu1_kms": 48.}, "increase SpecModel vmax"),
    ({"sigma2_kms": 20.}, "increase SpecModel vmax"),
])
def test_invalid_fixed_parameters(change, message):
    with pytest.raises(ValueError, match=message):
        SpecModel(make_library(), fixed_ip=ip_values(**change))


def test_exclusive_ip_modes():
    library = make_library()
    with pytest.raises(ValueError, match="exactly"):
        SpecModel(library, fixed_ip={"fraction2": .3})
    with pytest.raises(ValueError, match="custom broadening_operator"):
        SpecModel(library, fixed_ip=ip_values(), broadening_operator=lambda *args: None)
    model = SpecModel(library, fixed_ip=ip_values())
    with pytest.raises(ValueError, match="omit instrument"):
        model(parameters(), observations(library))


def test_inference_reconstruction_and_saved_fixed_ip(tmp_path):
    from test_specfit import case
    original, config, point = case()
    write_calibration(tmp_path)
    calibration = read_ip_calibration(tmp_path, orders=[105, 104])
    physical = SpecModel(original.specmodel.spectra, vmax=30., fixed_ip=calibration.median)
    fit = SpecFit(original.observation, physical)
    config.pop("resolving_power")
    trace = handlers.trace(handlers.substitute(model_single, data=point)).get_trace(
        fit.observation, physical, **config)
    assert "resolving_power" not in trace
    assert {name for name, site in trace.items() if site["type"] == "sample"
            and not site["is_observed"]} == {"teff"}
    density = lambda theta: log_density(model_single, (fit.observation, physical), config, theta)[0]
    value, gradient = jax.jit(jax.value_and_grad(density))({"teff": jnp.array(5800.)})
    assert np.isfinite(value) and np.isfinite(gradient["teff"])
    fixed = dict(config["broadening"], rv=12., jitter=.004)
    model = make_model_single(fit, fixed=fixed, use_gp=False, degree=2)
    params = single_star_params(point, **config["broadening"], rv=12.)
    prediction = model.predict(point, fit.observation, physical)
    np.testing.assert_allclose(prediction["physical_flux"], physical(params, fit.observation.wavelength))
    with pytest.raises(ValueError, match="omit resolving_power"):
        make_model_single(fit, resolving_power=70000., fixed=fixed, use_gp=False)
    with pytest.raises(ValueError, match="omit resolving_power"):
        model_single(fit.observation, physical, resolving_power=np.inf, **config)
    with pytest.raises(ValueError, match="resolving_power is required"):
        make_model_single(original, fixed=fixed, use_gp=False)
    output = save_fit_result(tmp_path / "result", sf=fit, model=model, point=point)
    result = load_fit_result(output)
    recorded = result.metadata["grid"]["fixed_ip"]
    restored_ip = {name: np.array(item["values"], dtype=item["dtype"])
                   for name, item in recorded.items()}
    restored = SpecFit(result.observation,
        SpecModel(physical.spectra, vmax=result.metadata["grid"]["vmax"], fixed_ip=restored_ip))
    restored_model = make_model_single(restored, fixed=fixed, use_gp=False, degree=2)
    np.testing.assert_array_equal(
        restored_model.predict(result.point, restored.observation, restored.specmodel)["mean_flux"],
        prediction["mean_flux"])


def test_grid_factories_forward_fixed_ip(tmp_path):
    from test_specfit_grid_setup import write_grids, observation
    _, wave = write_grids(tmp_path)
    obs = observation(wave, regions=("blue", "red"))
    write_calibration(tmp_path)
    # The caller associates unrelated region labels with order identifiers.
    order_for_region = {"blue": 105, "red": 104}
    values = read_ip_calibration(tmp_path, orders=[order_for_region[r] for r in obs.region]).median
    model = SpecModel.from_grid(tmp_path, wave, regions=obs.region, pixels=501, fixed_ip=values)
    fit = SpecFit.from_grid(tmp_path, obs, pixels=501, fixed_ip=values)
    np.testing.assert_array_equal(model.fixed_ip["mu1_kms"], [-1., -2.])
    np.testing.assert_array_equal(model._ip_fourier, fit.specmodel._ip_fourier)
    assert fit.rv_bounds is not None
