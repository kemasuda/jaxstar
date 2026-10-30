"""Gaussian macro limits, fixed-IP convolution, gradients and mode selection."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit import SpecModel, model_single
from jaxstar.specfit.broadening import combined_kernel, double_gaussian_fourier, gaussian_sigma
from test_specmodel import make_library, observations, parameters


def fixed_ip():
    return dict(mu1_kms=-1.1, mu2_kms=1.8, sigma1_kms=1.4, sigma2_kms=3.2,
                fraction2=.3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("double_ip", [False, True])
def test_nonrotating_kernel_matches_analytic_gaussians(dtype, double_ip, x64_context):
    with x64_context(dtype == np.float64):
        # Binary-exact spacing isolates the profile from float32 grid rounding.
        velocity = np.linspace(-64., 64., 513).astype(dtype)
        sigma_macro, beta = 3.7, 1.8
        ip = fixed_ip() if double_ip else dict(
            mu1_kms=0., mu2_kms=0., sigma1_kms=beta, sigma2_kms=beta, fraction2=0.)
        factor = double_gaussian_fourier(velocity, **ip) if double_ip else None
        # An explicit IP replaces beta, even when beta is nonzero.
        kernel = combined_kernel(velocity, sigma_macro, 0., .5, .2, beta,
                                 ip_fourier=factor, macro_model="gaussian")
        expected = np.zeros_like(velocity)
        for i, weight in ((1, 1 - ip["fraction2"]), (2, ip["fraction2"])):
            sigma = np.hypot(ip[f"sigma{i}_kms"], sigma_macro)
            expected += weight / sigma * np.exp(-.5 * ((velocity - ip[f"mu{i}_kms"]) / sigma)**2)
        expected /= expected.sum()
        np.testing.assert_allclose(kernel, expected, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(kernel.sum(), 1., atol=2e-7)


def test_rt_default_and_zero_macro_modes_agree(x64_context):
    with x64_context():
        velocity = np.linspace(-50., 50., 501)
        args = (velocity, 3.1, 6.3, .5, .2, 1.8)
        np.testing.assert_array_equal(combined_kernel(*args), combined_kernel(*args, macro_model="rt"))
        for ip in (None, double_gaussian_fourier(velocity, **fixed_ip())):
            args = (velocity, 0., 6.3, .5, .2, 1.8)
            np.testing.assert_array_equal(
                combined_kernel(*args, ip_fourier=ip, macro_model="gaussian"),
                combined_kernel(*args, ip_fourier=ip, macro_model="rt"))
        with pytest.raises(ValueError, match="macro_model"):
            combined_kernel(*args, macro_model="unknown")


@pytest.mark.parametrize("double_ip", [False, True])
def test_rotating_model_jit_gradients_and_fwhm_match_absorbed_macro(double_ip, x64_context):
    with x64_context():
        library = make_library()
        wave, params = observations(library), parameters()
        sigma_macro = np.array([2.3, 4.1])
        params["components"][0]["broadening"]["vmacro"] = sigma_macro
        params["components"][0]["broadening"]["vsini"] = np.array([6.3, 9.2])
        reference_params = copy.deepcopy(params)
        reference_params["components"][0]["broadening"]["vmacro"] = 0.
        ip = fixed_ip() if double_ip else None
        if double_ip:
            params.pop("instrument")
            reference_params.pop("instrument")
            effective_ip = dict(ip)
            for name in ("sigma1_kms", "sigma2_kms"):
                effective_ip[name] = np.hypot(ip[name], sigma_macro)
            reference = SpecModel(library, fixed_ip=effective_ip)
        else:
            sigma_ip = gaussian_sigma(params["instrument"]["resolving_power"])
            reference_params["instrument"]["resolving_power"] = (
                299792.458 / 2.354820 / np.hypot(sigma_ip, sigma_macro))
            reference = SpecModel(library)
        model = SpecModel(library, fixed_ip=ip, macro_model="gaussian")
        leaves, treedef = jax.tree_util.tree_flatten(model)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert restored.macro_model == "gaussian"
        evaluate = jax.jit(lambda m, p: m(p, wave))
        expected = reference(reference_params, wave)
        np.testing.assert_allclose(evaluate(restored, params), expected, atol=2e-12)
        np.testing.assert_allclose(model(params, wave), expected, atol=2e-12)
        widths = jax.jit(lambda m, p: m.broadening_fwhm(p))
        np.testing.assert_allclose(widths(restored, params),
                                   reference.broadening_fwhm(reference_params), atol=2e-12)
        # The same compiled callable must distinguish static RT/Gaussian modes.
        rt = SpecModel(library, fixed_ip=ip)
        assert np.max(np.abs(evaluate(rt, params) - expected)) > 1e-3

        def center_flux(sigma):
            point = copy.deepcopy(params)
            point["components"][0]["broadening"]["vmacro"] = sigma
            return evaluate(model, point)[0, wave.shape[1] // 2]

        gradient = jax.jit(jax.grad(center_flux))(3.)
        numerical = (center_flux(3. + 1e-3) - center_flux(3. - 1e-3)) / 2e-3
        assert gradient > 0
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6)


@pytest.mark.parametrize("double_ip", [False, True])
def test_numpyro_keeps_sampled_vmacro_and_differentiable_density(double_ip, x64_context):
    from test_numpyro_spectral import _case
    with x64_context():
        obs, original, config, point, _ = _case()
        if double_ip:
            config.pop("resolving_power")
        model = SpecModel(original.spectra, vmax=30., macro_model="gaussian",
                          fixed_ip=fixed_ip() if double_ip else None)
        config["broadening"]["vmacro"] = dist.Uniform(0., 10.)
        theta = dict(teff=jnp.array(point["teff"]), vmacro=jnp.array(3.))
        trace = handlers.trace(handlers.substitute(model_single, data=theta)).get_trace(
            obs, model, **config)
        assert trace["vmacro"]["type"] == "sample"
        density = lambda p: log_density(model_single, (obs, model), config, p)[0]
        value, gradient = jax.jit(jax.value_and_grad(density))(theta)
        assert np.isfinite(value)
        assert np.isfinite(gradient["vmacro"]) and gradient["vmacro"] != 0


def test_invalid_selection_and_custom_operator_contract():
    library, params = make_library(), parameters()
    for invalid in ("unknown", None, 1):
        with pytest.raises(ValueError, match="macro_model"):
            SpecModel(library, macro_model=invalid)

    def identity(wavelength, flux, *, broadening, resolving_power, velocity_grid):
        return wavelength, flux

    custom = SpecModel(library, broadening_operator=identity)
    assert np.isfinite(custom(params, observations(library))).all()
    with pytest.raises(ValueError, match="built-in broadening operator"):
        SpecModel(library, broadening_operator=identity, macro_model="gaussian")
