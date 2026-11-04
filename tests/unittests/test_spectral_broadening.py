"""Low-level combined kernel and Gaussian convention checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import j0, j1

from jaxstar.specfit.broadening import _j0, combined_kernel, gaussian_sigma


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bessel_and_kernel_precision_normalization_and_symmetry(dtype, x64_context):
    with x64_context():
        arguments = np.linspace(-30, 30, 501).astype(dtype)
        np.testing.assert_allclose(_j0(arguments), j0(arguments), atol=3e-7 if dtype == np.float32 else 3e-15)
        velocity = np.linspace(-50, 50, 101).astype(dtype)
        kernel = combined_kernel(velocity, 3.1, 6.3, .5, .2, gaussian_sigma(70000.))
        np.testing.assert_allclose(kernel.sum(), 1., atol=2e-7)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=2e-7)
        assert np.all(np.isfinite(kernel))


def test_gaussian_sigma_frozen_convention_and_disabled_ip():
    np.testing.assert_allclose(gaussian_sigma(70000.), 299792.458 / 70000. / 2.354820, rtol=1e-7)
    assert gaussian_sigma(np.inf) == 0


@pytest.mark.parametrize("x64", [False, True])
def test_bessel_weighted_reverse_mode_against_scipy(x64, x64_context):
    # Cover both sides of each approximation boundary, roots, zero, and large
    # arguments reached by fast rotation on a fine velocity grid.
    positive = np.concatenate((np.linspace(0., 5.1, 513),
                               [1e-6, 9e-6, 1e-5, 1.1e-5, np.sqrt(5.783185962946785),
                                4.9999, 5., 5.0001, 30., 100., 1000., 10000.]))
    with x64_context(x64):
        x = jnp.asarray(np.concatenate((-positive, positive)))
        reference_x = np.asarray(x, dtype=np.float64)

        def objective(x, weight):
            values = _j0(x)
            return jnp.sum(weight * values), values

        evaluate = jax.value_and_grad(objective, has_aux=True)
        for function in (evaluate, jax.jit(evaluate)):
            for weight in (1., 1e-2, 1e-4):
                weight = jnp.asarray(weight, dtype=x.dtype)
                (_, values), gradient = function(x, weight)
                assert values.dtype == gradient.dtype == x.dtype
                tolerance = (dict(rtol=2e-12, atol=1e-13) if x64
                             else dict(rtol=2e-5, atol=2e-6))
                np.testing.assert_allclose(values, j0(reference_x), **tolerance)
                # A unit cotangent alone misses underflow in the derivative of
                # the original rational expression's very large denominator.
                np.testing.assert_allclose(np.asarray(gradient) / float(weight),
                                           -j1(reference_x), **tolerance)


def _scipy_rt_kernel_and_derivatives(velocity, speeds, beta):
    """Independent float64 quadrature and analytic derivatives of its integrand.

    Keep Nt=500 to isolate arithmetic/AD accuracy from quadrature convergence.
    SciPy J0 and dJ0/dx=-J1 avoid differentiating the production approximation.
    """
    vmacro, vsini = speeds
    frequency = np.fft.fftfreq(len(velocity), d=np.median(np.diff(velocity)))
    t = np.linspace(0., 1., 500)[:, None]
    t2 = t * t
    mu = np.sqrt(1 - t2)
    limb = (1 - (1 - mu) * (.5 + .2 * (1 - mu))) / (1 - .5 / 3 - .2 / 6)
    a = np.pi**2 * frequency**2
    radial = np.exp(-a * vmacro**2 * (1 - t2))
    tangential = np.exp(-a * vmacro**2 * t2)
    rt = radial + tangential
    dmacro = -2 * a * vmacro * ((1 - t2) * radial + t2 * tangential)
    argument_per_vsini = 2 * np.pi * frequency * t
    rotation = j0(argument_per_vsini * vsini)
    drotation = -j1(argument_per_vsini * vsini) * argument_per_vsini
    common = limb * np.exp(-2 * a * beta**2) * t
    integrands = common * np.stack((rt * rotation, dmacro * rotation, rt * drotation))
    fourier = trapezoid(integrands, axis=1)
    rows = np.fft.fftshift(np.fft.fft(fourier, axis=-1).real, axes=-1)
    area = rows[0].sum()
    kernel = rows[0] / area
    derivatives = (rows[1:] - kernel * rows[1:].sum(axis=-1, keepdims=True)) / area
    return kernel, derivatives


def _kernel_objective(speeds, velocity, beta, weights):
    kernel = combined_kernel(velocity, speeds[0], speeds[1], .5, .2, beta)
    return jnp.sum(kernel * weights), kernel


_kernel_value_and_grad = jax.value_and_grad(_kernel_objective, has_aux=True)
_compiled_kernel_value_and_grad = jax.jit(_kernel_value_and_grad)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("beta", [0., 1.8])
@pytest.mark.parametrize("dv,vmacro,vsini", [
    (.5, 0., 0.), (.25, .01, .01), (.25, 3.1, 6.3),
    (.25, 5.2, 18.7), (.25, 3.1, 30.), (.1, 3.1, 40.),
])
def test_rt_kernel_values_and_gradients_against_scipy(dv, vmacro, vsini, beta,
                                                     x64, x64_context):
    with x64_context(x64):
        half = round(50 / dv)
        velocity = jnp.asarray(np.arange(-half, half + 1) * dv)
        speeds = jnp.asarray([vmacro, vsini])
        beta = jnp.asarray(beta)
        weights = jnp.asarray(np.exp(-.5 * (np.asarray(velocity, dtype=np.float64) / 5)**2))
        expected_kernel, derivatives = _scipy_rt_kernel_and_derivatives(
            np.asarray(velocity, dtype=np.float64), np.asarray(speeds, dtype=np.float64),
            float(beta))
        expected_gradient = derivatives @ np.asarray(weights, dtype=np.float64)
        for function in (_kernel_value_and_grad, _compiled_kernel_value_and_grad):
            (_, kernel), gradient = function(speeds, velocity, beta, weights)
            assert kernel.dtype == gradient.dtype == velocity.dtype
            np.testing.assert_allclose(kernel, expected_kernel, rtol=2e-12 if x64 else 2e-5,
                                       atol=2e-14 if x64 else 2e-7)
            # Near zero width these tiny derivatives involve cancellation in
            # the weighted sum; allow float32 reduction/FFT roundoff as well.
            np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-10 if x64 else 3e-5,
                                       atol=2e-13 if x64 else 5e-8)
