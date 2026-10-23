"""Independent conditional/dense references for the experimental k=1 density."""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from jaxstar.specfit import (
    Observation, chebyshev_basis, gp_conditional_mean, gp_continuum_posterior,
    gp_marginalized_continuum_log_likelihood, marginalized_continuum_log_likelihood,
    prepare_gp_observation,
)
from jaxstar.specfit.likelihood import _vecchia1_factors, _vecchia1_whiten


OPTIONS = dict(gp_amplitude=.07, gp_scale=.6, jitter=.009,
               sigma_constant=.1, sigma_continuum=.03, gp_solver="vecchia1")


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def problem(regions=None):
    wave = 5000 + np.array([0., .01, .03, .07, .12, .2, .4, 1., 2., 4.])
    f = 1 - .3 * np.exp(-.5 * ((wave - 5000.15) / .08)**2)
    y = f * 1.025 + .03 * np.sin(7 * (wave - 5000))
    # A factor of 300 in errors, including adjacent high/low-noise pairs.
    error = np.array([.001, .3, .002, .1, .003, .05, .004, .02, .006, .01])
    mask = np.array([False, False, True, True, False, False, True, True, False, False])
    if regions:
        wave = np.stack([wave + 1000 * i for i in range(regions)])
        f, y, error, mask = [np.broadcast_to(a, wave.shape).copy() for a in (f, y, error, mask)]
        mask[1, 4] = True  # Different retained counts; do not pad as real observations.
    y[mask], error[mask] = np.nan, np.inf
    return Observation(wave, y, error, mask), jnp.asarray(f)


def reference_factors(wave, error, amplitude, scale, jitter):
    variance = amplitude**2 + error**2 + jitter**2
    t = np.sqrt(3.) * np.diff(wave) / scale
    adjacent = amplitude**2 * (1 + t) * np.exp(-t)
    b = adjacent / variance[:-1]
    v = np.r_[variance[0], variance[1:] - adjacent**2 / variance[:-1]]
    return b, v


@pytest.mark.parametrize("pixels", [1, 2, 10])
@pytest.mark.parametrize("jitter", [0., .009])
def test_normalized_conditional_product_with_heteroscedastic_errors(pixels, jitter):
    obs, _ = problem()
    obs = Observation(obs.wavelength[:pixels], obs.flux[:pixels] - 1,
                      obs.uncertainty[:pixels], obs.mask[:pixels])
    valid = obs.valid
    r = obs.flux[valid]
    b, v = reference_factors(obs.wavelength[valid], obs.uncertainty[valid], .07, .6, jitter)
    expected = norm.logpdf(r[0], scale=np.sqrt(v[0]))
    expected += norm.logpdf(r[1:], loc=b*r[:-1], scale=np.sqrt(v[1:])).sum()
    # Zero model makes all continuum columns zero, isolating the noise density.
    actual = gp_marginalized_continuum_log_likelihood(
        obs, jnp.zeros(obs.shape), **dict(OPTIONS, jitter=jitter))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_adjacent_covariance_matches_tinygp_and_uses_mask_gaps():
    import tinygp
    obs, _ = problem()
    x, e = jnp.asarray(obs.wavelength[obs.valid]), jnp.asarray(obs.uncertainty[obs.valid])
    b, v = _vecchia1_factors(x, e, .07, .6, .009)
    k = .07**2 * tinygp.kernels.Matern32(scale=.6)
    c = np.diag(np.asarray(k(x, x)), k=-1)
    variance = .07**2 + np.asarray(e)**2 + .009**2
    np.testing.assert_allclose(b * variance[:-1], c, rtol=2e-14, atol=1e-18)
    np.testing.assert_allclose(v[1:], variance[1:] - c**2/variance[:-1], rtol=2e-13)
    assert c[-1] < c[0] / 20  # Actual 2 Angstrom gap, not one pixel step.


def dense_reference(obs, flux, degree, options):
    """Build the covariance implied by the conditionals, then integrate X c."""
    basis = np.asarray(chebyshev_basis(obs.wavelength, degree))
    basis = basis.reshape(obs.n_regions, obs.n_pixels, degree+1)
    jitters, constants, slopes = [np.broadcast_to(options[name], (obs.n_regions,))
                                  for name in ("jitter", "sigma_constant", "sigma_continuum")]
    total, means, covariances = 0., [], []
    for i in range(obs.n_regions):
        valid = np.atleast_2d(obs.valid)[i]
        mean = np.r_[1., np.zeros(degree)]
        prior = np.diag(np.r_[constants[i], np.full(degree, slopes[i])]**2)
        if not valid.any():
            means.append(mean)
            covariances.append(prior)
            continue
        x, y, e = [np.atleast_2d(a)[i, valid] for a in (obs.wavelength, obs.flux, obs.uncertainty)]
        b, v = reference_factors(x, e, options["gp_amplitude"], options["gp_scale"], jitters[i])
        transition = np.eye(len(x)) - np.diag(b, k=-1)
        inverse = np.linalg.inv(transition)
        covariance = inverse @ np.diag(v) @ inverse.T
        design = np.atleast_2d(flux)[i, valid, None] * basis[i, valid]
        full = covariance + design @ prior @ design.T
        residual = y - design @ mean
        solved = np.linalg.solve(full, residual)
        total -= .5 * (residual @ solved + np.linalg.slogdet(full)[1] + len(x)*np.log(2*np.pi))
        cross = prior @ design.T
        means.append(mean + cross @ solved)
        covariances.append(prior - cross @ np.linalg.solve(full, cross.T))
    return total, np.asarray(means), np.asarray(covariances)


@pytest.mark.parametrize("degree", [0, 4])
@pytest.mark.parametrize("regions", [None, 3])
def test_marginalization_and_posterior_against_explicit_covariance(degree, regions):
    obs, flux = problem(regions)
    options = dict(OPTIONS)
    if regions:
        mask = obs.mask.copy()
        mask[2] = True
        obs = Observation(obs.wavelength, obs.flux, obs.uncertainty, mask)
        options.update(jitter=[.009, .02, .04], sigma_constant=[.1, .2, .3],
                       sigma_continuum=[.03, .01, .02])
    expected, mean, covariance = dense_reference(obs, flux, degree, options)
    dirty = jnp.where(obs.mask, jnp.nan, flux)
    prepared = prepare_gp_observation(obs)
    fn = lambda data, f: gp_marginalized_continuum_log_likelihood(data, f, degree=degree, **options)
    value, gradient = jax.jit(jax.value_and_grad(fn, argnums=1))(prepared, dirty)
    post = jax.jit(lambda data, f: gp_continuum_posterior(data, f, degree=degree, **options))(prepared, dirty)
    np.testing.assert_allclose(value, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(post.mean, mean[0] if regions is None else mean, atol=2e-12)
    np.testing.assert_allclose(post.covariance, covariance[0] if regions is None else covariance, atol=2e-12)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[obs.mask], 0)


def test_gradients_against_finite_differences_and_dynamic_data():
    obs, flux = problem()
    prepared = prepare_gp_observation(obs)
    def objective(theta, data):
        a, scale, jitter, s0, sc = jnp.exp(theta[1:])
        model = flux + theta[0] * jnp.exp(-((jnp.asarray(obs.wavelength)-5000.15)/.08)**2)
        return gp_marginalized_continuum_log_likelihood(
            data, model, gp_amplitude=a, gp_scale=scale, jitter=jitter,
            sigma_constant=s0, sigma_continuum=sc, gp_solver="vecchia1")
    theta = jnp.r_[.015, jnp.log(jnp.array([.07, .6, .009, .1, .03]))]
    fn = jax.jit(objective)
    value, gradient = jax.jit(jax.value_and_grad(objective))(theta, prepared)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    for i in range(len(theta)):
        step = jnp.eye(len(theta))[i] * 1e-5
        finite = (fn(theta+step, prepared)-fn(theta-step, prepared)) / 2e-5
        np.testing.assert_allclose(gradient[i], finite, rtol=3e-6, atol=3e-6)
    changed = prepare_gp_observation(Observation(obs.wavelength, obs.flux+.01, obs.uncertainty, obs.mask))
    assert not np.isclose(fn(theta, changed), value)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_near_degenerate_variance_has_no_subtractive_cancellation(dtype):
    x = jnp.array([0., 1e-10, 2e-10], dtype=dtype)
    errors = jnp.full(3, 1e-12, dtype=dtype)
    b, v = jax.jit(_vecchia1_factors)(x, errors, dtype(1.), dtype(1.), dtype(0.))
    with localcontext() as context:
        context.prec = 70
        t = Decimal(3).sqrt() * Decimal(float(x[1]))
        rho = (1+t) * (-t).exp()
        variance = 1 + Decimal(float(errors[0]))**2
        expected = float(variance - rho**2/variance)
    np.testing.assert_allclose(v[1:], expected, rtol=3e-7 if dtype == jnp.float32 else 2e-15)
    assert np.all(v > 0)
    assert v.dtype == dtype and b.dtype == dtype


def test_white_limit_empty_regions_and_no_tinygp_dependency(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_tinygp(name, *args, **kwargs):
        if name == "tinygp":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_tinygp)
    obs, flux = problem(2)
    options = dict(sigma_constant=.1, sigma_continuum=.03, jitter=[.009, .02])
    actual = gp_marginalized_continuum_log_likelihood(
        obs, flux, gp_amplitude=1e-12, gp_scale=.6, gp_solver="vecchia1", **options)
    expected = marginalized_continuum_log_likelihood(obs, flux, **options)
    np.testing.assert_allclose(actual, expected, atol=2e-11)
    empty = Observation(obs.wavelength, np.full(obs.shape, np.nan), np.full(obs.shape, np.nan),
                        np.ones(obs.shape, dtype=bool))
    assert gp_marginalized_continuum_log_likelihood(empty, flux, **OPTIONS) == 0
    post = gp_continuum_posterior(empty, flux, **OPTIONS)
    np.testing.assert_array_equal(post.mean, np.tile([1., 0., 0., 0., 0.], (2, 1)))


@pytest.mark.parametrize("name,value", [("gp_amplitude", -.01), ("gp_scale", 0.), ("jitter", -.01)])
def test_invalid_parameters_are_not_hidden(name, value):
    obs, flux = problem()
    with pytest.raises(ValueError, match=name):
        gp_marginalized_continuum_log_likelihood(obs, flux, **dict(OPTIONS, **{name: value}))


@pytest.mark.parametrize("fully_masked", [False, True])
def test_latent_prediction_is_not_defined(fully_masked):
    obs, flux = problem()
    if fully_masked:
        obs = Observation(obs.wavelength, obs.flux, obs.uncertainty,
                          np.ones(obs.shape, dtype=bool))
    with pytest.raises(ValueError, match="does not define a latent GP conditional mean"):
        gp_conditional_mean(obs, flux, gp_amplitude=.07, gp_scale=.6, gp_solver="vecchia1")


def test_whitening_has_only_linear_storage_and_no_scan():
    def whiten(x, e, rhs):
        b, v = _vecchia1_factors(x, e, .07, .6, .009)
        return _vecchia1_whiten(rhs, b, v)
    graph = jax.make_jaxpr(whiten)(jnp.arange(31.), jnp.full(31, .01), jnp.ones((31, 6)))
    assert "scan[" not in str(graph)
    for eqn in graph.jaxpr.eqns:
        for var in eqn.outvars:
            assert getattr(var.aval, "shape", None) != (31, 31)
