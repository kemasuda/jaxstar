"""Chebyshev, normalized Gaussian integration, conditional recovery and AD."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.specfit import (
    Observation, SpecModel, apply_continuum, chebyshev_basis,
    continuum_design_matrix, continuum_posterior, continuum_prior,
    evaluate_continuum, marginalized_continuum_log_likelihood,
)


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def problem(pixels=13, regions=None, dtype=np.float64):
    x = np.linspace(-1, 1, pixels)
    wave = (5000 + 2 * x).astype(dtype)
    model = (1 - .3 * np.exp(-.5 * (x / .15)**2)).astype(dtype)
    flux = (model * (1.04 + .025 * x - .01 * (2*x**2 - 1)) + .003 * np.sin(5*x)).astype(dtype)
    error = (.02 + .003 * (x + 1)).astype(dtype)
    mask = np.arange(pixels) % 5 == 2
    if regions is not None:
        wave = np.stack([wave + 1000*i for i in range(regions)])
        model = np.stack([model * (1 - .03*i) for i in range(regions)])
        flux = np.stack([flux * (1 - .03*i) for i in range(regions)])
        error, mask = np.broadcast_to(error, wave.shape), np.broadcast_to(mask, wave.shape)
    return Observation(wave, flux, error, mask), jnp.asarray(model)


def dense_reference(obs, model, degree, s0, sc, jitter=0.):
    basis = np.asarray(chebyshev_basis(obs.wavelength, degree))
    model = np.asarray(model)
    region_count = obs.n_regions
    s0, sc = np.broadcast_to(s0, (region_count,)), np.broadcast_to(sc, (region_count,))
    jitter = np.broadcast_to(jitter, (region_count,))
    if obs.ndim == 1:
        basis, model = basis[None], model[None]
    values, means, covariances = [], [], []
    for i in range(region_count):
        usable = np.atleast_2d(obs.valid)[i]
        y = np.atleast_2d(obs.flux)[i, usable]
        error = np.atleast_2d(obs.uncertainty)[i, usable]
        error = np.hypot(error, jitter[i])
        design = model[i, usable, None] * basis[i, usable]
        mean = np.zeros(degree+1)
        mean[0] = 1
        scales = np.array([s0[i]] + [sc[i]]*degree)
        prior_covariance = np.diag(scales**2)
        covariance = np.diag(error**2) + design @ prior_covariance @ design.T
        residual = y - design @ mean
        logdet = np.linalg.slogdet(covariance)[1]
        values.append(-.5*(residual @ np.linalg.solve(covariance, residual)
                            + logdet + len(y)*np.log(2*np.pi)))
        precision = np.diag(1/scales**2) + design.T @ (design/error[:, None]**2)
        covariances.append(np.linalg.solve(precision, np.eye(degree+1)))
        means.append(np.linalg.solve(precision, mean/scales**2 + design.T @ (y/error**2)))
    means, covariances = np.array(means), np.array(covariances)
    return sum(values), means[0] if obs.ndim == 1 else means, covariances[0] if obs.ndim == 1 else covariances


@pytest.mark.parametrize("degree", [0, 1, 4, 7])
@pytest.mark.parametrize("regions", [None, 1, 3])
def test_basis_exact_values_per_region(degree, regions):
    obs, _ = problem(pixels=5, regions=regions)
    basis = chebyshev_basis(obs.wavelength, degree)
    x = np.linspace(-1, 1, 5)
    expected = np.polynomial.chebyshev.chebvander(x, degree)
    if regions is not None:
        expected = np.broadcast_to(expected, (regions,)+expected.shape)
    np.testing.assert_allclose(basis, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(jax.jit(lambda w: chebyshev_basis(w, degree))(obs.wavelength), basis)
    assert basis.shape == obs.shape + (degree+1,)


def test_irregular_wavelength_endpoints_and_single_pixel():
    wave = np.array([[5000., 5001., 5004.], [6000., 6008., 6010.]])
    expected_x = np.array([[-1., -.5, 1.], [-1., .6, 1.]])
    np.testing.assert_allclose(chebyshev_basis(wave, 1)[..., 1], expected_x)
    np.testing.assert_array_equal(chebyshev_basis([5000.], 4), [[1., 0., -1., 0., 1.]])
    gradient = jax.grad(lambda w: chebyshev_basis(w, 4)[..., 2:].sum())(jnp.asarray(wave))
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("regions", [None, 1, 3])
def test_explicit_continuum_and_design(regions):
    obs, flux = problem(regions=regions)
    basis = chebyshev_basis(obs.wavelength)
    coefficients = np.array([1.1, -.02, .03, 0., .01])
    if regions is not None:
        coefficients = np.broadcast_to(coefficients, (regions, 5)).copy()
        coefficients[:, 0] += np.arange(regions)*.1
    expected = np.sum(np.asarray(basis)*coefficients[..., None, :], axis=-1)
    np.testing.assert_allclose(evaluate_continuum(basis, coefficients), expected)
    np.testing.assert_allclose(apply_continuum(flux, basis, coefficients), flux*expected)
    design = continuum_design_matrix(flux, basis)
    np.testing.assert_allclose(design, flux[..., None]*basis)
    np.testing.assert_allclose(np.sum(design*coefficients[..., None, :], axis=-1), flux*expected)
    # Unconstrained linear Gaussian coefficients; never enforce positivity.
    np.testing.assert_array_equal(evaluate_continuum(basis[..., :1], -np.ones(coefficients.shape[:-1]+(1,))), -np.ones(obs.shape))


def test_constant_only_and_prior_convention():
    obs, model = problem()
    basis = chebyshev_basis(obs.wavelength, 0)
    np.testing.assert_array_equal(apply_continuum(model, basis, [1.23]), model*1.23)
    prior = continuum_prior(sigma_constant=.2, sigma_continuum=.03)
    np.testing.assert_array_equal(prior.mean, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(prior.scale, [.2, .03, .03, .03, .03])
    np.testing.assert_allclose(prior.covariance, np.diag(prior.scale**2))
    multi = continuum_prior(sigma_constant=[.1, .2], sigma_continuum=.03, n_regions=2)
    np.testing.assert_array_equal(multi.scale[:, 0], [.1, .2])
    np.testing.assert_array_equal(multi.scale[:, 1:], np.full((2, 4), .03))


@pytest.mark.parametrize("degree", [0, 1, 4, 8])
@pytest.mark.parametrize("pixels", [3, 17])
@pytest.mark.parametrize("regions", [None, 3])
@pytest.mark.parametrize("scales", [(.1, .02), (1., .3), (1e-5, 1e-6)])
def test_normalized_marginal_and_posterior_against_dense(degree, pixels, regions, scales):
    obs, model = problem(pixels, regions)
    s0, sc = scales
    if regions:
        s0, sc = s0*np.array([1., 1.3, .7]), sc*np.array([1.2, 1., .8])
    expected, mean, covariance = dense_reference(obs, model, degree, s0, sc)
    options = dict(sigma_constant=s0, sigma_continuum=sc, degree=degree)
    logp = marginalized_continuum_log_likelihood(obs, model, **options)
    posterior = continuum_posterior(obs, model, **options)
    np.testing.assert_allclose(logp, expected, rtol=3e-11, atol=3e-10)
    np.testing.assert_allclose(posterior.mean, mean, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(posterior.covariance, covariance, rtol=3e-10, atol=3e-12)
    np.testing.assert_allclose(posterior.covariance, np.swapaxes(posterior.covariance, -1, -2), atol=1e-14)
    assert np.all(np.linalg.eigvalsh(posterior.covariance) > 0)
    assert evaluate_continuum(chebyshev_basis(obs.wavelength, degree), posterior.mean).shape == obs.shape


def test_broad_and_tight_priors_and_region_independence():
    obs, model = problem(regions=3)
    basis = np.asarray(chebyshev_basis(obs.wavelength))
    broad = continuum_posterior(obs, model, sigma_constant=100., sigma_continuum=100.)
    for i in range(3):
        valid = obs.valid[i]
        design = np.asarray(model)[i, valid, None]*basis[i, valid]
        expected = np.linalg.lstsq(design/obs.uncertainty[i, valid, None], obs.flux[i, valid]/obs.uncertainty[i, valid], rcond=None)[0]
        np.testing.assert_allclose(broad.mean[i], expected, rtol=1e-7, atol=1e-8)
    tight = continuum_posterior(obs, model, sigma_constant=1e-10, sigma_continuum=1e-10)
    np.testing.assert_allclose(tight.mean, np.broadcast_to([1, 0, 0, 0, 0], (3, 5)), rtol=0, atol=1e-14)
    options = dict(sigma_constant=.1, sigma_continuum=.03)
    total = marginalized_continuum_log_likelihood(obs, model, **options)
    sum_regions = 0.
    for i in range(3):
        single = Observation(obs.wavelength[i], obs.flux[i], obs.uncertainty[i], obs.mask[i])
        sum_regions += marginalized_continuum_log_likelihood(single, model[i], **options)
    np.testing.assert_allclose(total, sum_regions, rtol=0, atol=1e-11)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_masked_nan_independence_jit_and_gradients(dtype):
    original, model = problem(regions=2, dtype=dtype)
    flux, error = original.flux.copy(), original.uncertainty.copy()
    flux[original.mask] = np.nan
    error[original.mask] = np.inf
    dirty = Observation(original.wavelength, flux, error, original.mask)
    flux[original.mask] = np.inf
    error[original.mask] = -1
    changed = Observation(original.wavelength, flux, error, original.mask)
    error[original.mask] = np.nan
    nan_error = Observation(original.wavelength, flux, error, original.mask)
    fn = lambda o, f, s0, sc: marginalized_continuum_log_likelihood(o, f, sigma_constant=s0, sigma_continuum=sc)
    posterior = lambda o, f: continuum_posterior(o, f, sigma_constant=.1, sigma_continuum=.03)
    for o in (original, dirty, changed, nan_error):
        np.testing.assert_array_equal(fn(o, model, .1, .03), fn(original, model, .1, .03))
        p = jax.jit(posterior)(o, model)
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(p))
    np.testing.assert_allclose(jax.jit(fn)(dirty, model, .1, .03), fn(dirty, model, .1, .03), rtol=2e-6)
    dirty_model = jnp.where(original.mask, jnp.nan, model)
    value, gradient = jax.jit(jax.value_and_grad(fn, argnums=(1, 2, 3)))(dirty, dirty_model, .1, .03)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(x)) for x in gradient)
    np.testing.assert_array_equal(gradient[0][original.mask], 0)
    expected, mean, covariance = dense_reference(original, model, 4, .1, .03)
    np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-5)
    p = posterior(dirty, model)
    np.testing.assert_allclose(p.mean, mean, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(p.covariance, covariance, rtol=2e-6, atol=1e-9)


def test_all_masked_returns_prior_and_zero_likelihood():
    obs = Observation([5000., 5001.], [np.nan, np.inf], [np.nan, -1.], [True, True])
    prior = continuum_prior(sigma_constant=.1, sigma_continuum=.03)
    options = dict(sigma_constant=.1, sigma_continuum=.03)
    np.testing.assert_array_equal(marginalized_continuum_log_likelihood(obs, [np.nan, np.inf], **options), 0.)
    posterior = continuum_posterior(obs, [np.nan, np.inf], **options)
    np.testing.assert_array_equal(posterior.mean, prior.mean)
    np.testing.assert_array_equal(posterior.covariance, prior.covariance)
    single = Observation([5000.], [1.], [.01])
    assert np.isfinite(marginalized_continuum_log_likelihood(single, [1.], **options))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_cached_basis_vmap_and_precision_unchanged(dtype):
    obs, model = problem(regions=2, dtype=dtype)
    before = jax.config.x64_enabled
    basis = chebyshev_basis(obs.wavelength)
    options = dict(sigma_constant=.1, sigma_continuum=.03)
    logp = marginalized_continuum_log_likelihood(obs, model, basis=basis, **options)
    posterior = continuum_posterior(obs, model, basis=basis, **options)
    assert logp.dtype == dtype
    assert posterior.mean.dtype == dtype and posterior.covariance.dtype == dtype
    fn = lambda f, s0, sc: marginalized_continuum_log_likelihood(obs, f, basis=basis, sigma_constant=s0, sigma_continuum=sc)
    models = jnp.stack([model, model*.99, model*1.01])
    scales = jnp.array([[.1, .2], [.15, .25], [.2, .3]], dtype=dtype)
    actual = jax.jit(jax.vmap(fn))(models, scales, scales*.2)
    expected = jnp.stack([fn(f, s0, s0*.2) for f, s0 in zip(models, scales)])
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    batched_posterior = jax.jit(jax.vmap(lambda f: continuum_posterior(obs, f, **options)))(models)
    assert batched_posterior.mean.shape == (3, 2, 5)
    assert jax.config.x64_enabled == before


def test_helpers_do_not_enable_x64(x64_context):
    with x64_context(False):
        obs, model = problem(dtype=np.float32)
        value = marginalized_continuum_log_likelihood(obs, model, sigma_constant=.1, sigma_continuum=.03)
        assert value.dtype == np.float32
        assert not jax.config.x64_enabled


def test_gradients_against_finite_differences():
    obs, model = problem()
    fn = lambda f, s0, sc: marginalized_continuum_log_likelihood(obs, f, sigma_constant=s0, sigma_continuum=sc)
    gradient = jax.grad(fn, argnums=(0, 1, 2))(model, .1, .03)
    h = 1e-6
    for i in (0, 5, 8):
        finite = (fn(model.at[i].add(h), .1, .03) - fn(model.at[i].add(-h), .1, .03))/(2*h)
        np.testing.assert_allclose(gradient[0][i], finite, rtol=1e-7, atol=1e-7)
    for index, scale in ((1, .1), (2, .03)):
        args_plus, args_minus = [model, .1, .03], [model, .1, .03]
        args_plus[index], args_minus[index] = scale+h, scale-h
        finite = (fn(*args_plus) - fn(*args_minus))/(2*h)
        np.testing.assert_allclose(gradient[index], finite, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_degree_zero_strong_continuum_avoids_quadratic_cancellation(dtype):
    pixels, error, scale = 100, .001, .5
    wave = np.linspace(5000., 5001., pixels).astype(dtype)
    model = jnp.ones(pixels, dtype=dtype)
    obs = Observation(wave, np.full(pixels, 1.2, dtype=dtype), np.full(pixels, error, dtype=dtype))
    # Exact rank-one Gaussian result; the residual lies along the constant mode.
    offset = float(obs.flux[0])-1.
    quadratic = offset**2*pixels/(error**2+pixels*scale**2)
    logdet = (pixels-1)*np.log(error**2) + np.log(error**2+pixels*scale**2)
    expected = -.5*(quadratic+logdet+pixels*np.log(2*np.pi))
    value = marginalized_continuum_log_likelihood(obs, model, degree=0,
                                                sigma_constant=scale, sigma_continuum=.02)
    np.testing.assert_allclose(value, expected, rtol=2e-7, atol=2e-5)


@pytest.mark.parametrize("jitter", [0., .005])
def test_end_to_end_specmodel_eager_jit_physical_gradients(jitter):
    from test_specmodel import make_library, parameters

    library = make_library(pixels=513)
    model, params = SpecModel(library), parameters()
    second = copy.deepcopy(params["components"][0])
    second["atmosphere"]["custom_temperature"] = .7
    second["rv"] = 3.2
    params.update(components=params["components"]+(second,), flux_weights=jnp.array([1., .5]))
    wave = np.asarray(library.wavelength)[:, 180:333:2]
    physical = model(params, wave)
    basis = chebyshev_basis(wave)
    coefficients = jnp.array([[1.02, .01, -.005, 0., .002], [.98, -.02, 0., .001, 0.]])
    obs = Observation(wave, apply_continuum(physical, basis, coefficients), jnp.full(wave.shape, .01))
    fn = lambda m, p, o, b: marginalized_continuum_log_likelihood(o, m(p, o.wavelength), basis=b,
                                                               sigma_constant=.1, sigma_continuum=.03, jitter=jitter)
    eager = fn(model, params, obs, basis)
    compiled = jax.jit(fn)(model, params, obs, basis)
    np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-10)
    value, gradient = jax.jit(jax.value_and_grad(fn, argnums=1))(model, params, obs, basis)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))


@pytest.mark.parametrize("degree", [-1, 1.5, True])
def test_degree_validation(degree):
    with pytest.raises(ValueError, match="degree"):
        chebyshev_basis([1., 2.], degree)
    with pytest.raises(ValueError, match="degree"):
        continuum_prior(sigma_constant=.1, sigma_continuum=.02, degree=degree)


@pytest.mark.parametrize("wave", [[], [[1., 2.]], [1., np.nan], [0., 1.], [2., 1.], [1., 1.]])
def test_wavelength_validation(wave):
    if wave == [[1., 2.]]:
        wave = np.ones((1, 2, 2))
    with pytest.raises(ValueError, match="wavelength"):
        chebyshev_basis(wave)


@pytest.mark.parametrize("name", ["sigma_constant", "sigma_continuum"])
@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_prior_scale_validation(name, value):
    options = dict(sigma_constant=.1, sigma_continuum=.03)
    options[name] = value
    with pytest.raises(ValueError, match=name):
        continuum_prior(**options)
    obs, model = problem()
    with pytest.raises(ValueError, match=name):
        marginalized_continuum_log_likelihood(obs, model, **options)


def test_jit_invalid_scale_clear_error():
    obs, model = problem()
    fn = jax.jit(lambda s: marginalized_continuum_log_likelihood(obs, model, sigma_constant=s, sigma_continuum=.02))
    with pytest.raises(Exception, match="sigma_constant must be finite and positive"):
        fn(-1.).block_until_ready()


def test_shape_errors_are_explicit():
    obs, model = problem(regions=2)
    basis = chebyshev_basis(obs.wavelength)
    with pytest.raises(ValueError, match="coefficients"):
        evaluate_continuum(basis, np.ones(5))
    with pytest.raises(ValueError, match="coefficients"):
        apply_continuum(model, basis, np.ones((2, 4)))
    with pytest.raises(ValueError, match="model_flux"):
        continuum_design_matrix(model[0], basis)
    with pytest.raises(ValueError, match="model_flux"):
        apply_continuum(model[0], basis, np.ones((2, 5)))
    with pytest.raises(ValueError, match="sigma_constant"):
        continuum_prior(sigma_constant=[.1, .2, .3], sigma_continuum=.02, n_regions=2)
    with pytest.raises(ValueError, match="n_regions"):
        continuum_prior(sigma_constant=.1, sigma_continuum=.02, n_regions=0)
    with pytest.raises(ValueError, match="basis"):
        evaluate_continuum(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="basis"):
        marginalized_continuum_log_likelihood(obs, model, basis=basis[..., :4], sigma_constant=.1, sigma_continuum=.03)
    with pytest.raises(ValueError, match="model_flux"):
        continuum_posterior(obs, model[0], sigma_constant=.1, sigma_continuum=.03)
    with pytest.raises(TypeError, match="Observation"):
        continuum_posterior(obs.flux, model, sigma_constant=.1, sigma_continuum=.03)


def test_compiled_coefficient_matrices_have_no_pixel_covariance():
    obs, model = problem(pixels=37, regions=2)
    fn = lambda o, f: (marginalized_continuum_log_likelihood(o, f, sigma_constant=.1, sigma_continuum=.03),
                       continuum_posterior(o, f, sigma_constant=.1, sigma_continuum=.03))
    program = jax.make_jaxpr(fn)(obs, model)
    def shapes(jaxpr):
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        for equation in jaxpr.eqns:
            for variable in equation.outvars:
                yield getattr(variable.aval, "shape", ())
            for parameter in equation.params.values():
                if hasattr(parameter, "jaxpr") or hasattr(parameter, "eqns"):
                    yield from shapes(parameter)
                elif isinstance(parameter, (tuple, list)):
                    for item in parameter:
                        if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                            yield from shapes(item)
    seen = list(shapes(program))
    assert (2, 5, 5) in seen
    assert not any(shape.count(37) > 1 for shape in seen)


@pytest.mark.parametrize("dtype, reference", [(np.float32, 43.680233001708984),
                                              (np.float64, 43.67998215950677)])
def test_zero_jitter_committed_reference_and_default_parity(dtype, reference):
    # Captured from 24b690a, before adding jitter, for the same generated fixture.
    obs, model = problem(regions=2, dtype=dtype)
    options = dict(sigma_constant=.1, sigma_continuum=.03)
    default = marginalized_continuum_log_likelihood(obs, model, **options)
    np.testing.assert_allclose(default, reference, rtol=1e-7, atol=1e-11)
    original_posterior = continuum_posterior(obs, model, **options)
    for jitter in (0., np.zeros(2, dtype=dtype)):
        np.testing.assert_array_equal(marginalized_continuum_log_likelihood(obs, model, jitter=jitter, **options), default)
        posterior = continuum_posterior(obs, model, jitter=jitter, **options)
        np.testing.assert_array_equal(posterior.mean, original_posterior.mean)
        np.testing.assert_array_equal(posterior.covariance, original_posterior.covariance)


@pytest.mark.parametrize("degree", [0, 4])
@pytest.mark.parametrize("pixels", [3, 17])
@pytest.mark.parametrize("regions", [None, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonzero_jitter_dense_likelihood_and_posterior(degree, pixels, regions, dtype):
    obs, model = problem(pixels, regions, dtype)
    jitter = .025 if regions is None else np.array([.005, .04], dtype=dtype)
    s0 = .1 if regions is None else np.array([.1, .15], dtype=dtype)
    options = dict(sigma_constant=s0, sigma_continuum=.03, degree=degree, jitter=jitter)
    expected, mean, covariance = dense_reference(obs, model, degree, s0, .03, jitter)
    value = marginalized_continuum_log_likelihood(obs, model, **options)
    posterior = continuum_posterior(obs, model, **options)
    tolerance = dict(rtol=3e-6, atol=3e-6) if dtype == np.float32 else dict(rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(value, expected, **tolerance)
    np.testing.assert_allclose(posterior.mean, mean, **tolerance)
    np.testing.assert_allclose(posterior.covariance, covariance, **tolerance)
    assert value.dtype == dtype and posterior.mean.dtype == dtype


def test_jitter_normalization_penalty_with_exactly_zero_residuals():
    # X=0 and y=0 eliminate every residual/prior term: only logdet(D) changes.
    obs = Observation(np.linspace(5000., 5002., 9), np.zeros(9), np.full(9, .01))
    model = jnp.zeros(9)
    values = []
    for jitter in (0., .02, .2):
        value = marginalized_continuum_log_likelihood(obs, model, sigma_constant=.1, sigma_continuum=.03, jitter=jitter)
        expected = -.5*9*np.log(2*np.pi*(.01**2+jitter**2))
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)
        values.append(value)
    assert values[0] > values[1] > values[2]


def test_scalar_jitter_region_equivalence_posterior_broadening_and_flux_units():
    obs, model = problem(regions=2)
    options = dict(sigma_constant=.1, sigma_continuum=.03)
    value = marginalized_continuum_log_likelihood(obs, model, jitter=.02, **options)
    np.testing.assert_array_equal(value, marginalized_continuum_log_likelihood(obs, model, jitter=[.02, .02], **options))
    zero = continuum_posterior(obs, model, jitter=0., **options)
    nonzero = continuum_posterior(obs, model, jitter=[.02, .04], **options)
    assert np.all(np.linalg.eigvalsh(nonzero.covariance-zero.covariance) > 0)
    assert not np.allclose(zero.mean, nonzero.mean, rtol=1e-7, atol=1e-7)
    noisy = continuum_posterior(obs, model, jitter=1e4, **options)
    prior = continuum_prior(n_regions=2, **options)
    np.testing.assert_allclose(noisy.mean, prior.mean, atol=1e-9)
    np.testing.assert_allclose(noisy.covariance, prior.covariance, atol=1e-9)
    # Absolute flux-unit jitter rescales with the data, independently of model f.
    scale = 7.
    scaled = Observation(obs.wavelength, obs.flux*scale, obs.uncertainty*scale, obs.mask)
    scaled_value = marginalized_continuum_log_likelihood(scaled, model*scale, jitter=.02*scale, **options)
    np.testing.assert_allclose(scaled_value, value-np.sum(obs.valid)*np.log(scale), rtol=1e-12)
    scaled_posterior = continuum_posterior(scaled, model*scale, jitter=.02*scale, **options)
    original = continuum_posterior(obs, model, jitter=.02, **options)
    np.testing.assert_allclose(scaled_posterior.mean, original.mean, atol=1e-13)
    np.testing.assert_allclose(scaled_posterior.covariance, original.covariance, atol=1e-13)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jitter_masked_nan_all_masked_region_and_gradients(dtype):
    original, model = problem(regions=2, dtype=dtype)
    mask = original.mask.copy()
    mask[1] = True
    flux, error = original.flux.copy(), original.uncertainty.copy()
    flux[mask], error[mask] = np.nan, np.inf
    dirty = Observation(original.wavelength, flux, error, mask)
    flux[mask], error[mask] = np.inf, np.nan
    changed = Observation(original.wavelength, flux, error, mask)
    fn = lambda o,f,s0,sc,j: marginalized_continuum_log_likelihood(o,f,sigma_constant=s0,sigma_continuum=sc,jitter=j)
    jitter = jnp.array([.015, .03], dtype=dtype)
    value, gradient = jax.jit(jax.value_and_grad(fn, argnums=(1, 2, 3, 4)))(dirty, model, .1, .03, jitter)
    assert np.isfinite(value) and all(np.all(np.isfinite(x)) for x in gradient)
    np.testing.assert_array_equal(gradient[0][mask], 0)
    np.testing.assert_array_equal(gradient[3][1], 0)
    np.testing.assert_array_equal(fn(dirty, model, .1, .03, jitter), fn(changed, model, .1, .03, jitter))
    expected, mean, covariance = dense_reference(dirty, model, 4, .1, .03, jitter)
    tolerance = dict(rtol=3e-6, atol=3e-6) if dtype == np.float32 else dict(rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(value, expected, **tolerance)
    posterior = jax.jit(lambda o,f,j: continuum_posterior(o,f,sigma_constant=.1,sigma_continuum=.03,jitter=j))(changed,model,jitter)
    np.testing.assert_allclose(posterior.mean, mean, **tolerance)
    np.testing.assert_allclose(posterior.covariance, covariance, **tolerance)
    prior = continuum_prior(sigma_constant=jnp.asarray(.1, dtype=dtype),
                            sigma_continuum=jnp.asarray(.03, dtype=dtype))
    np.testing.assert_array_equal(posterior.mean[1], prior.mean)
    np.testing.assert_array_equal(posterior.covariance[1], prior.covariance)


def test_jitter_jit_grad_vmap_and_finite_difference():
    obs, model = problem(regions=2)
    fn = lambda f,s0,sc,j: marginalized_continuum_log_likelihood(obs,f,sigma_constant=s0,sigma_continuum=sc,jitter=j)
    value, gradients = jax.jit(jax.value_and_grad(fn,argnums=(0, 1, 2, 3)))(model,.1,.03,.02)
    assert np.isfinite(value) and all(np.all(np.isfinite(x)) for x in gradients)
    step = 1e-6
    finite = (fn(model,.1,.03,.02+step)-fn(model,.1,.03,.02-step))/(2*step)
    np.testing.assert_allclose(gradients[3],finite,rtol=1e-7,atol=1e-6)
    np.testing.assert_array_equal(jax.grad(fn,argnums=3)(model,.1,.03,0.),0.)
    jitters = jnp.array([[.005,.01],[.015,.02],[.025,.03]])
    compiled = jax.jit(jax.vmap(lambda j: fn(model,.1,.03,j)))(jitters)
    expected = jnp.stack([fn(model,.1,.03,j) for j in jitters])
    np.testing.assert_allclose(compiled,expected,rtol=1e-12,atol=1e-12)
    posterior = jax.jit(jax.vmap(lambda j: continuum_posterior(obs,model,sigma_constant=.1,sigma_continuum=.03,jitter=j)))(jitters)
    assert posterior.mean.shape == (3,2,5) and np.all(np.isfinite(posterior.covariance))


@pytest.mark.parametrize("jitter", [-.01, np.nan, np.inf, [0., .01, .02], np.ones((2,13)), np.ones((2,1))])
def test_jitter_validation(jitter):
    obs, model = problem(regions=2)
    for function in (marginalized_continuum_log_likelihood, continuum_posterior):
        with pytest.raises(ValueError, match="jitter"):
            function(obs, model, sigma_constant=.1, sigma_continuum=.03, jitter=jitter)


def test_jitter_jit_validation():
    obs, model = problem()
    fn = jax.jit(lambda j: marginalized_continuum_log_likelihood(obs,model,sigma_constant=.1,sigma_continuum=.03,jitter=j))
    with pytest.raises(Exception, match="jitter must be finite and nonnegative"):
        fn(-.01).block_until_ready()
