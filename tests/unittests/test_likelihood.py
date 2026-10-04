"""Exact masked GP integration, joint conditional means and iid move parity."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tinygp")

from jaxstar.specfit import (
    Observation, apply_continuum, chebyshev_basis, continuum_posterior,
    continuum_prior, gp_conditional_mean, gp_continuum_posterior,
    gp_marginalized_continuum_log_likelihood, marginalized_continuum_log_likelihood,
    prepare_gp_observation,
)
from jaxstar.specfit import likelihood


OPTIONS = dict(gp_amplitude=.025, gp_scale=.55, jitter=.009,
               sigma_constant=.1, sigma_continuum=.03)


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def problem(regions=None, dtype=np.float64, masking="mixed"):
    x = np.cumsum(np.random.default_rng(17).uniform(.08, .3, 17))
    wave = (5000 + x).astype(dtype)
    x = wave.astype(float) - float(wave[0])
    f = 1 - .3 * np.exp(-.5 * ((x - 1.5) / .2)**2)
    y = f * (1.02 + .02 * (x - x.mean())) + .012 * np.sin(3*x)
    error = .02 + .004 * np.arange(len(x)) / len(x)
    mask = np.zeros(len(x), dtype=bool)
    if masking in ("mixed", "isolated"):
        mask[[1, 12]] = True
    if masking in ("mixed", "block"):
        mask[5:8] = True
    if regions is not None:
        wave = np.stack([wave + 1000*i for i in range(regions)])
        f = np.stack([f*(1-.03*i) for i in range(regions)])
        y = np.stack([y*(1-.03*i) for i in range(regions)])
        error = np.broadcast_to(error, wave.shape).copy()
        mask = np.broadcast_to(mask, wave.shape).copy()
        if regions > 1:
            mask[1, 2:4] = True
    y, error = np.asarray(y, dtype=dtype), np.asarray(error, dtype=dtype)
    y[mask], error[mask] = np.nan, np.inf
    return Observation(wave, y, error, mask), jnp.asarray(f, dtype=dtype)


def kernel(x, z, amplitude, scale):
    distance = np.abs(x[:, None] - z[None, :])
    argument = np.sqrt(3.) * distance / scale
    return amplitude**2 * (1 + argument) * np.exp(-argument)


def dense_reference(obs, flux, *, gp_amplitude, gp_scale, jitter,
                    sigma_constant, sigma_continuum, degree=4):
    """Independent full Gaussian covariance, used only on tiny test problems."""
    basis = np.asarray(chebyshev_basis(obs.wavelength, degree), dtype=float)
    basis = basis.reshape((obs.n_regions, obs.n_pixels, degree+1))
    wave, flux = np.atleast_2d(obs.wavelength), np.atleast_2d(np.asarray(flux))
    jitter = np.broadcast_to(jitter, (obs.n_regions,))
    s0 = np.broadcast_to(sigma_constant, (obs.n_regions,))
    sc = np.broadcast_to(sigma_continuum, (obs.n_regions,))
    total, means, covariances, gp_means = 0., [], [], []
    for region in range(obs.n_regions):
        valid = np.atleast_2d(obs.valid)[region]
        mu = np.zeros(degree+1)
        mu[0] = 1
        prior_cov = np.diag(np.array([s0[region]] + [sc[region]]*degree)**2)
        if not np.any(valid):
            means.append(mu)
            covariances.append(prior_cov)
            gp_means.append(np.zeros(obs.n_pixels))
            continue
        w = wave[region].astype(float)
        y = np.atleast_2d(obs.flux)[region, valid].astype(float)
        error = np.atleast_2d(obs.uncertainty)[region, valid].astype(float)
        design = flux[region, valid, None] * basis[region, valid]
        c = kernel(w[valid], w[valid], gp_amplitude, gp_scale)
        c += np.diag(error**2 + jitter[region]**2)
        v = c + design @ prior_cov @ design.T
        residual = y - design @ mu
        solved = np.linalg.solve(v, residual)
        total -= .5*(residual @ solved + np.linalg.slogdet(v)[1]
                     + valid.sum()*np.log(2*np.pi))
        cross = prior_cov @ design.T
        means.append(mu + cross @ solved)
        covariances.append(prior_cov - cross @ np.linalg.solve(v, cross.T))
        # Joint GP posterior mean after integrating continuum uncertainty.
        gp_means.append(kernel(w, w[valid], gp_amplitude, gp_scale) @ solved)
    mean, cov, gp_mean = map(np.asarray, (means, covariances, gp_means))
    if obs.ndim == 1:
        mean, cov, gp_mean = mean[0], cov[0], gp_mean[0]
    return total, mean, cov, gp_mean


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
@pytest.mark.parametrize("regions", [None, 2])
@pytest.mark.parametrize("degree", [0, 4])
def test_dense_joint_likelihood_posterior_and_reconstruction(solver, regions, degree):
    obs, flux = problem(regions)
    opts = dict(OPTIONS, degree=degree, gp_solver=solver)
    if regions is not None:
        opts.update(jitter=[.005, .02], sigma_constant=[.1, .2], sigma_continuum=[.02, .04])
    expected = dense_reference(obs, flux, **{k: v for k, v in opts.items() if k != "gp_solver"})
    actual = gp_marginalized_continuum_log_likelihood(obs, flux, **opts)
    posterior = gp_continuum_posterior(obs, flux, **opts)
    basis = chebyshev_basis(obs.wavelength, degree)
    mean_flux = apply_continuum(flux, basis, posterior.mean)
    gp_mean = gp_conditional_mean(obs, mean_flux, gp_amplitude=opts["gp_amplitude"],
                                 gp_scale=opts["gp_scale"], jitter=opts["jitter"], gp_solver=solver)
    for value, reference in zip((actual, posterior.mean, posterior.covariance, gp_mean), expected):
        np.testing.assert_allclose(value, reference, rtol=2e-11, atol=2e-11)
    assert np.all(np.isfinite(gp_mean))  # Includes predictions at masked pixels.


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
@pytest.mark.parametrize("masking", ["none", "isolated", "block"])
def test_mask_submatrix_dense_parity(solver, masking):
    obs, flux = problem(masking=masking)
    expected = dense_reference(obs, flux, **OPTIONS)
    value = gp_marginalized_continuum_log_likelihood(obs, flux, gp_solver=solver, **OPTIONS)
    posterior = gp_continuum_posterior(obs, flux, gp_solver=solver, **OPTIONS)
    for actual, reference in zip((value, posterior.mean, posterior.covariance), expected[:3]):
        np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("regions", [None, 2])
def test_solvers_jit_dynamic_data_dtype_and_all_gradients(dtype, regions, x64_context):
    with x64_context(dtype == np.float64):
        obs, flux = problem(regions, dtype)
        prepared = prepare_gp_observation(obs)
        results = []
        before = jax.config.x64_enabled
        for solver in ("quasisep", "direct"):
            fn = lambda data, f, a, l, s0, sc, j: gp_marginalized_continuum_log_likelihood(
                data, f, gp_amplitude=a, gp_scale=l, sigma_constant=s0,
                sigma_continuum=sc, jitter=j, gp_solver=solver)
            args = (prepared, flux, .025, .55, .1, .03, .009)
            result = jax.jit(jax.value_and_grad(fn, argnums=(1, 2, 3, 4, 5, 6)))(*args)
            value, gradients = result
            assert value.dtype == dtype
            assert all(np.all(np.isfinite(x)) for x in gradients)
            np.testing.assert_array_equal(gradients[0][obs.mask], 0)
            tolerance = dict(rtol=2e-4, atol=2e-4) if dtype == np.float32 else dict(rtol=2e-11, atol=2e-11)
            np.testing.assert_allclose(value, fn(*args), **tolerance)
            post = jax.jit(lambda data, f: gp_continuum_posterior(
                data, f, gp_solver=solver, **OPTIONS))(prepared, flux)
            assert post.mean.dtype == dtype and post.covariance.dtype == dtype
            prediction = jax.jit(lambda data, f: gp_conditional_mean(
                data, f, gp_amplitude=.025, gp_scale=.55, jitter=.009, gp_solver=solver))(prepared, flux)
            assert prediction.dtype == dtype and np.all(np.isfinite(prediction))
            results.append((result, post, prediction))
        tolerance = dict(rtol=5e-4, atol=2e-3) if dtype == np.float32 else dict(rtol=3e-10, atol=3e-10)
        for q, d in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
            np.testing.assert_allclose(q, d, **tolerance)
        assert before == jax.config.x64_enabled


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
def test_gradients_match_finite_differences(solver):
    obs, flux = problem()
    prepared = prepare_gp_observation(obs)
    # Include nonlinear physical-model sensitivity as well as all nuisance scales.
    def fn(theta):
        f = flux * (1 + theta[0])
        return gp_marginalized_continuum_log_likelihood(prepared, f,
            gp_amplitude=theta[1], gp_scale=theta[2], sigma_constant=theta[3],
            sigma_continuum=theta[4], jitter=theta[5], gp_solver=solver)
    theta = jnp.array([.01, .025, .55, .1, .03, .009])
    gradient = jax.jit(jax.grad(fn))(theta)
    step = 1e-6
    for i in range(len(theta)):
        finite = (fn(theta.at[i].add(step)) - fn(theta.at[i].add(-step)))/(2*step)
        np.testing.assert_allclose(gradient[i], finite, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
@pytest.mark.parametrize("regions", [None, 2])
def test_conditional_mean_at_other_wavelengths_and_gradients(solver, regions):
    obs, flux = problem(regions)
    prepared = prepare_gp_observation(obs)
    wave = np.linspace(4999.5, 5004.5, 23)
    if regions is not None:
        wave = np.stack([wave + 1000*i for i in range(regions)])
    opts = dict(gp_amplitude=.025, gp_scale=.55, jitter=.009, gp_solver=solver)
    fn = lambda data, f, w: gp_conditional_mean(data, f, wavelength=w, **opts)
    actual = jax.jit(fn)(prepared, flux, wave)
    expected = []
    for r in range(obs.n_regions):
        valid = np.atleast_2d(obs.valid)[r]
        x = np.atleast_2d(obs.wavelength)[r, valid]
        error = np.atleast_2d(obs.uncertainty)[r, valid]
        residual = np.atleast_2d(obs.flux)[r, valid] - np.atleast_2d(flux)[r, valid]
        c = kernel(x, x, .025, .55) + np.diag(error**2 + .009**2)
        expected.append(kernel(np.atleast_2d(wave)[r], x, .025, .55) @ np.linalg.solve(c, residual))
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected[0] if regions is None else expected, rtol=1e-12, atol=1e-12)
    grad = jax.jit(jax.grad(lambda f: fn(prepared, f, wave).sum()))(flux)
    assert np.all(np.isfinite(grad))
    np.testing.assert_array_equal(grad[obs.mask], 0)


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_poisoned_masked_values_do_not_enter_arithmetic_or_gradients(solver, dtype, x64_context):
    with x64_context(dtype == np.float64):
        obs, flux = problem(2, dtype)
        y, error = obs.flux.copy(), obs.uncertainty.copy()
        y[obs.mask], error[obs.mask] = np.inf, -np.inf
        changed = Observation(obs.wavelength, y, error, obs.mask)
        y[obs.mask], error[obs.mask] = np.nan, np.nan
        other = Observation(obs.wavelength, y, error, obs.mask)
        dirty_model = jnp.where(obs.mask, jnp.nan, flux)
        fn = lambda data, f: gp_marginalized_continuum_log_likelihood(data, f, gp_solver=solver, **OPTIONS)
        for current in (obs, changed, other):
            prepared = prepare_gp_observation(current)
            value, gradient = jax.jit(jax.value_and_grad(fn, argnums=1))(prepared, dirty_model)
            np.testing.assert_allclose(value, fn(obs, flux), rtol=2e-6, atol=2e-6)
            assert np.all(np.isfinite(gradient))
            np.testing.assert_array_equal(gradient[obs.mask], 0)
            post = gp_continuum_posterior(prepared, dirty_model, gp_solver=solver, **OPTIONS)
            assert all(np.all(np.isfinite(x)) for x in post)


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
@pytest.mark.parametrize("regions", [None, 2])
def test_fully_masked_region_prior_zero_density_and_prediction(solver, regions):
    original, flux = problem(regions)
    mask = original.mask.copy()
    if regions is None:
        mask[:] = True
    else:
        mask[1] = True
    y, error = original.flux.copy(), original.uncertainty.copy()
    y[mask], error[mask] = np.nan, -np.inf
    obs = Observation(original.wavelength, y, error, mask)
    prepared = prepare_gp_observation(obs)
    dirty = jnp.where(mask, jnp.inf, flux)
    opts = dict(OPTIONS, gp_solver=solver)
    value = jax.jit(lambda data, f: gp_marginalized_continuum_log_likelihood(data, f, **opts))(prepared, dirty)
    post = gp_continuum_posterior(prepared, dirty, **opts)
    pred = gp_conditional_mean(prepared, dirty, gp_amplitude=.025, gp_scale=.55, gp_solver=solver)
    prior = continuum_prior(sigma_constant=.1, sigma_continuum=.03)
    if regions is None:
        np.testing.assert_array_equal(value, 0.)
        np.testing.assert_array_equal(post.mean, prior.mean)
        np.testing.assert_array_equal(post.covariance, prior.covariance)
        np.testing.assert_array_equal(pred, 0.)
        grad = jax.grad(lambda a: gp_marginalized_continuum_log_likelihood(prepared, dirty, **dict(opts, gp_amplitude=a)))(.025)
        np.testing.assert_array_equal(grad, 0.)
    else:
        expected = dense_reference(obs, flux, **OPTIONS)
        np.testing.assert_allclose(value, expected[0], atol=1e-11)
        np.testing.assert_array_equal(post.mean[1], prior.mean)
        np.testing.assert_array_equal(post.covariance[1], prior.covariance)
        np.testing.assert_array_equal(pred[1], 0.)
        jitters = jnp.array([.009, .05])
        grad = jax.grad(lambda j: gp_marginalized_continuum_log_likelihood(prepared, dirty, **dict(opts, jitter=j)))(jitters)
        np.testing.assert_array_equal(grad[1], 0.)


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
def test_white_gp_limit_matches_iid_with_jitter(solver):
    obs, flux = problem(2)
    options = dict(sigma_constant=.1, sigma_continuum=.03, jitter=[.005, .03])
    value = gp_marginalized_continuum_log_likelihood(obs, flux, gp_amplitude=1e-10,
                                                   gp_scale=.55, gp_solver=solver, **options)
    post = gp_continuum_posterior(obs, flux, gp_amplitude=1e-10, gp_scale=.55, gp_solver=solver, **options)
    np.testing.assert_allclose(value, marginalized_continuum_log_likelihood(obs, flux, **options), atol=1e-11)
    reference = continuum_posterior(obs, flux, **options)
    for actual, expected in zip(post, reference):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_gaussian_normalization_at_zero_residual():
    obs = Observation(np.linspace(5000., 5002., 9), np.zeros(9), np.full(9, .01))
    for a, j in ((.01, 0.), (.03, .02), (.1, .1)):
        options = dict(OPTIONS, gp_amplitude=a, jitter=j)
        expected = dense_reference(obs, np.zeros(9), **options)[0]
        for solver in ("direct", "quasisep"):
            actual = gp_marginalized_continuum_log_likelihood(obs, np.zeros(9), gp_solver=solver, **options)
            np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_fixed_mask_setup_is_required_for_dynamic_observation_and_rebuild():
    obs, flux = problem()
    fn = jax.jit(lambda data, f: gp_marginalized_continuum_log_likelihood(data, f, **OPTIONS))
    with pytest.raises(ValueError, match="prepare_gp_observation.*outside JIT/NUTS"):
        fn(obs, flux)
    prepared = prepare_gp_observation(obs)
    assert prepare_gp_observation(prepared) is prepared
    assert not any(isinstance(x, tuple) for x in jax.tree.leaves(prepared))
    np.testing.assert_allclose(fn(prepared, flux), gp_marginalized_continuum_log_likelihood(obs, flux, **OPTIONS), atol=1e-11)
    # Closing over concrete data also works: only nonlinear parameters are traced.
    closed = jax.jit(lambda f: gp_marginalized_continuum_log_likelihood(obs, f, **OPTIONS))
    np.testing.assert_allclose(closed(flux), fn(prepared, flux), atol=1e-11)
    mask = obs.mask.copy()
    mask[3:5] = True
    changed = Observation(obs.wavelength, obs.flux, obs.uncertainty, mask)
    rebuilt = prepare_gp_observation(changed)
    assert rebuilt.indices != prepared.indices
    np.testing.assert_allclose(fn(rebuilt, flux), dense_reference(changed, flux, **OPTIONS)[0], atol=1e-11)


@pytest.mark.parametrize("platform, expected", [("cpu", "quasisep"), ("gpu", "direct"), ("rocm", "direct")])
def test_auto_solver_policy_and_actual_classes(monkeypatch, platform, expected):
    import tinygp
    monkeypatch.setattr(jax, "default_backend", lambda: platform)
    assert likelihood._resolve_gp_solver("auto") == expected
    obs, _ = problem()
    gp, *_ = likelihood._region_gp(obs, 0, tuple(np.flatnonzero(obs.valid)), .025, .55, .009, jnp.float64, expected)
    cls = tinygp.solvers.QuasisepSolver if expected == "quasisep" else tinygp.solvers.DirectSolver
    assert isinstance(gp.solver, cls)


@pytest.mark.parametrize("name", ["gp_amplitude", "gp_scale"])
@pytest.mark.parametrize("value", [0., -.1, np.nan, np.inf, [.1], [.1, .2]])
def test_positive_scalar_validation(name, value):
    obs, flux = problem(2)
    options = dict(OPTIONS, **{name: value})
    for fn in (gp_marginalized_continuum_log_likelihood, gp_continuum_posterior):
        with pytest.raises(ValueError, match=name):
            fn(obs, flux, **options)
    with pytest.raises(ValueError, match=name):
        gp_conditional_mean(obs, flux, **{k: v for k, v in options.items() if k.startswith("gp_") or k == "jitter"})


def test_shape_solver_jitter_and_traced_value_validation():
    obs, flux = problem(2)
    for j in (-.01, np.nan, np.inf, [0., .01, .02], np.ones(obs.shape)):
        with pytest.raises(ValueError, match="jitter"):
            gp_marginalized_continuum_log_likelihood(obs, flux, **dict(OPTIONS, jitter=j))
    with pytest.raises(ValueError, match="gp_solver"):
        gp_marginalized_continuum_log_likelihood(obs, flux, gp_solver="unknown", **OPTIONS)
    with pytest.raises(ValueError, match="model_flux"):
        gp_continuum_posterior(obs, flux[0], **OPTIONS)
    with pytest.raises(ValueError, match="basis"):
        gp_marginalized_continuum_log_likelihood(obs, flux, basis=np.ones(obs.shape+(4,)), **OPTIONS)
    with pytest.raises(ValueError, match="degree"):
        gp_continuum_posterior(obs, flux, degree=-1, **OPTIONS)
    with pytest.raises(TypeError, match="Observation"):
        prepare_gp_observation(obs.flux)
    for wave in (obs.wavelength[0], np.ones((3, 5)), np.empty((2, 0)), obs.wavelength[:, ::-1]):
        with pytest.raises(ValueError, match="wavelength"):
            gp_conditional_mean(obs, flux, gp_amplitude=.025, gp_scale=.55, wavelength=wave)
    prepared = prepare_gp_observation(obs)
    fn = jax.jit(lambda a: gp_marginalized_continuum_log_likelihood(prepared, flux, **dict(OPTIONS, gp_amplitude=a)))
    with pytest.raises(Exception, match="gp_amplitude must be finite and positive"):
        fn(-.01).block_until_ready()


@pytest.mark.parametrize("dtype, value, jitter_gradient", [
    (np.float32, 39.65412521362305, [-117.41610717773438, -133.04185485839844]),
    (np.float64, 39.653982519385124, [-117.4125990346435, -133.03897025225]),
])
def test_iid_move_signature_imports_and_pre_move_numerical_reference(dtype, value, jitter_gradient):
    from jaxstar.specfit.continuum import marginalized_continuum_log_likelihood as old_path
    assert old_path is marginalized_continuum_log_likelihood
    assert marginalized_continuum_log_likelihood.__module__ == "jaxstar.specfit.likelihood"
    assert tuple(inspect.signature(old_path).parameters) == (
        "observation", "model_flux", "sigma_constant", "sigma_continuum", "degree", "basis", "jitter")
    x = np.linspace(-1, 1, 13)
    wave = np.stack([5000+2*x, 6000+2*x]).astype(dtype)
    f = np.stack([1-.3*np.exp(-.5*(x/.15)**2), .97*(1-.3*np.exp(-.5*(x/.15)**2))]).astype(dtype)
    y = (f*(1.04+.025*x-.01*(2*x**2-1))+.003*np.sin(5*x)).astype(dtype)
    error = np.broadcast_to((.02+.003*(x+1)).astype(dtype), wave.shape)
    mask = np.broadcast_to(np.arange(13)%5 == 2, wave.shape)
    obs = Observation(wave, y, error, mask)
    fn = lambda flux, j: old_path(obs, flux, sigma_constant=.1, sigma_continuum=.03, jitter=j)
    actual, grads = jax.value_and_grad(fn, argnums=(0, 1))(f, np.asarray([.015, .03], dtype=dtype))
    # Captured from 0de1c5c's original continuum.py before moving the function.
    np.testing.assert_allclose(actual, value, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(grads[1], jitter_gradient, rtol=1e-7, atol=1e-11)


def test_optional_tinygp_dependency_does_not_affect_iid(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_tinygp(name, *args, **kwargs):
        if name == "tinygp":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_tinygp)
    obs, flux = problem()
    assert np.isfinite(marginalized_continuum_log_likelihood(obs, flux, sigma_constant=.1, sigma_continuum=.03))
    with pytest.raises(ImportError, match=r"jaxstar\[spectral-inference\]"):
        gp_marginalized_continuum_log_likelihood(obs, flux, **OPTIONS)


def test_float32_x64_mixture_has_setup_guidance():
    obs, flux = problem(dtype=np.float32)
    with pytest.raises(ValueError, match="float32 calculations require x64 disabled"):
        gp_marginalized_continuum_log_likelihood(obs, flux, **OPTIONS)


def test_masked_gp_nuts_smoke_without_changing_model_single():
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    obs, flux = problem()
    prepared = prepare_gp_observation(obs)
    def toy_model():
        offset = numpyro.sample("offset", dist.Normal(0., .03))
        numpyro.factor("spectrum", gp_marginalized_continuum_log_likelihood(
            prepared, flux+offset, **OPTIONS))
    sampler = MCMC(NUTS(toy_model), num_warmup=8, num_samples=8, progress_bar=False)
    sampler.run(jax.random.PRNGKey(3))
    assert np.all(np.isfinite(sampler.get_samples()["offset"]))
