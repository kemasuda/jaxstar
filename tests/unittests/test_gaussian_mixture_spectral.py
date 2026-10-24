"""Normalized pixel mixtures, responsibilities and continuous-only inference."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from jaxstar.specfit import (
    Observation, chebyshev_basis, make_model_single, model_single,
    gaussian_mixture_log_likelihood, gaussian_mixture_responsibility,
    save_fit_result, load_fit_result,
)
from test_specfit import case, configured


def reference(y, mean, error, jitter, fraction, mismatch):
    narrow = np.hypot(error, jitter)
    terms = np.stack((np.log1p(-fraction) + norm.logpdf(y, loc=mean, scale=narrow),
                      np.log(fraction) + norm.logpdf(y, loc=mean, scale=np.hypot(narrow, mismatch))))
    logp = logsumexp(terms, axis=0)
    return logp, np.exp(terms[1] - logp)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("double", [False, True])
def test_reference_density_responsibility_masks_jit_and_gradients(single, double, x64_context):
    with x64_context(double):
        dtype = np.float64 if double else np.float32
        wave = np.array([[5000., 5001., 5002.], [6000., 6001., 6002.]], dtype=dtype)
        y = np.array([[1., np.nan, .7], [np.inf, 1.1, .98]], dtype=dtype)
        error = np.array([[.01, np.inf, .02], [np.nan, .03, .01]], dtype=dtype)
        mask = np.array([[False, True, False], [True, False, False]])
        mean = np.array([[.99, np.inf, .9], [np.nan, 1., 1.]], dtype=dtype)
        jitter = np.array([.004, .008], dtype=dtype)
        if single:
            wave, y, error, mask, mean = (v[0] for v in (wave, y, error, mask, mean))
            jitter = jitter[0]
        obs = Observation(wave, y, error, mask)
        options = dict(mismatch_fraction=.08, mismatch_scale=.15, jitter=jitter)
        expected, q = reference(y, mean, error, jitter if single else jitter[:, None], .08, .15)
        actual = gaussian_mixture_log_likelihood(obs, mean, **options)
        probability = gaussian_mixture_responsibility(obs, mean, **options)
        tolerance = 3e-6 if not double else 1e-12
        np.testing.assert_allclose(actual, expected[~mask].sum(), rtol=tolerance)
        np.testing.assert_allclose(probability[~mask], q[~mask], rtol=tolerance)
        assert np.isnan(probability[mask]).all()
        def density(m, f, s, j, data):
            return gaussian_mixture_log_likelihood(data, m, mismatch_fraction=f, mismatch_scale=s, jitter=j)
        value, grads = jax.jit(jax.value_and_grad(density, argnums=(0, 1, 2, 3)))(
            mean, .08, .15, jitter, obs)
        np.testing.assert_allclose(value, actual, rtol=tolerance)
        assert all(np.isfinite(g).all() for g in grads)
        np.testing.assert_array_equal(grads[0][mask], 0.)
        compiled_q = jax.jit(gaussian_mixture_responsibility)(obs, mean, **options)
        np.testing.assert_allclose(compiled_q, probability, rtol=tolerance, equal_nan=True)
        clean = Observation(wave, np.where(mask, 100., y), np.where(mask, 1., error), mask)
        np.testing.assert_allclose(gaussian_mixture_log_likelihood(
            clean, np.where(mask, -100., mean), **options), actual)


@pytest.mark.parametrize("fraction,mismatch", [(0., .1), (1., .1), (.2, 0.), (0., 0.), (1., 0.)])
def test_exact_gaussian_limits_and_continuous_gradients(fraction, mismatch, x64_context):
    with x64_context():
        obs = Observation([1., 2., 3.], [.99, 1.02, 1.04], [.01, .02, .03])
        mean = jnp.ones(3)
        options = dict(mismatch_fraction=fraction, mismatch_scale=mismatch, jitter=.004)
        effective = np.sqrt(obs.uncertainty**2 + .004**2 + (mismatch**2 if fraction == 1. else 0.))
        np.testing.assert_allclose(gaussian_mixture_log_likelihood(obs, mean, **options),
                                   norm.logpdf(obs.flux, loc=mean, scale=effective).sum(), atol=1e-12)
        q = gaussian_mixture_responsibility(obs, mean, **options)
        np.testing.assert_allclose(q, fraction, atol=1e-15)
        fn = lambda m, j, s: gaussian_mixture_log_likelihood(
            obs, m, mismatch_fraction=fraction, mismatch_scale=s, jitter=j)
        _, grads = jax.jit(jax.value_and_grad(fn, argnums=(0, 1, 2)))(mean, 0., mismatch)
        assert all(np.isfinite(g).all() for g in grads)
        assert grads[1] == 0.


def test_gradients_normalization_absolute_flux_units_and_extreme_residuals(x64_context):
    with x64_context():
        obs = Observation([1., 2., 3.], [1., 1.03, .8], [.01, .02, .01])
        mean = jnp.ones(3)
        options = dict(mismatch_fraction=.1, mismatch_scale=.12, jitter=.006)
        fn = lambda f, s, j: gaussian_mixture_log_likelihood(obs, mean, mismatch_fraction=f, mismatch_scale=s, jitter=j)
        params = [.1, .12, .006]
        gradients = jax.grad(fn, argnums=(0, 1, 2))(*params)
        for index in range(3):
            upper, lower = params.copy(), params.copy()
            upper[index] += 1e-6
            lower[index] -= 1e-6
            np.testing.assert_allclose(gradients[index], (fn(*upper) - fn(*lower)) / 2e-6, rtol=2e-6)
        # All absolute flux amplitudes must rescale together; f is dimensionless.
        factor = 37.
        scaled = Observation(obs.wavelength, obs.flux * factor, obs.uncertainty * factor)
        scaled_options = dict(mismatch_fraction=.1, mismatch_scale=.12 * factor, jitter=.006 * factor)
        np.testing.assert_allclose(gaussian_mixture_log_likelihood(scaled, mean * factor, **scaled_options),
            fn(*params) - obs.n_pixels * np.log(factor), atol=1e-12)
        np.testing.assert_allclose(gaussian_mixture_responsibility(scaled, mean * factor, **scaled_options),
                                   gaussian_mixture_responsibility(obs, mean, **options))
        exact = replace(obs, flux=np.ones(3))
        assert gaussian_mixture_log_likelihood(exact, mean, **dict(options, jitter=.03)) < (
            gaussian_mixture_log_likelihood(exact, mean, **dict(options, jitter=0.)))
        extreme = Observation([1.], [1000.], [.01])
        value = gaussian_mixture_log_likelihood(extreme, [0.], **options)
        expected, _ = reference(extreme.flux, [0.], extreme.uncertainty, .006, .1, .12)
        np.testing.assert_allclose(value, expected.sum(), rtol=1e-12)
        np.testing.assert_allclose(gaussian_mixture_responsibility(extreme, [0.], **options), [1.])


def test_fully_masked_zero_density_and_prior_only_gradients(x64_context):
    with x64_context():
        obs = Observation([1., 2.], [np.nan, np.inf], [np.inf, np.nan], [True, True])
        def density(f, s, j):
            return gaussian_mixture_log_likelihood(obs, jnp.full(2, jnp.nan),
                mismatch_fraction=f, mismatch_scale=s, jitter=j)
        value, grads = jax.jit(jax.value_and_grad(density, argnums=(0, 1, 2)))(.1, .2, .01)
        assert value == 0.
        np.testing.assert_array_equal(grads, [0., 0., 0.])
        assert np.isnan(gaussian_mixture_responsibility(obs, [np.inf, np.nan],
            mismatch_fraction=.1, mismatch_scale=.2)).all()


@pytest.mark.parametrize("changes,message", [
    ({"mismatch_fraction": -.01}, "mismatch_fraction"),
    ({"mismatch_fraction": 1.01}, "mismatch_fraction"),
    ({"mismatch_fraction": np.nan}, "mismatch_fraction"),
    ({"mismatch_fraction": np.inf}, "mismatch_fraction"),
    ({"mismatch_fraction": [.1]}, "global scalar"),
    ({"mismatch_scale": -.01}, "mismatch_scale"),
    ({"mismatch_scale": np.nan}, "mismatch_scale"),
    ({"mismatch_scale": np.inf}, "mismatch_scale"),
    ({"mismatch_scale": [.1]}, "global scalar"),
    ({"jitter": -.01}, "jitter"),
])
def test_invalid_parameters(changes, message):
    obs = Observation([1.], [1.], [.01])
    options = dict(mismatch_fraction=.1, mismatch_scale=.2)
    options.update(changes)
    for function in (gaussian_mixture_log_likelihood, gaussian_mixture_responsibility):
        with pytest.raises(ValueError, match=message):
            function(obs, [1.], **options)


@pytest.mark.parametrize("single", [False, True])
def test_model_joint_and_reconstruction_use_sampled_continuum(single, tmp_path, x64_context):
    with x64_context():
        fit, settings, point = case(single)
        coefficients = np.broadcast_to([1.01, -.015, .007], (3,) if single else (2, 3)).copy()
        settings.update(likelihood="gaussian_mixture", mismatch_fraction=dist.Beta(1., 9.),
                        mismatch_scale=dist.HalfNormal(.1), degree=2,
                        basis=chebyshev_basis(fit.observation.wavelength, 2))
        model = configured(fit, settings)
        point = dict(point, continuum_coefficients=coefficients, mismatch_fraction=.08, mismatch_scale=.1)
        value, trace = log_density(model, (fit.observation, fit.specmodel), {}, point)
        latent = {k for k, v in trace.items() if v["type"] == "sample" and not v["is_observed"]}
        assert latent == {"teff", "continuum_coefficients", "mismatch_fraction", "mismatch_scale"}
        assert all(not trace[k]["fn"].support.is_discrete for k in latent)
        prediction = fit.predict(point=point, model=model)
        expected_continuum = np.sum(np.asarray(settings["basis"]) * coefficients[..., None, :], axis=-1)
        np.testing.assert_allclose(prediction["continuum"], expected_continuum)
        np.testing.assert_allclose(prediction["model_flux"], prediction["physical_flux"] * expected_continuum)
        np.testing.assert_array_equal(prediction["mean_flux"], prediction["model_flux"])
        assert "continuum_flux" not in prediction
        assert "continuum_posterior" not in prediction and "gp_mean" not in prediction
        good = fit.observation.valid
        logp, q = reference(fit.observation.flux, prediction["physical_flux"] * expected_continuum,
                            fit.observation.uncertainty, settings["jitter"], .08, .1)
        np.testing.assert_allclose(prediction["mismatch_probability"][good], q[good], atol=1e-12)
        np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, logp[good].sum(), atol=1e-10)
        prior = norm.logpdf(coefficients, [1., 0., 0.], [.1, .03, .03]).sum() - np.log(1500.)
        prior += dist.Beta(1., 9.).log_prob(.08) + dist.HalfNormal(.1).log_prob(.1)
        np.testing.assert_allclose(value, logp[good].sum() + prior, atol=1e-10)
        report = model.check_prior()
        assert "continuum_coefficients | sigma_constant, sigma_continuum" in report.index
        assert "mismatch_fraction" in report.index and "mismatch_scale" in report.index
        assert "spectrum" not in model.prior_trace()
        with pytest.raises(ValueError, match="continuum_coefficients"):
            fit.predict(point={k: v for k, v in point.items() if k != "continuum_coefficients"}, model=model)
        save_fit_result(tmp_path / "mixture", sf=fit, model=model, point=point)
        restored = load_fit_result(tmp_path / "mixture")
        np.testing.assert_allclose(fit.predict(point=restored.point, model=model)["mean_flux"], prediction["mean_flux"])
        fit.set_fit_mask(np.ones(fit.observation.shape, bool))
        masked_value, _ = log_density(model, (fit.fit_observation(), fit.specmodel), {}, point)
        np.testing.assert_allclose(masked_value, prior, atol=1e-10)
        # Excluding every pixel from inference does not prevent classification
        # at a supplied point; only the fixed observation mask hides probabilities.
        np.testing.assert_allclose(fit.predict(point=point, model=model)["mismatch_probability"],
                                   prediction["mismatch_probability"], equal_nan=True)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("show_masked", [False, True])
def test_draw_probabilities_are_averaged_before_plotting_without_clipping(single, show_masked, tmp_path, x64_context):
    with x64_context():
        fit, settings, point = case(single)
        settings.update(likelihood="gaussian_mixture", mismatch_fraction=.08, mismatch_scale=.1,
                        degree=0, basis=chebyshev_basis(fit.observation.wavelength, 0))
        model = configured(fit, settings)
        fit_mask = np.zeros(fit.observation.shape, dtype=bool)
        fit_mask[..., 10] = True
        fit.set_fit_mask(fit_mask)
        point = dict(point, continuum_coefficients=np.full((1,) if single else (2, 1), 1.01))
        other = dict(point, continuum_coefficients=point["continuum_coefficients"] + .04)
        first, second = (fit.predict(point=p, model=model) for p in (point, other))
        draws = {k: np.stack([point[k], other[k]]) for k in point}
        average = fit.predict(samples=draws, model=model, max_draws=None)
        q = .5 * (first["mismatch_probability"] + second["mismatch_probability"])
        np.testing.assert_allclose(average["mismatch_probability"], q, equal_nan=True)
        assert np.isnan(q[fit.observation.mask]).all()
        assert np.isfinite(q[fit.fit_mask & ~fit.observation.mask]).all()
        expected_bands = np.quantile([first["model_flux"], second["model_flux"]], [.16, .84], axis=0)
        np.testing.assert_allclose(average["model_flux_quantiles"], expected_bands)
        assert "continuum_flux_quantiles" not in average
        plugin = gaussian_mixture_responsibility(fit.observation, average["mean_flux"],
            mismatch_fraction=.08, mismatch_scale=.1, jitter=.004)
        assert not np.allclose(q[~fit.effective_mask], plugin[~fit.effective_mask])
        original_mask = fit.effective_mask.copy()
        try:
            for options in (dict(point=point), dict(samples=draws, max_draws=None)):
                plots = fit.plot_models(model=model, show_mismatch=True, show_masked=show_masked,
                                        show_interval=True, save_path=tmp_path / "mixture.png", **options)
                expected = first["mismatch_probability"] if "point" in options else q
                expected = np.where(fit.observation.mask if show_masked else fit.effective_mask,
                                    np.nan, expected)
                for i, (fig, axes) in enumerate(plots):
                    assert axes.shape == (3,)
                    np.testing.assert_allclose(axes[2].lines[0].get_ydata(), np.atleast_2d(expected)[i], equal_nan=True)
                    assert axes[2].get_ylim()[0] <= 0 and axes[2].get_ylim()[1] >= 1
                    if not show_masked:
                        assert not {"fixed mask", "fit mask"}.intersection(
                            axes[0].get_legend_handles_labels()[1])
                    if "samples" in options:
                        assert axes[0].collections
                    plt.close(fig)
            np.testing.assert_array_equal(fit.effective_mask, original_mask)
        finally:
            plt.close("all")


@pytest.mark.parametrize("single", [False, True])
def test_factory_fixed_and_sampled_mixture_and_gaussian_defaults(single, x64_context):
    with x64_context():
        fit, _, _ = case(single)
        common = dict(resolving_power=70000., fixed={"rv": 12., "vsini": 6.}, use_gp=False)
        for fraction, scale in ((.1, .2), (dist.Beta(1., 9.), dist.HalfNormal(.1))):
            model = make_model_single(fit, **common, likelihood="gaussian_mixture",
                                      mismatch_fraction=fraction, mismatch_scale=scale, degree=0)
            trace = model.prior_trace()
            point = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
            assert trace["continuum_coefficients"]["value"].shape == ((1,) if single else (2, 1))
            assert trace["mismatch_fraction"]["type"] == ("sample" if isinstance(fraction, dist.Distribution) else "deterministic")
            assert np.isfinite(fit.predict(point=point, model=model)["mean_flux"]).all()
        gaussian = make_model_single(fit, **common)
        assert not {"mismatch_fraction", "mismatch_scale", "continuum_coefficients"}.intersection(gaussian.prior_trace())
        with pytest.raises(ValueError, match="show_mismatch requires"):
            gaussian_point = {k: v["value"] for k, v in gaussian.prior_trace().items() if v["type"] == "sample"}
            fit.plot_models(point=gaussian_point, model=gaussian, show_mismatch=True)


@pytest.mark.parametrize("changes,message", [
    ({"likelihood": "robust"}, "likelihood must"),
    ({"likelihood": "gaussian"}, "mismatch parameters require"),
    ({"use_gp": True}, "requires use_gp=False"),
    ({"gp_amplitude": .01, "gp_scale": .2}, "requires use_gp=False"),
    ({"fixed": {"gp_scale": .2}}, "requires use_gp=False"),
    ({"mismatch_fraction": None}, "supply mismatch_fraction"),
    ({"mismatch_scale": None}, "supply mismatch_fraction"),
    ({"mismatch_fraction": dist.Beta(jnp.ones(2), jnp.ones(2))}, "scalar"),
    ({"mismatch_scale": -.1}, "mismatch_scale"),
    ({"basis": np.ones((2, 96, 3))}, "basis shape"),
])
def test_invalid_model_configuration(changes, message):
    fit, settings, _ = case()
    settings.update(likelihood="gaussian_mixture", mismatch_fraction=.1, mismatch_scale=.2, prior_only=True)
    settings.update(changes)
    with pytest.raises(ValueError, match=message):
        handlers.seed(model_single, 0)(fit.observation, fit.specmodel, **settings)


def test_factory_rejects_gp_and_requires_explicit_mixture_settings():
    fit, _, _ = case()
    common = dict(resolving_power=70000., likelihood="gaussian_mixture")
    with pytest.raises(ValueError, match="requires use_gp=False"):
        make_model_single(fit, **common, mismatch_fraction=.1, mismatch_scale=.2)
    with pytest.raises(ValueError, match="supply mismatch_fraction"):
        make_model_single(fit, **common, use_gp=False)


def test_joint_jit_gradient_with_sampled_hyperparameters(x64_context):
    with x64_context():
        fit, settings, point = case()
        settings.update(likelihood="gaussian_mixture", mismatch_fraction=dist.Beta(1., 9.),
                        mismatch_scale=dist.HalfNormal(.1), jitter=dist.HalfNormal(jnp.array([.01, .02])),
                        sigma_constant=jnp.array([.1, .2]), sigma_continuum=dist.LogNormal(jnp.log(.03), .5))
        point = dict(point, mismatch_fraction=.1, mismatch_scale=.1, jitter=jnp.array([.004, .008]),
                     sigma_continuum=.03, continuum_coefficients=jnp.array([[1., .01, 0., 0., 0.], [1.01, 0., 0., 0., 0.]]))
        def density(p, obs, specmodel):
            return log_density(model_single, (obs, specmodel), settings, p)[0]
        value, gradient = jax.jit(jax.value_and_grad(density))(point, fit.observation, fit.specmodel)
        np.testing.assert_allclose(value, density(point, fit.observation, fit.specmodel), atol=1e-9)
        assert all(np.isfinite(g).all() for g in jax.tree.leaves(gradient))


def test_factory_run_svi_optimizes_explicit_continuum_and_predicts_without_masking(x64_context):
    with x64_context():
        fit, _, _ = case(single=True)
        contaminated = fit.observation.flux.copy()
        contaminated[[18, 20, 78]] += [.25, .5, -.3]
        fit.observation = replace(fit.observation, flux=contaminated)
        model = make_model_single(
            fit, resolving_power=70000., degree=1, use_gp=False,
            fixed=dict(vsini=6., vmacro=3., q1=.36, q2=.3, rv=12., jitter=.004),
            sigma_constant=.1, sigma_continuum=.03,
            likelihood="gaussian_mixture", mismatch_fraction=dist.Beta(1., 9.),
            mismatch_scale=dist.HalfNormal(.2))
        initial = dict(teff=5600., continuum_coefficients=jnp.array([1., 0.]),
                       mismatch_fraction=.05, mismatch_scale=.2)
        mask_before = fit.effective_mask.copy()
        point = fit.run_svi(model=model, rng_key=jax.random.PRNGKey(2),
                            p_initial=initial, step_size=.02, num_steps=200, progress_bar=False)
        # The guide median also includes the model's fixed/derived sites.
        assert set(initial).issubset(point)
        assert point["rv"] == 12.
        assert point["jitter"] == .004
        assert all(np.isfinite(value).all() for value in point.values())
        assert log_density(model, (fit.observation, fit.specmodel), {}, point)[0] > (
            log_density(model, (fit.observation, fit.specmodel), {}, initial)[0])
        np.testing.assert_allclose(point["continuum_coefficients"], [1.01, .01], atol=.005)
        np.testing.assert_allclose(point["teff"], 5800., atol=100.)
        prediction = fit.predict(point=point, model=model)
        assert np.all(np.asarray(prediction["mismatch_probability"])[[18, 20, 78]] > .99)
        assert np.nanmedian(prediction["mismatch_probability"]) < .05
        np.testing.assert_array_equal(fit.effective_mask, mask_before)


def test_small_svi_nuts_and_saved_posterior_without_discrete_sites(tmp_path, x64_context):
    with x64_context():
        fit, settings, _ = case(single=True)
        flux = fit.observation.flux.copy()
        flux[[18, 20, 50]] += [.25, .5, -.3]
        fit.observation = replace(fit.observation, flux=flux)
        settings.update(atmosphere={"teff": 5800.}, likelihood="gaussian_mixture", degree=1,
                        mismatch_fraction=dist.Beta(1., 9.), mismatch_scale=dist.HalfNormal(.2),
                        basis=chebyshev_basis(fit.observation.wavelength, 1))
        model = configured(fit, settings)
        initial = {"continuum_coefficients": jnp.array([1., 0.]), "mismatch_fraction": .05, "mismatch_scale": .2}
        guide = AutoNormal(model, init_loc_fn=init_to_value(values=initial), init_scale=.01)
        svi = SVI(model, guide, numpyro.optim.Adam(.01), Trace_ELBO())
        result = svi.run(jax.random.PRNGKey(0), 100, fit.observation, fit.specmodel, progress_bar=False)
        assert np.isfinite(result.losses).all()
        estimate = guide.median(result.params)
        sampler = MCMC(NUTS(model, init_strategy=init_to_value(values=estimate)),
                       num_warmup=60, num_samples=40, progress_bar=False)
        sampler.run(jax.random.PRNGKey(1), fit.observation, fit.specmodel)
        assert set(sampler.last_state.z) == set(initial)
        samples = sampler.get_samples()
        assert all(np.isfinite(v).all() for v in samples.values())
        np.testing.assert_allclose(samples["continuum_coefficients"].mean(0), [1.01, .01], atol=.007)
        prediction = fit.predict(samples=samples, model=model, max_draws=5)
        assert np.all(prediction["mismatch_probability"][[18, 20, 50]] > .99)
        assert np.nanmedian(prediction["mismatch_probability"]) < .05
        save_fit_result(tmp_path / "posterior", sf=fit, model=model, mcmc=sampler)
        restored = load_fit_result(tmp_path / "posterior")
        np.testing.assert_allclose(fit.predict(samples=restored.samples, model=model, max_draws=5)["mean_flux"],
                                   prediction["mean_flux"])
