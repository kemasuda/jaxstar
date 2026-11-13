"""Dilution reaches the likelihood, reconstruction, and saved fitted sites."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import init_to_value
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit import (
    make_model_single, save_fit_result, load_fit_result,
    marginalized_continuum_log_likelihood, prepare_gp_observation,
    gp_marginalized_continuum_log_likelihood,
)
from jaxstar.specfit.likelihood import gaussian_mixture_log_likelihood
from test_specfit import case, configured


@pytest.mark.parametrize("mode", ["gaussian", "vecchia1", "mixture"])
@pytest.mark.parametrize("per_region", [False, True])
def test_likelihood_reconstruction_and_dilution_gradient(mode, per_region, x64_context):
    with x64_context():
        fit, settings, point = case()
        plain = configured(fit, settings)
        physical = fit.predict(point=point, model=plain)["physical_flux"]
        d = jnp.array([.15, .3]) if per_region else jnp.array(.23)
        prior = dist.Uniform(jnp.zeros(2), jnp.ones(2)) if per_region else dist.Uniform(0., 1.)
        settings.update(dilution=prior)
        point = dict(point, dilution=d)
        if mode == "vecchia1":
            settings.update(use_gp=True, gp_solver="vecchia1", gp_amplitude=.015, gp_scale=.3)
        elif mode == "mixture":
            settings.update(likelihood="gaussian_mixture", use_gp=False,
                            mismatch_fraction=.08, mismatch_scale=.1)
            point["continuum_coefficients"] = jnp.broadcast_to(
                jnp.array([1.01, -.015, 0., 0., 0.]), (2, 5))
        model = configured(fit, settings)
        prediction = fit.predict(point=point, model=model)
        fraction = d[:, None] if per_region else d
        expected_flux = fraction + (1 - fraction) * physical
        np.testing.assert_allclose(prediction["physical_flux"], expected_flux, atol=1e-13)
        _, trace = log_density(model, (fit.fit_observation(), fit.specmodel), {}, point)
        noise = dict(jitter=.004, sigma_constant=.1, sigma_continuum=.03, basis=settings["basis"])
        if mode == "mixture":
            baseline = expected_flux * prediction["continuum"]
            expected = gaussian_mixture_log_likelihood(fit.fit_observation(), baseline,
                jitter=.004, mismatch_fraction=.08, mismatch_scale=.1)
        elif mode == "vecchia1":
            expected = gp_marginalized_continuum_log_likelihood(
                prepare_gp_observation(fit.fit_observation()), expected_flux,
                gp_amplitude=.015, gp_scale=.3, gp_solver="vecchia1", **noise)
        else:
            expected = marginalized_continuum_log_likelihood(fit.fit_observation(), expected_flux, **noise)
        np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, expected, atol=1e-10)
        density = lambda value: log_density(model, (fit.fit_observation(), fit.specmodel),
                                           {}, dict(point, dilution=value))[0]
        value, gradient = jax.jit(jax.value_and_grad(density))(d)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        for index in range(d.size):
            delta = jnp.zeros_like(d).reshape(-1).at[index].set(1e-5).reshape(d.shape)
            finite_difference = (density(d + delta) - density(d - delta)) / 2e-5
            np.testing.assert_allclose(np.asarray(gradient).reshape(-1)[index], finite_difference,
                                       rtol=2e-6, atol=1e-7)


def test_omitted_and_zero_dilution_preserve_old_flux_and_density(x64_context):
    with x64_context():
        fit, settings, point = case()
        plain = configured(fit, settings)
        zero = configured(fit, dict(settings, dilution=0.))
        assert "dilution" not in plain.prior_trace()
        assert zero.prior_trace()["dilution"]["type"] == "deterministic"
        for name in ("physical_flux", "model_flux", "mean_flux"):
            np.testing.assert_array_equal(fit.predict(point=point, model=plain)[name],
                                          fit.predict(point=point, model=zero)[name])
        np.testing.assert_array_equal(
            log_density(plain, (fit.observation, fit.specmodel), {}, point)[0],
            log_density(zero, (fit.observation, fit.specmodel), {}, point)[0])


@pytest.mark.parametrize("dilution", [0., .2, np.array([.1, .3]), dist.Uniform(0., 1.)])
def test_factory_prior_and_returned_physical_params(dilution, x64_context):
    with x64_context():
        fit, _, _ = case()
        model = make_model_single(fit, resolving_power=70000., use_gp=False,
                                  rv_bounds=(10., 14.), vsini_max=20., dilution=dilution)
        trace = model.prior_trace()
        assert "dilution" in trace
        point = {name: site["value"] for name, site in trace.items()
                 if site["type"] == "sample" and not site["is_observed"]}
        prediction = fit.predict(point=point, model=model)
        np.testing.assert_array_equal(prediction["params"]["dilution"], trace["dilution"]["value"])


@pytest.mark.parametrize("bad", [-.01, 1., np.nan, np.inf, np.zeros(3), np.zeros((2, 1))])
def test_invalid_dilution_rejected_during_prior_inspection(bad):
    fit, _, _ = case()
    with pytest.raises(ValueError, match="dilution"):
        make_model_single(fit, resolving_power=70000., use_gp=False,
                          rv_bounds=(10., 14.), vsini_max=20., dilution=bad)


@pytest.mark.slow
def test_injected_dilution_svi_nuts_and_saved_reconstruction(tmp_path, x64_context):
    with x64_context():
        fit, settings, point = case()
        physical = fit.predict(point=point, model=configured(fit, settings))["physical_flux"]
        truth = .23
        measured = truth + (1 - truth) * np.asarray(physical)
        fit.observation = replace(fit.observation, flux=measured)
        settings.update(atmosphere={"teff": 5800.}, dilution=dist.Uniform(0., 1.))
        model = configured(fit, settings)
        estimate = fit.run_svi(model=model, rng_key=jax.random.PRNGKey(0),
            p_initial={"dilution": .01}, step_size=.05, num_steps=180, progress_bar=False)
        assert abs(float(estimate["dilution"]) - truth) < .05
        mcmc = fit.run_nuts(model=model, rng_key=jax.random.PRNGKey(1),
            num_warmup=40, num_samples=40, num_chains=1, progress_bar=False,
            nuts_kwargs={"init_strategy": init_to_value(values=estimate), "max_tree_depth": 4})
        assert abs(float(np.mean(mcmc.get_samples()["dilution"])) - truth) < .08
        output = save_fit_result(tmp_path/'diluted', sf=fit, model=model, point=estimate, mcmc=mcmc)
        saved = load_fit_result(output)
        np.testing.assert_array_equal(saved.samples["dilution"], mcmc.get_samples()["dilution"])
        assert saved.metadata["model"]["settings"]["dilution"]["parameters"] == {"low": 0., "high": 1.}
        assert "dilution" in saved.metadata["sampler"]["latent_names"]
        np.testing.assert_allclose(fit.predict(point=saved.point, model=model)["physical_flux"],
                                  fit.predict(point=estimate, model=model)["physical_flux"])
