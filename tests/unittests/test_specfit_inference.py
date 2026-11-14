"""Laplace SVI preserves native sites and model-dependent support transforms."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit import physical_logg_max, model_single
from jaxstar.specfit._inference import _LaplacePotentialLoss
from test_specfit import case
from test_numpyro_stellar_relations import stellar_case, derived_config


def conditional_model(*args):
    teff = numpyro.sample("teff", dist.Uniform(4500., 7000.))
    logg = numpyro.sample("logg", dist.Uniform(3.5, physical_logg_max(teff)))
    numpyro.deterministic("derived", teff / 1000. + logg)
    numpyro.sample("temperature", dist.Normal(teff, 10.), obs=6000.)
    numpyro.sample("gravity", dist.Normal(logg, .002), obs=4.4165)


def initialized(model, initial):
    guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=initial))
    svi = SVI(model, guide, numpyro.optim.Adam(.02), _LaplacePotentialLoss())
    state = svi.init(jax.random.PRNGKey(0))
    return guide, svi.get_params(state)


def move_temperature(guide, params, temperature, low, high):
    latent = guide._unpack_latent(params["auto_loc"])
    fraction = (temperature - low) / (high - low)
    latent["teff"] = jnp.asarray(np.log(fraction / (1 - fraction)))
    packed = jnp.concatenate([jnp.ravel(latent[name]) for name in guide._init_locs])
    return {**params, "auto_loc": packed}


def uniform_log_jacobian(value, low, high):
    fraction = (value - low) / (high - low)
    return jnp.log(high - low) + jnp.log(fraction) + jnp.log1p(-fraction)


@pytest.mark.parametrize("x64", [False, True])
def test_loss_uses_the_returned_conditional_point(x64, x64_context):
    with x64_context(x64):
        guide, params = initialized(conditional_model, {"teff": 6400., "logg": 4.4})
        params = move_temperature(guide, params, 6000., 4500., 7000.)

        def expected(p):
            point = guide.median(p)
            log_jacobian = uniform_log_jacobian(point["teff"], 4500., 7000.)
            log_jacobian += uniform_log_jacobian(
                point["logg"], 3.5, physical_logg_max(point["teff"]))
            return -log_density(conditional_model, (), {}, point)[0] - log_jacobian

        actual = lambda p: _LaplacePotentialLoss().loss(
            jax.random.PRNGKey(1), p, conditional_model, guide)
        value, gradient = jax.jit(jax.value_and_grad(actual))(params)
        wanted, wanted_gradient = jax.jit(jax.value_and_grad(expected))(params)
        np.testing.assert_allclose(value, wanted, rtol=3e-5 if not x64 else 1e-11)
        np.testing.assert_allclose(gradient["auto_loc"], wanted_gradient["auto_loc"],
                                   rtol=3e-5 if not x64 else 1e-11)


@pytest.mark.parametrize("x64", [False, True])
def test_fixed_support_preserves_loss_and_gradients(x64, x64_context):
    def model():
        scale = numpyro.param("scale", .4, constraint=dist.constraints.positive)
        x = numpyro.sample("x", dist.Uniform(-1., 2.))
        vector = numpyro.sample("vector", dist.LogUniform(
            jnp.full(2, .01), jnp.ones(2)).to_event(1))
        numpyro.sample("obs", dist.Normal(x + vector, scale), obs=jnp.array([.3, .6]))

    with x64_context(x64):
        guide, params = initialized(model, {"x": .1, "vector": jnp.array([.1, .2])})
        args = (jax.random.PRNGKey(1), params, model, guide)
        before, grad_before = jax.value_and_grad(Trace_ELBO().loss, argnums=1)(*args)
        after, grad_after = jax.value_and_grad(_LaplacePotentialLoss().loss, argnums=1)(*args)
        np.testing.assert_allclose(after, before, rtol=3e-6)
        for name in grad_before:
            np.testing.assert_allclose(grad_after[name], grad_before[name], rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("x64", [False, True])
def test_public_svi_conditional_multistart_native_point(x64, x64_context):
    with x64_context(x64):
        fit, _, _ = case(single=True)
        points = [fit.run_svi(
            model=conditional_model, rng_key=jax.random.PRNGKey(0),
            step_size=.02, num_steps=2000, progress_bar=False,
            p_initial={"teff": teff, "logg": 4.4, "derived": -99.})
            for teff in (6178., 6400.)]
        for point in points:
            assert set(point) == {"teff", "logg", "derived"}
            np.testing.assert_allclose(point["teff"], 6000., atol=.2)
            np.testing.assert_allclose(point["logg"], 4.4165, atol=3e-5)
            np.testing.assert_allclose(point["derived"], point["teff"] / 1000. + point["logg"])
        np.testing.assert_allclose(points[0]["logg"], points[1]["logg"], atol=2e-6)


@pytest.mark.slow
@pytest.mark.parametrize("x64", [False, True])
def test_real_spectral_loss_uses_returned_logg_and_vmic(x64, x64_context):
    with x64_context(x64):
        obs, specmodel, config, coordinates = stellar_case.__wrapped__()
        config = derived_config(config, use_physical_logg_max=True)
        model = partial(model_single, obs, specmodel, **config)
        guide, params = initialized(model, {"teff": 6400., "logg": 4.4165, "mh": -.1})
        params = move_temperature(guide, params, 6000., 5000., 6500.)
        point = guide.median(params)
        log_jacobian = uniform_log_jacobian(point["teff"], 5000., 6500.)
        log_jacobian += uniform_log_jacobian(point["logg"], 3.8, physical_logg_max(point["teff"]))
        log_jacobian += uniform_log_jacobian(point["mh"], -.3, .3)
        # Replay only sample sites, so derived vmic and the spectrum are recomputed.
        native = {name: point[name] for name in ("teff", "logg", "mh")}
        density, trace = log_density(model, (), {}, native)
        np.testing.assert_allclose(point["vmic"], trace["vmic"]["value"])
        np.testing.assert_allclose(point["model_flux"], trace["model_flux"]["value"])
        loss = _LaplacePotentialLoss().loss(jax.random.PRNGKey(1), params, model, guide)
        np.testing.assert_allclose(loss, -density - log_jacobian, atol=2e-5 if not x64 else 1e-10)
