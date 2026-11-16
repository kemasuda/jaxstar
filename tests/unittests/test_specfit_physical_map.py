"""Physical MAP must not inherit coordinate-dependent mode shifts."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit._inference import _PhysicalAutoDelta, find_map_svi
from test_specfit_inference import conditional_model


@pytest.mark.parametrize("x64", [False, True])
def test_beta_mode_is_physical_not_unconstrained(x64, x64_context):
    def model():
        x = numpyro.sample("x", dist.Beta(2., 5.))
        numpyro.deterministic("twice", 2*x)
    with x64_context(x64):
        result = find_map_svi(model, .03, 1200, rng_key=jax.random.PRNGKey(0),
                              p_initial={"x": .65, "twice": -10.}, progress_bar=False)
        # Physical Beta(2,5) mode is 1/5; the logit-coordinate mode is 2/7.
        np.testing.assert_allclose(result["x"], .2, atol=2e-5)
        np.testing.assert_allclose(result["twice"], .4, atol=4e-5)


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("x64", [False, True])
def test_loss_and_gradient_equal_native_joint_density(conditional, x64, x64_context):
    def fixed_model():
        x = numpyro.sample("x", dist.Beta(2., 5.))
        scale = numpyro.sample("scale", dist.LogUniform(.01, 1.))
        numpyro.sample("obs", dist.Normal(x, scale), obs=.2)
        numpyro.deterministic("twice", x*2)
    model = conditional_model if conditional else fixed_model
    initial = {"teff": 6400., "logg": 4.4} if conditional else {"x": .35, "scale": .2}
    with x64_context(x64):
        guide = _PhysicalAutoDelta(model, init_loc_fn=init_to_value(values=initial))
        assert isinstance(guide, AutoDelta)
        svi = SVI(model, guide, numpyro.optim.Adam(.02), Trace_ELBO())
        params = svi.get_params(svi.init(jax.random.PRNGKey(0)))
        if conditional:
            fraction = (6000.-4500.)/(7000.-4500.)
            params["teff_auto_unconstrained_loc"] = jnp.log(fraction/(1-fraction))
        def expected(p):
            point = guide.median(p)
            return -log_density(model, (), {}, {name:point[name] for name in initial})[0]
        def actual(p):
            return Trace_ELBO().loss(jax.random.PRNGKey(1), p, model, guide)
        value, grad = jax.jit(jax.value_and_grad(actual))(params)
        target, target_grad = jax.jit(jax.value_and_grad(expected))(params)
        tolerance = 3e-5 if not x64 else 1e-10
        np.testing.assert_allclose(value, target, rtol=tolerance, atol=tolerance)
        for name in grad:
            np.testing.assert_allclose(grad[name], target_grad[name], rtol=tolerance, atol=tolerance)
        native = guide.median(params)
        draws = guide.sample_posterior(None, params, sample_shape=(3,))
        for name, item in native.items():
            np.testing.assert_allclose(draws[name], np.broadcast_to(item, (3,)+np.shape(item)))
