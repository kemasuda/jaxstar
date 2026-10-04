"""Frozen legacy relations and their JAX contracts, without a spectral grid."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.util import log_density
import pytest
from scipy.stats import truncnorm

from jaxstar.specfit import (
    physical_logg_max, empirical_vmic, empirical_vmacro_valenti_fischer2005,
)


# Values evaluated from jaxspec e7b2f30:src/jaxspec/numpyro_model.py.
# The reference stays independent of the migrated implementation and jaxspec imports.
@pytest.mark.parametrize("relation,inputs,expected", [
    (physical_logg_max, ([4500., 5500., 5770., 7000.],),
     [4.768690574575, 4.692121995575, 4.663402725182871, 4.4892796907]),
    (empirical_vmic,
     ([5200., 5500., 6000., 6500.], [3.6, 4., 4.5, 4.2], [-.8, 0., .3, -.5]),
     [1.0098, 1.05, 1.15765, 1.4003]),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_frozen_legacy_values_and_jax_transforms(relation, inputs, expected, dtype, x64_context):
    with x64_context(dtype == "float64"):
        inputs = tuple(jnp.asarray(value, dtype=dtype) for value in inputs)
        tolerance = 3e-7 if dtype == "float32" else 1e-14
        for evaluate in (relation, jax.jit(relation), jax.vmap(relation)):
            actual = evaluate(*inputs)
            assert actual.dtype == jnp.dtype(dtype)
            np.testing.assert_allclose(actual, expected, rtol=tolerance)


def test_vmic_broadcasts_stellar_inputs():
    teff = jnp.array([[5500.], [6000.]])
    # At logg=4, feh=0 the legacy relation is 1.05 and 1.213 km/s.
    # Metallicity and gravity offsets are shared by each temperature row.
    actual = jax.jit(empirical_vmic)(teff, jnp.array([4., 4.5]), jnp.array([0., .3]))
    np.testing.assert_allclose(actual, [[1.05, .99465], [1.213, 1.15765]], rtol=3e-7)


@pytest.mark.parametrize("relation,point,expected_gradient", [
    (physical_logg_max, (5770.,), (-0.000112702907538,)),
    (empirical_vmic, (6000., 4.5, .3), (.000401, -.145, .056)),
])
def test_scalar_relations_have_finite_jit_gradients(relation, point, expected_gradient):
    value, gradient = jax.jit(jax.value_and_grad(
        relation, argnums=tuple(range(len(point)))))(*point)
    assert value.shape == () and np.isfinite(value)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=5e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vmacro_prior_matches_legacy_location_and_scatter(dtype, x64_context):
    with x64_context(dtype == "float64"):
        teff = jnp.array([4500., 5500., 5770., 7000.], dtype=dtype)
        expected_location = np.array([2.0261538461538464, 3.5646153846153847,
                                      3.98, 5.872307692307692])
        tolerance = 2e-6 if dtype == "float32" else 1e-13
        prior = empirical_vmacro_valenti_fischer2005(teff)
        assert prior.batch_shape == (4,) and prior.event_shape == ()
        np.testing.assert_allclose(prior.base_dist.loc, expected_location, rtol=tolerance)
        np.testing.assert_array_equal(jnp.broadcast_to(prior.base_dist.scale, (4,)), jnp.ones(4))
        np.testing.assert_array_equal(jnp.broadcast_to(prior.low, (4,)), jnp.zeros(4))
        values = jnp.array([.1, 3., 4., 8.], dtype=dtype)
        expected_logp = truncnorm.logpdf(values, -expected_location, np.inf,
                                         loc=expected_location, scale=1.)
        logp = jax.jit(lambda t, v: empirical_vmacro_valenti_fischer2005(t).log_prob(v))
        np.testing.assert_allclose(logp(teff, values), expected_logp, rtol=tolerance)
        samples = jax.jit(lambda t, key: empirical_vmacro_valenti_fischer2005(t).sample(
            key, (32,)))(teff, jax.random.PRNGKey(0))
        assert samples.shape == (32, 4)
        assert np.all(np.isfinite(samples)) and np.all(samples >= 0.)


def test_vmacro_prior_broadcasts_custom_scatter_and_has_jit_gradients():
    teff = jnp.array([[5770.], [6420.]])
    sigma = jnp.array([.5, 1.5])
    prior = empirical_vmacro_valenti_fischer2005(teff, sigma=sigma)
    assert prior.batch_shape == (2, 2)
    np.testing.assert_allclose(jnp.broadcast_to(prior.base_dist.loc, (2, 2)),
                               [[3.98, 3.98], [4.98, 4.98]], rtol=1e-7)
    np.testing.assert_allclose(jnp.broadcast_to(prior.base_dist.scale, (2, 2)),
                               [[.5, 1.5], [.5, 1.5]])
    def density(t, s):
        return empirical_vmacro_valenti_fischer2005(t, sigma=s).log_prob(4.2).sum()
    value, gradients = jax.jit(jax.value_and_grad(density, argnums=(0, 1)))(teff, sigma)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(g)) and np.any(g != 0.) for g in gradients)


def test_vmacro_prior_in_custom_model_has_only_explicit_sample_sites():
    def model():
        teff = numpyro.sample("teff", dist.Uniform(4500., 7000.))
        numpyro.sample("vmacro", empirical_vmacro_valenti_fischer2005(teff))

    values = {"teff": 5770., "vmacro": 4.2}
    trace = handlers.trace(handlers.substitute(model, data=values)).get_trace()
    assert list(trace) == ["teff", "vmacro"]
    np.testing.assert_allclose(trace["vmacro"]["fn"].base_dist.loc, 3.98, rtol=1e-7)
    value, gradient = jax.jit(jax.value_and_grad(
        lambda p: log_density(model, (), {}, p)[0]))(values)
    assert np.isfinite(value)
    assert all(np.isfinite(g) and g != 0. for g in gradient.values())
