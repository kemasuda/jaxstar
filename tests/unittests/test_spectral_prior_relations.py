"""Empirical relations and their JAX contracts, without a spectral grid."""

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
    empirical_vmacro_doyle2014, empirical_vmacro_lubin2026,
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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vmic_three_branches_and_jax_transforms(dtype, x64_context):
    # Bergemann & Hoppe (2026): solar-like dwarf, F dwarf, cool dwarf, red giant.
    with x64_context(dtype == "float64"):
        inputs = tuple(jnp.asarray(value, dtype=dtype) for value in (
            [5772., 6500., 4500., 4500.], [4.44, 4.2, 4.6, 2.], [0., -.5, -.5, -1.]))
        tolerance = 3e-7 if dtype == "float32" else 1e-14
        for evaluate in (empirical_vmic, jax.jit(empirical_vmic), jax.vmap(empirical_vmic)):
            actual = evaluate(*inputs)
            assert actual.dtype == jnp.dtype(dtype)
            np.testing.assert_allclose(actual, [1.0668016, 1.4003, .8537, 1.379],
                                       rtol=tolerance)


def test_vmic_broadcasts_stellar_inputs():
    teff = jnp.array([[4500.], [5500.], [6000.]])
    actual = jax.jit(empirical_vmic)(teff, jnp.array([4., 4.5, 2.]),
                                    jnp.array([0., .3, -1.]))
    np.testing.assert_allclose(actual, [[.962, .90665, 1.379], [1.05, .99465, 1.47],
                                       [1.213, 1.15765, 1.748]], rtol=3e-7)


def test_vmic_temperature_boundary_is_continuous(x64_context):
    with x64_context():
        teff = jnp.array([4999.999, 5000., 5000.001])
        actual = jax.jit(empirical_vmic)(teff, 4.5, 0.)
        np.testing.assert_allclose(actual, [.89075, .89075, .89075010100015], rtol=1e-14)


@pytest.mark.parametrize("teff,dwarf,giant_limit", [
    (4500., 1.03075, 1.22775),
    (5000., 1.03075, 1.19575),
    (5500., 1.11875, 1.31875),
    (6500., 1.51975, 2.02975),
])
def test_vmic_gravity_boundary_preserves_published_jump(teff, dwarf, giant_limit, x64_context):
    with x64_context():
        logg = jnp.array([np.nextafter(3.5, -np.inf), 3.5, np.nextafter(3.5, np.inf)])
        actual = jax.jit(empirical_vmic)(teff, logg, 0.)
        np.testing.assert_allclose(actual, [giant_limit, dwarf, dwarf], rtol=1e-14)


@pytest.mark.parametrize("relation,point,expected_gradient", [
    (physical_logg_max, (5770.,), (-0.000112702907538,)),
    (empirical_vmic, (6000., 4.5, .3), (.000401, -.145, .056)),
    (empirical_vmic, (4500., 4.6, -.5), (0., -.146, .04)),
    (empirical_vmic, (4500., 2., -1.), (-.000219, -.12, .03)),
    (empirical_vmic, (5000., 4.5, 0.), (.000101, -.145, .05)),
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


@pytest.mark.parametrize("relation,temperatures,gravities,expected,sigma", [
    (empirical_vmacro_doyle2014, [5200., 5777., 6400.], [4.6, 4.44, 4.0],
     [2.211448, 3.21, 6.317848], .37),
    (empirical_vmacro_lubin2026, [5772., 6000., 5300.], [4.44, 4.2, 4.5],
     [3.02, 4.021878413012682, 2.564165946055237], .09),
])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_teff_logg_vmacro_priors(relation, temperatures, gravities, expected, sigma,
                               dtype, x64_context):
    with x64_context(dtype == "float64"):
        teff, logg = (jnp.asarray(values, dtype=dtype) for values in (temperatures, gravities))
        tolerance = 2e-5 if dtype == "float32" else 1e-12
        prior = relation(teff, logg)
        np.testing.assert_allclose(prior.base_dist.loc, expected, rtol=tolerance)
        np.testing.assert_allclose(prior.base_dist.scale, sigma, rtol=tolerance)
        np.testing.assert_array_equal(prior.low, 0.)
        location = lambda t, g: relation(t, g).base_dist.loc
        for evaluate in (jax.jit(location), jax.vmap(location)):
            actual = evaluate(teff, logg)
            assert actual.dtype == jnp.dtype(dtype)
            np.testing.assert_allclose(actual, expected, rtol=tolerance)
        values = jnp.asarray(expected, dtype=dtype) + .1
        expected_logp = truncnorm.logpdf(values, -np.asarray(expected) / sigma, np.inf,
                                         loc=expected, scale=sigma)
        logp = jax.jit(lambda t, g, v: relation(t, g).log_prob(v))
        np.testing.assert_allclose(logp(teff, logg, values), expected_logp, rtol=tolerance)
        samples = jax.jit(lambda t, g, key: relation(t, g).sample(key, (32,)))(
            teff, logg, jax.random.PRNGKey(0))
        assert samples.shape == (32, 3)
        assert np.all(np.isfinite(samples)) and np.all(samples >= 0.)


@pytest.mark.parametrize("relation", [empirical_vmacro_doyle2014, empirical_vmacro_lubin2026])
def test_teff_logg_vmacro_broadcasts_sigma_and_has_jit_gradients(relation):
    teff = jnp.array([[5772.], [6000.]])
    logg = jnp.array([4.0, 4.44, 4.6])
    sigma = jnp.array([.2, .4, .6])
    prior = relation(teff, logg, sigma=sigma)
    assert prior.batch_shape == (2, 3) and prior.event_shape == ()
    np.testing.assert_array_equal(jnp.broadcast_to(prior.base_dist.scale, (2, 3)),
                                  jnp.broadcast_to(sigma, (2, 3)))
    density = lambda t, g, s: relation(t, g, sigma=s).log_prob(4.2).sum()
    value, gradients = jax.jit(jax.value_and_grad(density, argnums=(0, 1, 2)))(teff, logg, sigma)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(g)) and np.any(g != 0.) for g in gradients)


def test_lubin_uses_solar_ratio_and_does_not_clip_location():
    ratios = jnp.array([.4, 1., 1.4, 3.])
    prior = empirical_vmacro_lubin2026(5772., 4.44 + jnp.log10(ratios))
    np.testing.assert_allclose(prior.base_dist.loc, [4.4, 3.02, 2.1, -1.58], rtol=3e-6)
    assert np.all(prior.low == 0.)


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
