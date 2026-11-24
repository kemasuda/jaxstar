"""Whole-region white/white mixture: dense reference, masks and inference."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer.util import log_density
from scipy.stats import multivariate_normal
import pytest

from jaxstar.specfit import (Observation, SpecFit, make_model_region_mixture,
                            region_mixture_statistics, chebyshev_basis)
from test_specfit import case

@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def test_region_density_matches_dense_integral_and_empty_region():
    wave = np.tile(np.linspace(5000, 5001, 9), (3, 1))
    model = np.tile(1-.4*np.exp(-np.linspace(-2, 2, 9)**2), (3, 1))
    y = model.copy(); y[1, 4:7] += .15
    mask = np.zeros_like(y, bool); mask[0, 2] = True; mask[2] = True
    y[mask] = np.nan
    error = np.full_like(y, .01); error[mask] = np.inf
    obs = Observation(wave, y, error, mask)
    opts = dict(mismatch_fraction=.2, mismatch_scale=.1, jitter=np.array([.003,.004,.005]),
                sigma_constant=.03, sigma_continuum=.02, degree=1)
    stats = region_mixture_statistics(obs, model, **opts)
    basis = np.asarray(chebyshev_basis(wave, 1))
    expected = np.zeros((3, 2))
    for r in (0, 1):
        keep = ~mask[r]
        design = model[r, keep, None]*basis[r, keep]
        for component, extra in enumerate((0., .1)):
            covariance = np.eye(sum(keep))*(.01**2+opts['jitter'][r]**2+extra**2) + (design*np.array([.03,.02])**2)@design.T
            expected[r, component] = multivariate_normal.logpdf(y[r, keep], mean=design[:, 0], cov=covariance)
    total = np.logaddexp(np.log(.8)+expected[:,0], np.log(.2)+expected[:,1])
    np.testing.assert_allclose(stats.component_log_likelihood, expected, atol=2e-10)
    np.testing.assert_allclose(stats.log_likelihood, total, atol=2e-10)
    np.testing.assert_allclose(stats.region_mismatch_probability, np.exp(np.log(.2)+expected[:,1]-total), atol=2e-10)
    assert stats.region_mismatch_probability[0] < .01
    assert stats.region_mismatch_probability[1] > .99
    for fraction in (0., 1.):
        limit = region_mixture_statistics(obs, model, **{**opts, 'mismatch_fraction': fraction})
        np.testing.assert_allclose(limit.log_likelihood, expected[:, int(fraction)])
        np.testing.assert_allclose(limit.region_mismatch_probability, fraction)
    gradient = jax.grad(lambda scale: region_mixture_statistics(obs, model*scale, **opts).log_likelihood.sum())(1.)
    assert np.isfinite(gradient)


def test_configured_prediction_and_draw_average():
    sf, _, _ = case()
    options = dict(resolving_power=70000., rv_bounds=(9.,15.), vsini_max=12., degree=1,
                   fixed={'q1':.36, 'q2':.3}, mismatch_fraction=.2, mismatch_scale=.08)
    model = make_model_region_mixture(sf, **options)
    trace = model.prior_trace()
    point = {name: site['value'] for name,site in trace.items() if site['type']=='sample' and not site['is_observed']}
    point.update(teff=5800., vsini=6., vmacro=3., rv=jnp.array([12.,12.]), depth_scale=jnp.ones(2), jitter=.004)
    assert 'continuum_coefficients' not in trace
    prediction = sf.predict(point=point, model=model)
    assert prediction['region_mismatch_probability'].shape == (2,)
    assert prediction['continuum_posterior'].mean.shape == (2,2)
    assert np.isnan(prediction['mismatch_probability'][np.asarray(sf.observation.mask)]).all()
    np.testing.assert_allclose(prediction['model_flux'], prediction['physical_flux']*prediction['continuum'])
    np.testing.assert_allclose(prediction['mean_flux'], prediction['model_flux'])
    samples = {name:jnp.stack([value,value]) for name,value in point.items()}
    average = sf.predict(samples=samples, model=model)
    np.testing.assert_allclose(average['region_mismatch_probability'], prediction['region_mismatch_probability'])
    for name in ('teff','depth_scale'):
        grad = jax.grad(lambda x: log_density(model, (sf.observation,sf.specmodel), {}, {**point,name:x})[0])(jnp.asarray(point[name]))
        assert np.isfinite(grad).all()
    with pytest.raises(ValueError, match='white noise'):
        make_model_region_mixture(sf, **options, use_gp=True)


@pytest.mark.slow
def test_prefit_and_nuts_use_same_region_model():
    from numpyro.infer import init_to_value
    sf, _, _ = case(single=True)
    options = dict(resolving_power=70000., rv_bounds=(10.,14.), vsini_max=12., degree=1,
        fixed=dict(teff=5800.,vsini=6.,vmacro=3.,q1=.36,q2=.3,jitter=.004),
        depth_scale=dist.LogNormal(0.,.3).expand([1]), mismatch_fraction=.1, mismatch_scale=.08)
    model, point = sf.run_region_mixture_prefit(rng_key=jax.random.PRNGKey(5),
        model_kwargs=options, p_initial={'rv':jnp.array([11.5]),'depth_scale':jnp.array([.9])},
        num_steps=150, progress_bar=False)
    assert abs(float(point['rv'][0])-12.) < .3
    assert abs(float(point['depth_scale'][0])-1.) < .1
    mcmc = sf.run_nuts(model=model,rng_key=jax.random.PRNGKey(6),num_warmup=10,num_samples=5,
        progress_bar=False,nuts_kwargs={'init_strategy':init_to_value(values=point),'max_tree_depth':3})
    pred = sf.predict(samples=mcmc.get_samples(),model=model)
    assert np.isfinite(pred['region_mismatch_probability']).all()
