"""Independent inference entry points; synthetic grids and no native MultiNest."""

from dataclasses import FrozenInstanceError
import builtins

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density
import pytest

from jaxstar.mistfit import (MistFit, MistObservations, make_init_strategy,
                             model_isochrone, run_nuts, run_multinest)
from test_mistmodel import grid_path, make_fit, OPTIONS, PARAMS


def test_observations_copy_input_and_fit_has_no_mutable_data(grid_path):
    keys, values, errors = ['teff', 'parallax'], np.array([5400., 8.4]), [.1, .1]
    observations = MistObservations(keys, values, errors)
    fit = MistFit(grid_path, observations=observations)
    keys[0], values[0], errors[0] = 'mass', 1., 2.
    assert observations.keys == ('teff', 'parallax')
    assert observations.values == (5400., 8.4)
    assert observations.errors == (.1, .1)
    assert observations == MistObservations.from_mapping({'teff': (5400., .1), 'parallax': (8.4, .1)})
    with pytest.raises(FrozenInstanceError):
        fit.observations = observations
    with pytest.raises(FrozenInstanceError):
        observations.values = ()
    with pytest.raises(ValueError, match='unknown MistModel observation'):
        MistFit(model=fit.physical_model, observations={'typo': (1., .1)})


@pytest.mark.parametrize('keys,values,errors', [
    (['teff'], [], []), (['teff', 'teff'], [1., 2.], [.1, .1]),
    ([''], [1.], [.1]), ([1], [1.], [.1]),
    (['teff'], [np.nan], [.1]), (['teff'], [1.], [np.inf]),
    (['teff'], [1.], [0.]), (['teff'], [1.], [-1.]),
    (['teff'], [[1.]], [.1]), (['teff'], [1.j], [.1]),
])
def test_invalid_observations_fail_before_sampling(keys, values, errors):
    with pytest.raises(ValueError):
        MistObservations(keys, values, errors)


@pytest.mark.parametrize('parallax', [8.4, 0., -1.])
@pytest.mark.parametrize('linear_age', [True, False])
def test_initialization_matches_historical_rule(grid_path, parallax, linear_age, x64_context):
    with x64_context():
        fit = make_fit(MistFit, grid_path, 'named')
        obs = MistObservations(('parallax', 'feh', 'age'), (parallax, -.15, 2.), (.1, .1, .1))
        fit = MistFit(model=fit.physical_model, observations=obs)
        # Historical setup_hmc passed measured keys verbatim, so surface feh
        # must NOT become the initial-FeH start; missing sites keep the fallback.
        old_strategy = init_to_value(values={'distance': 1. / parallax if parallax > 0 else 8.,
                                             'feh': -.15, 'age': 2.})
        options = dict(OPTIONS, linear_age=linear_age, nodata=True)
        new = initialize_model(jax.random.PRNGKey(13), fit.model,
                               init_strategy=make_init_strategy(obs), model_kwargs=options)
        old = initialize_model(jax.random.PRNGKey(13), fit.model,
                               init_strategy=old_strategy, model_kwargs=options)
        for a, b in zip(jax.tree_util.tree_leaves(new.param_info), jax.tree_util.tree_leaves(old.param_info)):
            np.testing.assert_array_equal(a, b)
        constrained = new.postprocess_fn(new.param_info.z)
        np.testing.assert_allclose(constrained['distance'], 1. / parallax if parallax > 0 else 8.)
        if linear_age:
            np.testing.assert_allclose(constrained['age'], 2.)
        assert constrained['feh_init'] != -.15


def test_explicit_initialization_allows_prior_run_without_parallax(grid_path):
    fit = make_fit(MistFit, grid_path, 'named')
    fit = MistFit(model=fit.physical_model, observations={})
    with pytest.raises(ValueError, match='supply init_strategy'):
        fit.run_nuts(jax.random.PRNGKey(1))
    options = dict(OPTIONS, linear_age=False, nodata=True, output_keys=('kmag_abs',))
    result = fit.run_nuts(jax.random.PRNGKey(1), init_strategy=init_to_value(values=PARAMS),
                          model_kwargs=options, output_keys=('rho',),
                          num_warmup=4, num_samples=4, num_chains=1,
                          max_tree_depth=2, progress_bar=False)
    samples = result.get_samples()
    assert {'rho', 'kmag_abs'} <= set(samples)
    assert options['output_keys'] == ('kmag_abs',)
    assert np.all(np.isfinite(samples['rho']))


def test_standalone_isochrone_nuts_accepts_observations_and_physical_model(grid_path):
    fit = make_fit(MistFit, grid_path, 'named')
    result = run_nuts(model_isochrone, jax.random.PRNGKey(7),
                      model_args=(fit.observations, fit.physical_model),
                      model_kwargs=dict(OPTIONS, linear_age=False, output_keys=('rho',)),
                      init_strategy=init_to_value(values=PARAMS), num_chains=1,
                      num_warmup=4, num_samples=4, max_tree_depth=2, progress_bar=False)
    samples = result.get_samples()
    assert samples['rho'].shape == (4,)
    np.testing.assert_allclose(samples['rho'], samples['star_mass'] / samples['radius']**3, rtol=1e-6)


def test_standalone_nuts_accepts_model_args_and_keeps_extra_fields():
    def gaussian(center, *, width):
        numpyro.sample('x', dist.Normal(center, width))
    # Verify vectorized chain forwarding with a cheap independent model.
    options = {'width': .3}
    result = run_nuts(gaussian, jax.random.PRNGKey(4), model_args=(1.,), model_kwargs=options,
                      num_warmup=4, num_samples=5, num_chains=2, chain_method='vectorized',
                      extra_fields=('potential_energy',), progress_bar=False)
    assert result.get_samples(group_by_chain=True)['x'].shape == (2, 5)
    assert result.get_extra_fields(group_by_chain=True)['potential_energy'].shape == (2, 5)
    assert options == {'width': .3}


def test_multinest_unavailable_runner_has_actionable_error(monkeypatch):
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name.startswith('numpyro_inferutils'):
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', unavailable)
    with pytest.raises(ImportError, match=r'numpyro-inferutils with experimental\.MultiNestRunner support'):
        run_multinest(lambda: None, outputfiles_basename='unused/')


@pytest.mark.parametrize('x64', [False, True])
def test_multinest_gamma_quantiles_without_tfp_preserve_explicit_overrides(
        tmp_path, monkeypatch, x64_context, x64):
    from scipy.special import gammainc
    from jaxstar.mistfit import get_run_metadata
    mn = pytest.importorskip('numpyro_inferutils.experimental.multinest')
    original = builtins.__import__

    def without_tfp(name, *args, **kwargs):
        if name.startswith('tensorflow_probability'):
            pytest.fail('MultiNest Gamma transforms must not require TFP')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_tfp)

    def model():
        numpyro.sample('distance', dist.Gamma(3, rate=.7))
        numpyro.sample('vector', dist.Gamma(jnp.array([.5, 2.7]),
                                           rate=jnp.array([2., .3])).to_event(1))
        numpyro.sample('custom', dist.Gamma(2., rate=1.))

    def fake_solve(**kwargs):
        theta = kwargs['Prior'](np.full(kwargs['n_dims'], .5))
        assert np.isfinite(kwargs['LogLikelihood'](theta))
        return {'samples': np.stack([theta, theta])}

    monkeypatch.setattr(mn, '_import_pymultinest', lambda: (fake_solve, None))
    overrides = {'custom': lambda q: jnp.full_like(q, 1.25)}
    with x64_context(x64):
        runner = run_multinest(model, outputfiles_basename=str(tmp_path / 'gamma-'),
                                site_overrides=overrides, n_live_points=8, resume=False)
        for probability in (.0001, .05, .5, .95, .9999):
            theta = runner.prior(np.full(runner.ndim, probability))
            np.testing.assert_allclose(gammainc([3., .5, 2.7], theta[:3] * [.7, 2., .3]),
                                       probability, rtol=2e-6 if not x64 else 1e-12,
                                       atol=1e-8 if not x64 else 1e-14)
            assert theta[3] == 1.25
        assert runner.site_overrides['custom'] is overrides['custom']
        assert set(overrides) == {'custom'}
        recorded = get_run_metadata(runner)['site_overrides']
        assert set(recorded) == {'distance', 'vector', 'custom'}


@pytest.mark.parametrize('linear_age', [False, True])
def test_multinest_real_runner_uses_same_joint_and_deterministics(grid_path, tmp_path, monkeypatch,
                                                                x64_context, linear_age):
    mn = pytest.importorskip('numpyro_inferutils.experimental.multinest')
    calls = []
    def fake_solve(**kwargs):
        # Only the native sampler bridge is substituted. The actual runner
        # constructs the prior transform and compiled log probability functions.
        calls.append(kwargs)
        theta = kwargs['Prior'](np.full(kwargs['n_dims'], .5))
        assert np.isfinite(kwargs['LogLikelihood'](theta))
        return {'samples': np.stack([theta, theta])}
    monkeypatch.setattr(mn, '_import_pymultinest', lambda: (fake_solve, None))
    with x64_context():
        fit = make_fit(MistFit, grid_path, 'named')
        options = dict(OPTIONS, linear_age=linear_age, flat_age_marginal=True,
                       prot=18., rho_obs=1.1, rho_err=.2, age_obs=2., age_err=.3)
        runner = fit.run_multinest(outputfiles_basename=str(tmp_path / 'chains' / 'fit-'),
                                   model_kwargs=options, output_keys=('rho', 'kmag_abs'),
                                   n_live_points=12, resume=False)
        assert runner.result['samples'].shape == (2, 4)
        assert calls[0]['n_live_points'] == 12 and calls[0]['resume'] is False
        assert 'output_keys' not in options
        params = dict(PARAMS)
        if linear_age:
            params['age'] = 10**params.pop('logage') / 1e9
        theta = np.array([params[name] for name in runner.site_names])
        joint, trace = log_density(model_isochrone, (fit.observations, fit.physical_model), options, params)
        observed = sum(site['fn'].log_prob(site['value']).sum() for site in trace.values()
                       if site['type'] == 'sample' and site['is_observed'])
        # The mass Jacobian and optional factors must appear once in the target.
        np.testing.assert_allclose(runner.logposterior(theta), joint, rtol=1e-12)
        np.testing.assert_allclose(runner.loglikelihood(theta), observed, rtol=1e-12)
        samples = {name: jnp.array([params[name], params[name]]) for name in runner.site_names}
        derived = runner.get_deterministic_samples(samples, site_names=['rho', 'kmag_abs', 'mass'])
        predictions = fit.physical_model(PARAMS, keys=tuple(derived))
        for key in derived:
            np.testing.assert_allclose(derived[key], predictions[key], rtol=1e-12)
        direct = run_multinest(model_isochrone, model_args=(fit.observations, fit.physical_model),
                               model_kwargs=options, outputfiles_basename=str(tmp_path / 'direct-'),
                               n_live_points=12, resume=False)
        np.testing.assert_allclose(direct.logposterior(theta), joint, rtol=1e-12)
