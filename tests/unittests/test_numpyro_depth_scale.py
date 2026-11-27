"""Region depth scales share SpecModel physics, likelihood and reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit import (make_model_single, save_fit_result, load_fit_result,
                             marginalized_continuum_log_likelihood)
from test_specfit import case, configured


@pytest.mark.parametrize('mode', ['gaussian', 'vecchia1', 'pixel_mixture'])
def test_depth_scale_reconstruction_and_joint_atmosphere_gradient(mode, x64_context):
    with x64_context():
        fit, settings, point = case()
        plain = configured(fit, settings)
        physical = fit.predict(point=point, model=plain)['physical_flux']
        scales = jnp.array([.7, 1.4])
        settings.update(depth_scale=dist.LogNormal(jnp.zeros(2), .3))
        point = dict(point, depth_scale=scales)
        if mode == 'vecchia1':
            settings.update(use_gp=True, gp_solver='vecchia1', gp_amplitude=.015, gp_scale=.3)
        elif mode == 'pixel_mixture':
            settings.update(likelihood='gaussian_mixture', use_gp=False,
                            mismatch_fraction=.08, mismatch_scale=.1)
            point['continuum_coefficients'] = jnp.broadcast_to(jnp.array([1.01, -.015, 0., 0., 0.]), (2, 5))
        model = configured(fit, settings)
        prediction = fit.predict(point=point, model=model)
        np.testing.assert_allclose(prediction['physical_flux'], 1 + scales[:, None] * (physical - 1), atol=1e-13)
        np.testing.assert_allclose(prediction['model_flux'], prediction['physical_flux'] * prediction['continuum'])
        _, trace = log_density(model, (fit.fit_observation(), fit.specmodel), {}, point)
        if mode == 'gaussian':
            expected = marginalized_continuum_log_likelihood(fit.fit_observation(), prediction['physical_flux'],
                jitter=.004, sigma_constant=.1, sigma_continuum=.03, basis=settings['basis'])
            np.testing.assert_allclose(trace['spectrum']['fn'].log_factor, expected, atol=1e-9)
        def density(x):
            return log_density(model, (fit.fit_observation(), fit.specmodel), {},
                               dict(point, teff=x[0], depth_scale=x[1:]))[0]
        x = jnp.array([5800., .7, 1.4])
        value, gradient = jax.jit(jax.value_and_grad(density))(x)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        for i, step in enumerate((.01, 1e-5, 1e-5)):
            delta = jnp.zeros(3).at[i].set(step)
            expected = (density(x + delta) - density(x - delta)) / (2 * step)
            np.testing.assert_allclose(gradient[i], expected, rtol=2e-5, atol=1e-6)


def test_factory_fixed_shared_and_per_region_scales_save_predict(tmp_path, x64_context):
    with x64_context():
        fit, _, _ = case()
        for depth in (None, 0., 1.2, np.array([0., 1.3]), dist.LogNormal(0., .2),
                      dist.LogNormal(jnp.zeros(2), .2)):
            model = make_model_single(fit, resolving_power=70000., use_gp=False,
                rv_bounds=(np.full(2, 10.), np.full(2, 14.)), vsini_max=20.,
                depth_scale=depth, fixed={'q1': .36, 'q2': .3})
            trace = model.prior_trace()
            assert trace['teff']['type'] == 'sample'
            assert ('depth_scale' in trace) == (depth is not None)
            point = {name: site['value'] for name, site in trace.items()
                     if site['type'] == 'sample' and not site['is_observed']}
            prediction = fit.predict(point=point, model=model)
            assert np.isfinite(prediction['mean_flux']).all()
            if depth is not None:
                np.testing.assert_array_equal(prediction['params']['depth_scale'], trace['depth_scale']['value'])
        output = save_fit_result(tmp_path/'depth_fit', sf=fit, model=model, point=point)
        restored = load_fit_result(output)
        np.testing.assert_array_equal(restored.point['depth_scale'], point['depth_scale'])
        np.testing.assert_allclose(fit.predict(point=restored.point, model=model)['mean_flux'], prediction['mean_flux'])


def test_depth_applies_to_all_stages_and_component_decomposition(x64_context):
    with x64_context():
        fit, settings, point = case()
        params = fit.specmodel.make_params(dict(point, **settings['broadening'], rv=12., resolving_power=70000.))
        scaled = dict(params, depth_scale=jnp.array([.8, 1.4]))
        for stage in ('intrinsic', 'broadened', 'full'):
            evaluate = getattr(fit.specmodel, stage)
            flux = evaluate(params, fit.observation.wavelength)
            actual = evaluate(scaled, fit.observation.wavelength)
            np.testing.assert_allclose(actual, 1 + scaled['depth_scale'][:, None] * (flux - 1))
            decomposition = fit.specmodel.decompose(scaled, fit.observation.wavelength, stage=stage)
            np.testing.assert_allclose(decomposition.total, actual)
            np.testing.assert_allclose(decomposition.total,
                                      decomposition.dilution[:, None] + decomposition.weighted_components.sum(axis=0))
        flat = fit.specmodel.make_params(dict(point, **settings['broadening'], rv=12., resolving_power=70000., depth_scale=1.3))
        assert flat['depth_scale'] == 1.3


@pytest.mark.parametrize('bad', [[1., 1., 1.], [[1.], [1.]]])
def test_invalid_depth_scale_shape_is_rejected(bad):
    fit, _, _ = case()
    with pytest.raises(ValueError, match='depth_scale'):
        make_model_single(fit, resolving_power=70000., use_gp=False,
                          rv_bounds=(10., 14.), vsini_max=20., depth_scale=bad)


@pytest.mark.slow
def test_cropped_regions_svi_nuts_and_posterior_reconstruction(tmp_path, x64_context):
    from dataclasses import replace
    from numpyro.infer import init_to_value
    from jaxstar.specfit import (SpecFit, extract_regions, save_spectral_grid, single_star_params)
    with x64_context():
        original, _, _ = case()
        path = save_spectral_grid(tmp_path/'grid.npz', original.specmodel.spectra)
        obs, mapping = extract_regions(original.observation,
            wavelength_ranges=[[4999.5, 5001.0], [5999.4, 6001.2]], regions=('a', 'b'))
        fit = SpecFit.from_grid(path, obs, crop=True, wav_margin=1., vmax=30.,
                               source_regions=mapping.source_regions)
        truth = jnp.array([.7, 1.3])
        params = single_star_params({'teff': 5800.}, vsini=6., vmacro=3., q1=.36, q2=.3,
                                    rv=jnp.array([12., 9.]), resolving_power=70000., depth_scale=truth)
        fit.observation = replace(obs, flux=np.asarray(fit.specmodel(params, obs.wavelength)))
        model = make_model_single(fit, resolving_power=70000., use_gp=False, degree=1,
            depth_scale=dist.LogNormal(jnp.zeros(2), .4), rv_bounds=(6., 15.),
            sigma_constant=.001, sigma_continuum=.001,
            fixed=dict(teff=5800., vsini=6., vmacro=3., q1=.36, q2=.3, jitter=.001))
        estimate = fit.run_svi(model=model, rng_key=jax.random.PRNGKey(8),
            p_initial={'depth_scale': jnp.ones(2), 'rv': jnp.array([11., 11.])},
            num_steps=150, step_size=.03, progress_bar=False)
        np.testing.assert_allclose(estimate['depth_scale'], truth, atol=.05)
        np.testing.assert_allclose(estimate['rv'], [12., 9.], atol=.3)
        mcmc = fit.run_nuts(model=model, rng_key=jax.random.PRNGKey(9),
            num_warmup=20, num_samples=15, num_chains=1, progress_bar=False,
            nuts_kwargs={'init_strategy': init_to_value(values=estimate), 'max_tree_depth': 4})
        output = save_fit_result(tmp_path/'region_fit', sf=fit, model=model, point=estimate, mcmc=mcmc)
        saved = load_fit_result(output)
        assert saved.samples['depth_scale'].shape == (15, 2)
        assert saved.idata.posterior['depth_scale'].dims == ('chain', 'draw', 'region')
        np.testing.assert_array_equal(saved.idata.posterior.coords['region'], ['a', 'b'])
        prediction = fit.predict(samples=saved.samples, model=model, max_draws=3)
        assert np.isfinite(prediction['model_flux']).all()
        np.testing.assert_array_equal(saved.observation.mask, obs.mask)


@pytest.mark.parametrize("per_region", [False, True])
def test_zero_depth_and_unit_dilution_are_the_same_flat_spectrum(per_region, x64_context):
    with x64_context():
        fit, settings, point = case()
        zero = jnp.zeros(2) if per_region else jnp.array(0.)
        one = jnp.ones_like(zero)
        depth_model = configured(fit, dict(settings, depth_scale=zero))
        dilution_model = configured(fit, dict(settings, dilution=one))
        params = fit.predict(point=point, model=depth_model)['params']
        for stage in ('intrinsic', 'broadened', 'full'):
            evaluate = getattr(fit.specmodel, stage)
            depth_flux = jax.jit(lambda a: evaluate(dict(params, depth_scale=a), fit.observation.wavelength))(zero)
            dilution_flux = jax.jit(lambda d: evaluate(dict(params, depth_scale=one, dilution=d), fit.observation.wavelength))(one)
            np.testing.assert_allclose(depth_flux, 1., atol=0.)
            np.testing.assert_allclose(dilution_flux, depth_flux, atol=1e-14)
        for model in (depth_model, dilution_model):
            value = jax.jit(lambda: log_density(model, (fit.fit_observation(), fit.specmodel), {}, point)[0])()
            assert np.isfinite(value)


def test_positive_depth_prior_still_excludes_zero():
    fit, settings, point = case()
    model = configured(fit, dict(settings, depth_scale=dist.LogNormal(0., .3)))
    with pytest.raises(ValueError, match='outside its prior support'):
        model.prior_trace(condition=dict(point, depth_scale=0.))


@pytest.mark.parametrize("name,value", [("depth_scale", -.4), ("dilution", -.2), ("dilution", 1.2)])
def test_scale_ranges_are_chosen_by_the_prior(name, value, x64_context):
    with x64_context():
        fit, settings, point = case()
        plain = configured(fit, settings)
        physical = fit.predict(point=point, model=plain)['physical_flux']
        model = configured(fit, dict(settings, **{name: dist.Normal(0., 2.)}))
        point = dict(point, **{name: value})
        prediction = fit.predict(point=point, model=model)
        expected = 1 + value * (physical - 1) if name == 'depth_scale' else value + (1-value)*physical
        np.testing.assert_allclose(prediction['physical_flux'], expected, atol=1e-13)
        density = jax.jit(lambda x: log_density(model, (fit.fit_observation(), fit.specmodel), {}, dict(point, **{name:x}))[0])
        assert np.isfinite(density(jnp.asarray(value)))


@pytest.mark.parametrize("name", ['depth_scale', 'dilution'])
@pytest.mark.parametrize("gp", [False, True])
def test_nonfinite_trial_scales_have_zero_likelihood_without_callback_failure(name, gp, x64_context):
    with x64_context():
        fit, settings, point = case()
        if gp:
            settings.update(use_gp=True, gp_solver='direct', gp_amplitude=.015, gp_scale=.3)
        model = configured(fit, dict(settings, **{name:dist.Normal(0., 1.)}))
        def likelihood(value):
            _, trace = log_density(model, (fit.fit_observation(), fit.specmodel), {}, dict(point, **{name:value}))
            return trace['spectrum']['fn'].log_factor
        values = jax.jit(jax.vmap(likelihood))(jnp.array([.5, 0., -1., jnp.inf, -jnp.inf, jnp.nan]))
        jax.effects_barrier()
        assert np.isfinite(values[:3]).all()
        assert np.isneginf(values[3:]).all()


def test_vectorized_nuts_rejects_extreme_log_depth_steps(x64_context):
    from numpyro.infer import init_to_value
    with x64_context():
        fit, settings, _ = case()
        settings.update(atmosphere={'teff':5800.}, depth_scale=dist.LogNormal(0., .3))
        model = configured(fit, settings)
        mcmc = fit.run_nuts(model=model, rng_key=jax.random.PRNGKey(44),
            num_warmup=0, num_samples=2, num_chains=2, progress_bar=False,
            nuts_kwargs=dict(init_strategy=init_to_value(values={'depth_scale':1.}),
                step_size=1e6, adapt_step_size=False, adapt_mass_matrix=False, max_tree_depth=1),
            mcmc_kwargs={'chain_method':'vectorized'})
        samples = mcmc.get_samples()['depth_scale']
        jax.block_until_ready(samples)
        jax.effects_barrier()
        assert np.isfinite(samples).all() and np.all(samples > 0)
        assert np.asarray(mcmc.get_extra_fields()['diverging']).any()
