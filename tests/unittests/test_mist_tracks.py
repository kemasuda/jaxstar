"""Offline track integration: independent interpolation/density and result I/O."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density
from jax.flatten_util import ravel_pytree
import pytest
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit
from scipy.stats import gamma, norm

from jaxstar.mistfit import (MistFit, MistModel, MistObservations, model_track,
                             get_run_metadata, load_result)
from jaxstar.mistfit import run_multinest
from jaxstar.mistfit.mistgrid.tracks import centered_age_derivative


BANDS = ("Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I",
         "2MASS_J", "2MASS_H", "2MASS_Ks", "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev",
         "Gaia_RP_DR2Rev", "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
MAG_KEYS = ("umag", "bmag", "vmag", "rmag", "imag", "jmag", "hmag", "kmag",
            "gmag2", "bpmag2", "rpmag2", "gmag3", "bpmag3", "rpmag3")
PARAMS = dict(mass_init=1.13, feh_init=-.12, eep=377., distance=.15)
LATENTS = dict(mass=PARAMS["mass_init"], **{k: v for k, v in PARAMS.items() if k != "mass_init"})
OPTIONS = dict(massmin=.85, massmax=1.35,
               fmin=-.35, fmax=.3, eepmin=305., eepmax=495., dist_scale=1.7,
               age_max=None)


@pytest.fixture(autouse=True)
def track_precision(x64_context):
    with x64_context():
        yield


@pytest.fixture
def track_path(tmp_path):
    masses, fehs, eeps = np.array([.8, 1., 1.15, 1.4]), np.array([-.4, 0., .4]), np.arange(300., 501., 50.)
    m, f, e = np.meshgrid(masses, fehs, eeps, indexing="ij")
    x = (e - 300.) / 200.
    age = (.4 + 3 * x + 2 * x**2 + x**3) * m**-2 * 10**(.1 * f)
    fields = dict(logage=9 + np.log10(age), age_gyr=age,
                  logTeff=3.7 + .12*m - .1*x - .03*f,
                  logg=4.5 - .9*x + .05*f, logL=.4*m + .7*x,
                  logR=.2*m + .3*x, star_mass=m*(1-.05*x),
                  cmd_surface_feh=f-.05*x, bc_metallicity=f,
                  extra_native=m + f + x)
    fields["absolute_magnitudes"] = np.stack(
        [4 + i*.1 - 2*x + .2*f - .5*m for i in range(14)], axis=-1)
    fields["absolute_magnitudes"][..., 9] += 1.  # DR2 BP-RP for optional gyro.
    path = tmp_path / 'tracks.npz'
    np.savez(path, initial_mass=masses, initial_feh=fehs, eep=eeps,
             bands=np.array(BANDS), **fields)
    return path


def native_fields(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def reference(path, p=PARAMS):
    native = native_fields(path)
    axes = tuple(native[k] for k in ('initial_mass', 'initial_feh', 'eep'))
    point = [p[k] for k in ('mass_init', 'feh_init', 'eep')]
    raw = {k: RegularGridInterpolator(axes, v)(point)[0] for k, v in native.items()
           if k not in ('initial_mass', 'initial_feh', 'eep', 'bands')}
    # Independent NumPy derivative, with the documented endpoint copies.
    derivative = np.gradient(native['age_gyr'], axes[2], axis=-1, edge_order=2)
    derivative[..., 0], derivative[..., -1] = derivative[..., 1], derivative[..., -2]
    raw['dtdeep'] = RegularGridInterpolator(axes, derivative)(point)[0]
    return raw


def test_track_predictions_units_fields_and_log_interpolation(track_path):
    model = MistModel.from_tracks(track_path)
    result = jax.jit(lambda m, p: m(p))(model, PARAMS)
    expected = reference(track_path)
    assert model.kind == 'track' and model.input_keys == ('mass_init', 'feh_init', 'eep')
    assert model.data.sources == (str(track_path.resolve()),)
    assert not {'dmdeep', 'mmin', 'mmax'} & set(result)
    for key, value in expected.items():
        if key != 'absolute_magnitudes':
            np.testing.assert_allclose(result[key], value, rtol=3e-13)
    for k, raw in (('teff', 'logTeff'), ('radius', 'logR')):
        np.testing.assert_allclose(result[k], 10**expected[raw], rtol=3e-13)
    np.testing.assert_allclose(result['age'], 10**(expected['logage']-9), rtol=3e-13)
    assert abs(result['age'] - expected['age_gyr']) > .001
    for k in ('mass', 'mass_init'):
        assert result[k] == PARAMS['mass_init']
    for k in ('feh', 'feh_surface', 'feh_photosphere'):
        assert result[k] == result['cmd_surface_feh']
    assert result['mass_current'] == result['star_mass']
    np.testing.assert_allclose(result['rho'], result['star_mass']/result['radius']**3)
    np.testing.assert_allclose(result['rho_initial'], PARAMS['mass_init']/result['radius']**3)
    for i, band in enumerate(MAG_KEYS):
        np.testing.assert_allclose(result[band+'_abs'], expected['absolute_magnitudes'][i], rtol=3e-13)
        np.testing.assert_allclose(result[band], result[band+'_abs'] + 5*np.log10(PARAMS['distance'])+10)
    assert result['parallax'] == 1/PARAMS['distance']
    with pytest.raises(FrozenInstanceError):
        model.data.kind = 'isochrone'


def test_track_keys_broadcast_pytree_and_gradients(track_path, monkeypatch):
    model = MistModel.from_tracks(track_path)
    leaves, treedef = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('evaluation must not read files'))
    keys = ('age', 'teff', 'radius', 'kmag', 'dtdeep', 'rho')
    def evaluate(m, point):
        result = m(dict(zip(PARAMS, point)), keys=keys)
        return jnp.stack([result[k] for k in keys], axis=-1)
    point = jnp.array(list(PARAMS.values()))
    np.testing.assert_array_equal(evaluate(model, point), evaluate(restored, point))
    gradient = jax.jit(jax.jacrev(evaluate, argnums=1))(restored, point)
    eps = np.array([1e-5, 1e-5, 1e-3, 1e-6])
    finite = np.column_stack([(evaluate(model, point+step)-evaluate(model, point-step))/(2*h)
                             for step, h in zip(np.diag(eps), eps)])
    np.testing.assert_allclose(gradient, finite, rtol=2e-7, atol=2e-7)
    # The prior derivative is deliberately different from autodiff of age.
    assert abs(gradient[0, 2] - model(PARAMS, keys='dtdeep')['dtdeep']) > 1e-4
    points = jnp.stack((point, point+jnp.array([.02, .01, 4., .01])))
    np.testing.assert_allclose(jax.vmap(evaluate, in_axes=(None, 0))(model, points), evaluate(model, points.T))
    batch = model({**PARAMS, 'feh_init': jnp.array([[-.2], [.1]]),
                   'distance': jnp.array([[.1, .2, .3]])}, keys=keys)
    assert all(v.shape == (2, 3) for v in batch.values())
    intrinsic = {k: v for k, v in PARAMS.items() if k != 'distance'}
    model(intrinsic, keys=('kmag_abs', 'age', 'mass_init'))
    with pytest.raises(ValueError, match='distance'):
        model(intrinsic, keys='kmag')
    with pytest.raises(ValueError, match='mass_init'):
        model({'logage': 9.3, 'feh_init': 0., 'eep': 350.}, keys='age')
    with pytest.raises(ValueError, match='isochrone'):
        model.eep_given_mass(9.3, 0., 1.)


def test_track_nodes_and_boundaries(track_path):
    model = MistModel.from_tracks(track_path)
    for m, f, e in ((.8, -.4, 300.), (1.4, .4, 500.)):
        p = dict(PARAMS, mass_init=m, feh_init=f, eep=e)
        expected = reference(track_path, p)
        for k in ('logage', 'dtdeep', 'extra_native'):
            np.testing.assert_allclose(model(p, keys=k)[k], expected[k], rtol=3e-13)
    for change in ({'mass_init': .79}, {'feh_init': .41}, {'eep': 501.}, {'mass_init': np.nan}):
        result = model(dict(PARAMS, **change), keys=('age', 'teff', 'dtdeep', 'kmag'))
        assert all(np.isneginf(v) for v in result.values())


@pytest.mark.parametrize('linear_age', [True, False])
@pytest.mark.parametrize('nodata', [True, False])
def test_track_joint_density_and_support_normalization(track_path, linear_age, nodata):
    model = MistModel.from_tracks(track_path)
    observations = MistObservations.from_mapping({'teff': (6100., 120.), 'parallax': (6.7, .1)})
    fit = MistFit(model=model, observations=observations)
    options = dict(OPTIONS, linear_age=linear_age, nodata=nodata, output_keys=('kmag_abs', 'rho', 'eep'))
    actual, trace = log_density(fit.model, (), options, LATENTS)
    raw = reference(track_path)
    age = 10**(raw['logage']-9)
    expected = -np.log(.5) - np.log(.65) - np.log(190.) + gamma.logpdf(PARAMS['distance'], a=3, scale=1.7)
    expected += np.log(raw['dtdeep'])
    if not linear_age:
        expected -= np.log(age*np.log(10))
    if not nodata:
        expected += np.sum(norm.logpdf(observations.values,
                                      loc=[10**raw['logTeff'], 1/PARAMS['distance']], scale=observations.errors))
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    standalone = log_density(model_track, (observations, model), options, LATENTS)[0]
    np.testing.assert_array_equal(actual, standalone)
    assert trace['mass']['type'] == 'sample' and trace['age']['type'] == 'deterministic'
    assert 'dmdeep' not in trace and 'kmag_abs' in trace
    assert 'logage_bound' not in trace
    assert trace['eep_lower']['value'] == 305. and trace['eep_upper']['value'] == 495.
    # Changing a fixed EEP width contributes only the Uniform density constant.
    smaller = log_density(fit.model, (), dict(options, eepmin=350., eepmax=425.), LATENTS)[0]
    np.testing.assert_allclose(smaller-actual, np.log(190./75.), atol=2e-12)
    # Dynamic physical-model leaves work inside compiled density evaluation.
    compiled = jax.jit(lambda m, p: log_density(model_track, (observations, m), options, p)[0])
    np.testing.assert_allclose(compiled(model, LATENTS), actual, atol=2e-12)
    for bounds in ({'eepmin': 400.}, {'massmax': 1.1}):
        assert np.isneginf(log_density(fit.model, (), dict(options, **bounds), LATENTS)[0])


@pytest.mark.parametrize('bounds', [{}, {'age_min': 1.4}, {'age_max': 2.1},
                                   {'age_min': 1.4, 'age_max': 2.1}])
def test_track_potential_gradients_and_optional_constraints(track_path, bounds):
    fit = MistFit(model=MistModel.from_tracks(track_path), observations={})
    info = initialize_model(jax.random.PRNGKey(3), fit.model,
        init_strategy=init_to_value(values=LATENTS), model_kwargs=dict(OPTIONS, nodata=True, **bounds))
    flat, unravel = ravel_pytree(info.param_info.z)
    potential = jax.jit(lambda z: info.potential_fn(unravel(z)))
    actual = jax.grad(potential)(flat)
    h = 1e-5
    expected = np.array([(potential(flat+d)-potential(flat-d))/(2*h) for d in np.eye(len(flat))*h])
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)
    baseline, _ = log_density(fit.model, (), dict(OPTIONS, nodata=True), LATENTS)
    extra, trace = log_density(fit.model, (), dict(OPTIONS, nodata=True,
        rho_obs=.4, rho_err=.05, age_obs=2., age_err=.2, prot=15.), LATENTS)
    prediction = fit.physical_model(PARAMS, keys=('rho_initial', 'age'))
    expected_extra = (-.5*((.4-prediction['rho_initial'])/.05)**2
                      -.5*((2.-prediction['age'])/.2)**2
                      + trace['loglike_gyro']['fn'].log_prob(trace['loglike_gyro']['value']))
    np.testing.assert_allclose(extra-baseline, expected_extra)


@pytest.mark.parametrize('linear_age', [True, False])
@pytest.mark.parametrize('bounds', [{'age_min': 2.}, {'age_max': 1.8},
                                   {'age_min': 1.8, 'age_max': 2.}])
def test_track_soft_age_bounds_leave_eep_support_fixed(track_path, linear_age, bounds):
    fit = MistFit(model=MistModel.from_tracks(track_path), observations={})
    options = dict(OPTIONS, linear_age=linear_age, nodata=True)
    for mass, feh in ((.9, -.2), (1.13, -.12), (1.3, .2)):
        point = dict(LATENTS, mass=mass, feh_init=feh)
        base, base_trace = log_density(fit.model, (), options, point)
        # Exercise the default width/depth and an explicitly widened boundary.
        for width, depth in ((.1, 30.), (.4, 12.)):
            limited, trace = log_density(fit.model, (), dict(options, **bounds,
                age_bound_width=width, age_bound_depth=depth), point)
            age = float(base_trace['age']['value'])
            expected = sum(-depth * expit((bound-age)/width if key == 'age_min' else (age-bound)/width)
                           for key, bound in bounds.items())
            np.testing.assert_allclose(limited-base, expected, atol=2e-12)
            assert np.isfinite(limited)  # Outside a soft bound still has support.
            assert trace['eep']['fn'].low == OPTIONS['eepmin']
            assert trace['eep']['fn'].high == OPTIONS['eepmax']


@pytest.mark.parametrize('target_age', [.02, 13.8, 1000.])
def test_track_default_age_bound_opt_out_and_stable_soft_tails(track_path, tmp_path, target_age):
    age_scale = target_age / 10**(reference(track_path)['logage']-9)
    fields = native_fields(track_path)
    fields['age_gyr'] *= age_scale
    fields['logage'] += np.log10(age_scale)
    path = tmp_path/'rescaled.npz'
    np.savez(path, **fields)
    fit = MistFit(model=MistModel.from_tracks(path), observations={})
    baseline, trace = log_density(fit.model, (), {'age_max': None}, LATENTS)
    age = float(trace['age']['value'])
    np.testing.assert_allclose(age, target_age, rtol=2e-13)
    assert np.isfinite(baseline) and 'logage_bound' not in trace
    default, default_trace = log_density(fit.model, (), {}, LATENTS)
    explicit, _ = log_density(fit.model, (), {'age_max': 13.8}, LATENTS)
    np.testing.assert_array_equal(default, explicit)
    assert np.isfinite(default) and 'logage_bound' in default_trace
    np.testing.assert_allclose(default-baseline, -30.*expit((age-13.8)/.1), atol=2e-12)
    assert trace['eep_lower']['value'] == 300. and trace['eep_upper']['value'] == 500.
    # At the bound, a single sigmoid contributes exactly half of its depth.
    for name in ('age_min', 'age_max'):
        value, _ = log_density(fit.model, (), {'age_max': None, name: age}, LATENTS)
        np.testing.assert_allclose(value-baseline, -15., atol=2e-12)
    # Test both far-tail directions; exp-based derivatives can overflow here.
    for bound in ({'age_min': 10000.}, {'age_max': .001}):
        density = jax.jit(lambda p: log_density(fit.model, (), {'age_max': None, **bound}, p)[0])
        value, grad = jax.value_and_grad(density)(LATENTS)
        assert np.isfinite(value)
        assert all(np.isfinite(v) for v in grad.values())


@pytest.mark.parametrize('option', [{'massmin': .7}, {'massmax': 1.5}, {'fmax': .5},
    {'eepmin': 299.}, {'eepmax': 600.}, {'massmin': 1.3, 'massmax': 1.2},
    {'dist_scale': 0.}, {'age_min': 2., 'age_max': 1.}, {'age_min': -1.},
    {'age_max': 0.}, {'age_min': np.nan}, {'age_max': np.inf},
    {'age_bound_width': 0.}, {'age_bound_width': np.inf}, {'age_bound_width': None},
    {'age_bound_depth': -1.}, {'age_bound_depth': np.nan}, {'age_max': [13.8]}])
def test_invalid_track_inference_options(track_path, option):
    fit = MistFit(model=MistModel.from_tracks(track_path), observations={})
    with pytest.raises(ValueError):
        log_density(fit.model, (), dict(OPTIONS, **option), LATENTS)


def test_invalid_dynamic_options_are_rejected_under_jit(track_path):
    model = MistModel.from_tracks(track_path)
    observations = MistObservations((), (), ())
    density = jax.jit(lambda m, lower: log_density(model_track, (observations, m),
                      dict(OPTIONS, massmin=lower), LATENTS)[0])
    assert np.isfinite(density(model, .85))
    assert np.isneginf(density(model, .7))
    bounded = jax.jit(lambda upper, width: log_density(model_track, (observations, model),
                     dict(OPTIONS, age_min=1., age_max=upper, age_bound_width=width), LATENTS)[0])
    assert np.isfinite(bounded(13.8, .1))
    for upper, width in ((.5, .1), (np.inf, .1), (13.8, 0.)):
        assert np.isneginf(bounded(upper, width))


@pytest.mark.parametrize('mutation', ['age', 'gap', 'nonuniform', 'band', 'shape', 'missing'])
def test_loader_rejects_incomplete_or_inconsistent_track_grid(track_path, tmp_path, mutation):
    fields = native_fields(track_path)
    if mutation == 'age':
        fields['age_gyr'][0, 0, 1] = fields['age_gyr'][0, 0, 0]
        fields['logage'] = np.log10(fields['age_gyr'])+9
    elif mutation == 'gap':
        fields['logg'][0, 0, 1] = np.nan
    elif mutation == 'nonuniform':
        fields['eep'][1] += 1
    elif mutation == 'band':
        fields['bands'][1] = fields['bands'][0]
    elif mutation == 'shape':
        fields['logTeff'] = fields['logTeff'][:-1]
    else:
        del fields['logR']
    path = tmp_path/'invalid.npz'
    np.savez(path, **fields)
    with pytest.raises(ValueError):
        MistModel.from_tracks(path)


def test_loader_x64_band_order_and_centered_error(track_path, tmp_path, x64_context):
    with x64_context(False), pytest.raises(ValueError, match='x64'):
        MistModel.from_tracks(track_path)
    original = MistModel.from_tracks(track_path)(PARAMS)
    fields = native_fields(track_path)
    fields['bands'] = fields['bands'][::-1]
    fields['absolute_magnitudes'] = fields['absolute_magnitudes'][..., ::-1]
    path = tmp_path/'permuted.npz'
    np.savez(path, **fields)
    permuted = MistModel.from_tracks(path)(PARAMS)
    for key in MAG_KEYS:
        np.testing.assert_array_equal(original[key], permuted[key])
    errors = []
    for h in (.1, .05):
        e = np.arange(1., 3.+h/2, h)
        d = centered_age_derivative(e**3, e)
        errors.append(np.max(np.abs(d[1:-1]-3*e[1:-1]**2)))
        assert d[0] == d[1] and d[-1] == d[-2]
    np.testing.assert_allclose(errors[0]/errors[1], 4., rtol=1e-10)


def test_track_nuts_save_reload_and_metadata(track_path, tmp_path):
    fit = MistFit(model=MistModel.from_tracks(track_path), observations={'parallax': (6.7, .1)})
    options = {key: value for key, value in OPTIONS.items() if key != 'age_max'}
    result = fit.run_nuts(jax.random.PRNGKey(8), model_kwargs=options,
        init_strategy=init_to_value(values=LATENTS), output_keys=('rho', 'kmag_abs'),
        num_chains=1, num_warmup=12, num_samples=16, max_tree_depth=3, progress_bar=False)
    record = get_run_metadata(result)
    assert record['model']['callable']['callable'].endswith('.model_track')
    assert record['model']['physical_model']['kind'] == 'track'
    assert set(record['latent_names']) == set(LATENTS)
    assert record['model']['arguments']['massmin'] == OPTIONS['massmin']
    assert record['model']['arguments']['age_min'] is None
    assert record['model']['arguments']['age_max'] == 13.8
    assert record['model']['arguments']['age_bound_width'] == .1
    samples = result.get_samples()
    assert all(np.all(np.isfinite(v)) for v in samples.values())
    assert {'age', 'logage', 'mass_init', 'rho', 'kmag_abs', 'dtdeep'} <= set(samples)
    idata = fit.to_inference_data(result)
    assert idata.posterior.dtdeep.attrs['units'] == 'Gyr / EEP'
    output = fit.save_result(idata, tmp_path/'result')
    track_path.unlink()
    restored = load_result(output)
    for key, value in samples.items():
        np.testing.assert_array_equal(restored.posterior[key].values[0], value)
    import matplotlib.pyplot as plt
    figure = fit.plot_trace(restored, keys=('age', 'mass'))
    plt.close(figure)


@pytest.mark.parametrize('linear_age', [True, False])
def test_track_multinest_and_nuts_use_same_fixed_prior(track_path, tmp_path, monkeypatch, linear_age):
    mn = pytest.importorskip('numpyro_inferutils.experimental.multinest')
    def solve(**kwargs):
        theta = kwargs['Prior'](np.full(kwargs['n_dims'], .5))
        assert np.isfinite(kwargs['LogLikelihood'](theta))
        return {'samples': np.stack([theta, theta]), 'logZ': -10., 'logZerr': .1}
    monkeypatch.setattr(mn, '_import_pymultinest', lambda: (solve, None))
    fit = MistFit(model=MistModel.from_tracks(track_path), observations={'parallax': (6.7, .1)})
    options = dict(OPTIONS, linear_age=linear_age, age_max=2.1)
    runner = fit.run_multinest(outputfiles_basename=str(tmp_path/'native-'),
        model_kwargs=options, n_live_points=12, resume=False)
    assert get_run_metadata(runner)['model']['arguments']['age_max'] == 2.1
    for mass in (.9, 1.13, 1.3):
        point = dict(LATENTS, mass=mass)
        theta = np.array([point[k] for k in runner.site_names])
        joint, trace = log_density(fit.model, (), options, point)
        np.testing.assert_allclose(runner.logposterior(theta), joint, rtol=1e-12)
        observed = sum(site['fn'].log_prob(site['value']).sum() for site in trace.values()
                       if site['type'] == 'sample' and site['is_observed'])
        np.testing.assert_allclose(runner.loglikelihood(theta), observed, rtol=1e-12)
    # Both samplers use the same fixed EEP support, also with a soft age bound.
    for mass_q, feh_q in ((.1, .2), (.9, .8)):
        cube = np.full(4, .35)
        cube[runner.site_names.index('mass')] = mass_q
        cube[runner.site_names.index('feh_init')] = feh_q
        theta = runner.prior(cube)
        np.testing.assert_allclose(theta[runner.site_names.index('eep')], 305.+.35*(495.-305.))
    samples = {k: jnp.array([v, v]) for k, v in LATENTS.items()}
    derived = runner.get_deterministic_samples(samples, site_names=['age', 'mass_init', 'dtdeep'])
    predicted = fit.physical_model(PARAMS, keys=tuple(derived))
    for k in derived:
        np.testing.assert_allclose(derived[k], predicted[k])
    direct = run_multinest(model_track, model_args=(fit.observations, fit.physical_model),
        model_kwargs=options, outputfiles_basename=str(tmp_path/'direct-'), n_live_points=12, resume=False)
    assert direct.model_kwargs == options
