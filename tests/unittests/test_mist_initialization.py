"""Grid starts use the actual target and remain optional for both model kinds."""

import importlib
from itertools import product

import jax
import numpy as np
import pytest
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density

from jaxstar.mistfit import MistFit, MistModel, find_initial, get_run_metadata
from test_mistmodel import grid_path, OPTIONS as ISO_OPTIONS
from test_mist_tracks import track_path, track_precision, OPTIONS as TRACK_OPTIONS
from test_mist_masked_tracks import masked_path
from jaxstar.mistfit.inference import _feh_scan_axis


@pytest.fixture(params=['track', 'isochrone'])
def fit_options(request, track_path, grid_path):
    track = request.param == 'track'
    model = (MistModel.from_tracks(track_path) if track
             else MistModel.from_isochrones(grid_path))
    fit = MistFit(model=model, observations={'parallax': (8., .1), 'teff': (5400., 100.),
                                            'kmag': (9., .1)})
    return fit, dict(TRACK_OPTIONS if track else ISO_OPTIONS)


@pytest.mark.parametrize('linear_age', [True, False])
@pytest.mark.parametrize('nodata', [True, False])
def test_best_start_matches_scalar_joint_density(fit_options, linear_age, nodata):
    fit, options = fit_options
    options.update(linear_age=linear_age, nodata=nodata, prot=15.,
                   age_obs=2., age_err=.6, rho_obs=.8, rho_err=.2)
    # Include all native age nodes; do not mistake the iso soft mass prior for
    # a hard search bound. Track mass bounds, in contrast, are hard.
    track = fit.physical_model.kind == 'track'
    if not track:
        options.update(logamin=7.9, logamax=10.1, massmin=1.2, massmax=1.3,
                       flat_age_marginal=True)
    original = options.copy()
    actual = fit.find_initial(feh_init=-.1, model_kwargs=options)
    assert options == original
    candidates = []
    for x, eep in product(([1., 1.15] if track else [8., 9., 10.]), [350., 400., 450.]):
        point = dict(feh_init=-.1, distance=.125, eep=eep)
        point['mass' if track else ('age' if linear_age else 'logage')] = (
            x if track or not linear_age else 10.**x / 1e9)
        candidates.append(point)
    scores = np.array([float(log_density(fit.model, (), options, p)[0]) for p in candidates])
    scores[~np.isfinite(scores)] = -np.inf
    assert actual == candidates[np.argmax(scores)]
    # A finite constrained density alone is insufficient: verify the actual
    # NumPyro unconstrained potential and gradients at the chosen start.
    info = initialize_model(jax.random.PRNGKey(3), fit.model, model_kwargs=options,
                            init_strategy=init_to_value(values=actual))
    assert np.isfinite(info.param_info.potential_energy)
    assert all(np.isfinite(v) for v in info.param_info.z_grad.values())


def test_narrow_ranges_use_interior_midpoints(fit_options):
    fit, options = fit_options
    track = fit.physical_model.kind == 'track'
    options.update(eepmin=370., eepmax=380.)
    options.update(dict(massmin=1.07, massmax=1.09) if track
                   else dict(logamin=9.1, logamax=9.3, linear_age=False))
    initial = fit.find_initial(model_kwargs=options)
    assert initial['eep'] == 375.
    np.testing.assert_allclose(initial['mass' if track else 'logage'], 1.08 if track else 9.2)


@pytest.mark.parametrize('key', ['feh', 'feh_surface', 'feh_photosphere'])
def test_observed_feh_default_and_explicit_override(fit_options, key):
    fit, options = fit_options
    fit = MistFit(model=fit.physical_model, observations={key: (-.2, .1), 'parallax': (8., .1)})
    initial, search = fit.find_initial(model_kwargs=options, return_grid=True)
    assert initial['feh_init'] == -.2
    assert search.fixed_values == dict(feh_init=-.2, distance=.125)
    assert fit.find_initial(feh_init=0., model_kwargs=options)['feh_init'] == 0.
    outside = MistFit(model=fit.physical_model, observations={key: (-2., .1), 'parallax': (8., .1)})
    with pytest.raises(ValueError, match='feh_init must be finite'):
        outside.find_initial(model_kwargs=options)
    assert outside.find_initial(feh_init=0., model_kwargs=options)['feh_init'] == 0.


@pytest.mark.parametrize('linear_age', [True, False])
def test_return_grid_retains_exact_search_target(fit_options, linear_age):
    fit, options = fit_options
    options.update(linear_age=linear_age)
    initial, search = fit.find_initial(model_kwargs=options, return_grid=True)
    assert initial == fit.find_initial(model_kwargs=options)
    assert initial['feh_init'] == 0.  # no FeH observation
    for i, x in enumerate(search.axes[0]):
        for j, eep in enumerate(search.axes[1]):
            first = search.axis_names[0]
            if first == 'logage' and linear_age:
                first, x_value = 'age', 10.**x / 1e9
            else:
                x_value = x
            point = dict(search.fixed_values, eep=eep, **{first: x_value})
            expected = log_density(fit.model, (), options, point)[0]
            np.testing.assert_allclose(search.log_density[i, j], expected, rtol=1e-12)
    options['dist_scale'] = 10.
    assert search.model_kwargs.get('dist_scale') != 10.


@pytest.mark.parametrize('linear_age', [True, False])
def test_feh_scan_profiles_scalar_density_and_selects_global_start(fit_options, linear_age):
    fit, options = fit_options
    options.update(linear_age=linear_age)
    initial, search = fit.find_initial(model_kwargs=options, feh_init=-.13,
                                       feh_step=.1, return_grid=True)
    assert -.13 in search.feh_values  # retain the explicit/observed guess too
    assert np.all(search.feh_values > options['fmin'])
    assert np.all(search.feh_values < options['fmax'])
    assert np.max(np.diff(search.feh_values)) <= .1 + 1e-14
    assert search.fixed_values == dict(distance=.125)
    reference = np.empty(search.log_density_by_feh.shape)
    candidates = []
    for k, feh in enumerate(search.feh_values):
        for i, x in enumerate(search.axes[0]):
            for j, eep in enumerate(search.axes[1]):
                name = search.axis_names[0]
                if name == 'logage' and linear_age:
                    name, value = 'age', 10.**x / 1e9
                else:
                    value = x
                point = dict(feh_init=feh, distance=.125, eep=eep, **{name: value})
                reference[k, i, j] = log_density(fit.model, (), options, point)[0]
                candidates.append(point)
    np.testing.assert_allclose(search.log_density_by_feh, reference, rtol=1e-12)
    np.testing.assert_allclose(search.log_density, reference.max(axis=0), rtol=1e-12)
    np.testing.assert_allclose(search.best_feh, search.feh_values[reference.argmax(axis=0)])
    assert initial == candidates[np.argmax(reference)]
    # Native FeH spacing is much coarser; the scan really evaluates interpolated slices.
    native = fit.physical_model.grid.axis('initial_feh' if fit.physical_model.kind == 'track' else 'feh')
    assert not np.isin(search.feh_values, native).all()
    info = initialize_model(jax.random.PRNGKey(8), fit.model, model_kwargs=options,
                            init_strategy=init_to_value(values=initial))
    assert np.isfinite(info.param_info.potential_energy)
    assert all(np.isfinite(v) for v in info.param_info.z_grad.values())


def test_feh_scan_bounds_and_narrow_interval():
    axis = np.array([-1., 0., .5])
    nodes = _feh_scan_axis(axis, -.3, .3, .1)
    np.testing.assert_allclose(nodes, [-.2, -.1, 0., .1, .2])
    np.testing.assert_allclose(_feh_scan_axis(axis, .031, .033, .1), [.032])
    # Model edges are allowed when strictly inside the prior bounds.
    np.testing.assert_allclose(_feh_scan_axis(axis, -2., 1., .5), [-1., -.5, 0., .5])
    for step in [0., -.1, np.nan, np.inf, [], True]:
        with pytest.raises(ValueError, match='feh_step'):
            _feh_scan_axis(axis, -.3, .3, step)


def test_feh_scan_can_skip_unsupported_automatic_observation(fit_options):
    fit, options = fit_options
    fit = MistFit(model=fit.physical_model, observations={'feh': (-2., .1), 'parallax': (8., .1)})
    initial = fit.find_initial(model_kwargs=options, feh_step=.1)
    assert options['fmin'] < initial['feh_init'] < options['fmax']
    with pytest.raises(ValueError, match='feh_init'):
        fit.find_initial(model_kwargs=options, feh_step=.1, feh_init=-2.)


@pytest.mark.parametrize('parallax', [None, 0., -1.])
def test_distance_override_and_invalid_fixed_values(fit_options, parallax):
    fit, options = fit_options
    fit = MistFit(model=fit.physical_model, observations=(
        {} if parallax is None else {'parallax': (parallax, .1)}))
    with pytest.raises(ValueError, match='positive parallax'):
        fit.find_initial(model_kwargs=options)
    actual = find_initial(fit.observations, fit.physical_model, distance=.2,
                          model_kwargs=options)
    assert actual['distance'] == .2
    for distance in [0., -1., np.nan, np.inf]:
        with pytest.raises(ValueError, match='distance'):
            fit.find_initial(distance=distance, model_kwargs=options)
    for feh in [options['fmin'], options['fmax'], np.nan, 2.]:
        with pytest.raises(ValueError, match='feh_init'):
            fit.find_initial(distance=.2, feh_init=feh, model_kwargs=options)


def test_masked_photometry_rejects_best_unavailable_node(masked_path):
    model = MistModel.from_tracks(masked_path)
    physical = model(dict(mass_init=1., feh_init=0., eep=400., distance=.15), keys='teff')
    fit = MistFit(model=model, observations={'teff': (float(physical['teff']), 1e-3),
                                            'kmag': (8., .1)})
    initial = fit.find_initial(distance=.15, model_kwargs=TRACK_OPTIONS)
    assert (initial['mass'], initial['eep']) != (1., 400.)
    assert np.isfinite(log_density(fit.model, (), TRACK_OPTIONS, initial)[0])
    # Restrict the search to the photometric hole; fail clearly instead of
    # returning an arbitrary all-inf argmax.
    options = dict(TRACK_OPTIONS, massmin=.99, massmax=1.01, eepmin=390., eepmax=410.)
    with pytest.raises(ValueError, match='no candidate with finite'):
        fit.find_initial(distance=.15, model_kwargs=options)
    prior = fit.find_initial(distance=.15, model_kwargs=dict(options, nodata=True))
    assert prior['eep'] == 400. and prior['mass'] == 1.


def test_out_of_grid_iso_bounds_and_unknown_options(grid_path):
    fit = MistFit(model=MistModel.from_isochrones(grid_path), observations={'parallax': (8., .1)})
    with pytest.raises(ValueError, match='no interior overlap'):
        fit.find_initial(model_kwargs=dict(logamin=11., logamax=12.))
    with pytest.raises(TypeError, match='unexpected keyword'):
        fit.find_initial(model_kwargs=dict(unknown=True))


def test_wrapper_grid_search_and_explicit_precedence(fit_options, monkeypatch):
    fit, options = fit_options
    module = importlib.import_module('jaxstar.mistfit.mistfit')
    calls, searches = [], []
    initial = dict(mass=1., feh_init=0., eep=400., distance=.2)
    def search(self, **kwargs):
        searches.append(kwargs)
        return initial
    def run(*args, **kwargs):
        calls.append(kwargs)
        return 'result'
    monkeypatch.setattr(MistFit, 'find_initial', search)
    monkeypatch.setattr(module, 'run_nuts', run)
    init_options = dict(feh_init=-.1, distance=.2)
    assert fit.run_nuts(jax.random.PRNGKey(0), model_kwargs=options,
                        init_kwargs=init_options, output_keys='rho', num_chains=3) == 'result'
    assert len(searches) == 1
    assert searches[0] == dict(init_options, model_kwargs=dict(options, output_keys=('rho',)))
    assert calls[-1]['init_strategy'].keywords['values'] == initial
    assert calls[-1]['num_chains'] == 3
    assert 'output_keys' not in options
    fit.run_nuts(jax.random.PRNGKey(1), grid_init=False)
    assert len(searches) == 1
    assert calls[-1]['init_strategy'].keywords['values']['distance'] == .125

    # Explicit initializations must bypass *all* automatic initialization,
    # including its observation requirements and unused init_kwargs.
    fit = MistFit(model=fit.physical_model, observations={})
    strategy = init_to_value(values=initial)
    fit.run_nuts(jax.random.PRNGKey(2), init_strategy=strategy,
                 init_kwargs={'deliberately_unused': True})
    assert calls[-1]['init_strategy'] is strategy
    unconstrained = {'eep': np.array(0.)}
    fit.run_nuts(jax.random.PRNGKey(2), init_params=unconstrained)
    assert calls[-1]['init_params'] is unconstrained and calls[-1]['init_strategy'] is None
    assert len(searches) == 1


def test_automatic_start_short_nuts_and_metadata(fit_options):
    fit, options = fit_options
    result = fit.run_nuts(jax.random.PRNGKey(4), model_kwargs=options,
                          num_warmup=8, num_samples=4, num_chains=1,
                          max_tree_depth=3, progress_bar=False)
    assert np.isfinite(result.get_extra_fields()['potential_energy']).all()
    for key in ('eep', 'distance', 'feh_init', 'age'):
        assert np.isfinite(result.get_samples()[key]).all()
    values = get_run_metadata(result)['initialization']['strategy']['kwargs']['values']
    assert values['feh_init'] == 0. and values['distance'] == .125
    # Default NUTS initialization remains fixed-FeH, while the later diagnostic
    # scans independently and recovers these exact sampling options.
    coverage = fit.check_missed_modes(result)
    assert coverage['grid_search'].feh_step == .1
    assert len(coverage['grid_search'].feh_values) > 1
    np.testing.assert_allclose(coverage['grid_search'].model_kwargs['dist_scale'],
                               get_run_metadata(result)['model']['arguments']['dist_scale'])
    assert coverage['counts_by_chain'].sum() == 4
