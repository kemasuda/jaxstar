"""Track plotting: physical coordinates, joint draws, observations and ranges."""

import json

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxstar.mistfit import (MistFit, MistModel, plot_track, plot_posterior_tracks,
                             plot_diagnostics, get_run_metadata)
from test_mist_tracks import track_path, track_precision, PARAMS, reference


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def posterior(track_path):
    model = MistModel.from_tracks(track_path)
    rng = np.random.default_rng(53)
    mass = rng.uniform(.85, 1.35, (2, 40))
    draws = dict(mass=mass, feh_init=.6*(mass-1.1),
                 distance=.1+.2*(mass-.8), eep=rng.uniform(320., 480., (2, 40)))
    idata = az.from_dict(posterior=draws)
    truth = model(PARAMS, keys=('teff', 'jmag', 'kmag', 'feh'))
    observed = dict(keys=list(truth), values=[float(v) for v in truth.values()], errors=[80., .02, .03, .1])
    idata.attrs['jaxstar_run_metadata'] = json.dumps(dict(sampler='nuts', latent_names=list(draws),
        model=dict(physical_model={'kind': 'track'}, observations=observed)))
    # Fitter data deliberately differ: plots must use the recorded run's data.
    fit = MistFit(model=model, observations={'teff': (1000., 2.), 'kmag': (50., 3.)})
    return fit, idata


def test_single_track_eep_and_age_ranges_match_physical_model(track_path, tmp_path):
    model = MistModel.from_tracks(track_path)
    fig = plot_track(model, PARAMS['mass_init'], PARAMS['feh_init'], x='eep', y='age',
                     eep_range=(323.5, 472.8), n_points=13, save_path=tmp_path/'track.png')
    line = fig.axes[0].lines[0]
    eep, age = line.get_data()
    np.testing.assert_array_equal(eep[[0, -1]], [323.5, 472.8])
    assert {350., 400., 450.} <= set(eep)
    expected = [10**(reference(track_path, dict(PARAMS, eep=e))['logage']-9) for e in eep]
    np.testing.assert_allclose(age, expected, rtol=3e-13)
    assert (tmp_path/'track.png').stat().st_size > 1000
    fig = plot_track(model, PARAMS['mass_init'], PARAMS['feh_init'], x='age', y='eep',
                     age_range=(1.1, 3.7), n_points=7)
    ages, eeps = fig.axes[0].lines[0].get_data()
    np.testing.assert_allclose(ages[[0, -1]], [1.1, 3.7], rtol=3e-13)
    np.testing.assert_allclose(ages, [10**(reference(track_path, dict(PARAMS, eep=e))['logage']-9)
                                    for e in eeps], rtol=3e-13)
    # Intersect a broad age interval with physical coverage, including age=0.
    broad = plot_track(model, 1.1, 0., x='eep', y='age', age_range=(0., 100.))
    np.testing.assert_array_equal(broad.axes[0].lines[0].get_xdata()[[0, -1]], [300., 500.])


def test_single_track_apparent_absolute_and_repeated_axis_direction(track_path):
    model = MistModel.from_tracks(track_path)
    fig = plot_track(model, 1.1, 0., y='kmag_abs')
    ax = fig.axes[0]
    assert ax.xaxis_inverted() and ax.yaxis_inverted()
    fit = MistFit(model=model, observations={})
    same = fit.plot_track(1.1, 0., distance=.2, y='kmag', ax=ax)
    assert same is fig and ax.xaxis_inverted() and ax.yaxis_inverted()
    np.testing.assert_allclose(ax.lines[1].get_ydata()-ax.lines[0].get_ydata(), 5*np.log10(.2)+10)
    with pytest.raises(ValueError, match='distance'):
        plot_track(model, 1.1, 0., y='kmag')


@pytest.mark.parametrize('options', [dict(eep_range=(400., 300.)), dict(eep_range=(200., 400.)),
    dict(age_range=(-1., 3.)), dict(age_range=(100., 200.)),
    dict(eep_range=(320., 400.), age_range=(1., 3.)), dict(n_points=1), dict(y='unknown')])
def test_invalid_track_ranges_fail_before_creating_figure(track_path, options):
    with pytest.raises(ValueError):
        plot_track(MistModel.from_tracks(track_path), 1.1, 0., **options)
    assert not plt.get_fignums()


def test_posterior_tracks_preserve_joint_draws_and_show_recorded_errorbars(posterior):
    fit, idata = posterior
    fig = fit.plot_posterior_tracks(idata)
    assert len(fig.axes) == 2  # J and Ks, in recorded observation order.
    indices = np.random.default_rng(0).choice(80, 20, replace=False)
    p = {k: np.asarray(v).reshape(-1)[indices] for k, v in idata.posterior.items()}
    p['mass_init'] = p.pop('mass')
    predictions = fit.physical_model(p, keys=('teff', 'jmag', 'kmag'))
    record = get_run_metadata(idata)['model']['observations']
    observed = dict(zip(record['keys'], zip(record['values'], record['errors'])))
    for ax, key in zip(fig.axes, ('jmag', 'kmag')):
        assert ax.xaxis_inverted() and ax.yaxis_inverted()
        assert len(ax.lines[:20]) == 20
        marker = next(line for line in ax.lines if line.get_label() == 'Posterior locations')
        np.testing.assert_allclose(marker.get_xdata(), predictions['teff'], rtol=2e-13)
        np.testing.assert_allclose(marker.get_ydata(), predictions[key], rtol=2e-13)
        # Each curve uses the same joint coordinates as the corresponding dot.
        for i, line in enumerate(ax.lines[:20]):
            point = {k: v[i] for k, v in p.items()}
            expected = fit.physical_model(dict(point, eep=300.), keys=('teff', key))
            np.testing.assert_allclose([line.get_xdata()[0], line.get_ydata()[0]],
                                       [expected['teff'], expected[key]], rtol=2e-13)
        errorbar, = ax.containers
        xv, xe = observed['teff']
        yv, ye = observed[key]
        np.testing.assert_allclose(np.asarray(errorbar.lines[0].get_data(), dtype=float), [[xv], [yv]])
        np.testing.assert_allclose(errorbar.lines[2][0].get_segments()[0], [[xv-xe, yv], [xv+xe, yv]])
        np.testing.assert_allclose(errorbar.lines[2][1].get_segments()[0], [[xv, yv-ye], [xv, yv+ye]])


def test_posterior_selection_reproducible_without_changing_global_rng(posterior):
    fit, idata = posterior
    np.random.seed(18)
    expected = np.random.random()
    np.random.seed(18)
    fig = fit.plot_posterior_tracks(idata, x='eep', y=('mass', 'distance', 'feh_init'), seed=12)
    assert np.random.random() == expected
    again = fit.plot_posterior_tracks(idata, x='eep', y=('mass', 'distance', 'feh_init'), seed=12)
    for a, b in zip(fig.axes, again.axes):
        for la, lb in zip(a.lines, b.lines):
            np.testing.assert_array_equal(la.get_data(), lb.get_data())
    small = idata.isel(chain=[0], draw=slice(0, 5))
    fewer = fit.plot_posterior_tracks(small, x='eep', y='mass', n_samples=20)
    assert len(fewer.axes[0].lines) == 6  # five curves and one array of markers
    assert len(fewer.axes[0].lines[-1].get_xdata()) == 5


def test_posterior_range_and_exact_observation_key_matching(posterior):
    fit, idata = posterior
    limited = fit.plot_posterior_tracks(idata, x='age', y='mass', age_range=(.2, .3))
    assert 'selected tracks overlap' in limited._suptitle.get_text()
    for line in limited.axes[0].lines:
        assert np.all((line.get_xdata() >= .2-1e-12) & (line.get_xdata() <= .3+1e-12))
        assert line.get_label() != 'Posterior locations'  # sampled EEPs are older
    absolute = fit.plot_posterior_tracks(idata, y='kmag_abs')
    assert not absolute.axes[0].containers  # no apparent-K point on absolute-K axis
    assert absolute.axes[0].patches  # only the observed Teff band
    unobserved = fit.plot_posterior_tracks(idata, x='age', y='radius')
    assert not unobserved.axes[0].containers and not unobserved.axes[0].patches


@pytest.mark.parametrize('sampler', ['nuts', 'multinest'])
def test_diagnostics_add_tracks_save_and_keep_grid_free_option(posterior, tmp_path, monkeypatch, sampler):
    fit, idata = posterior
    record = get_run_metadata(idata)
    record['sampler'] = sampler
    idata.attrs['jaxstar_run_metadata'] = json.dumps(record)
    figs = fit.plot_diagnostics(idata, keys='mass', output_dir=tmp_path,
                                track_kwargs={'y': 'kmag', 'n_samples': 3})
    expected = {'corner', 'tracks'} | ({'trace'} if sampler == 'nuts' else set())
    assert set(figs) == expected
    for key in expected:
        assert (tmp_path/f'{key}.png').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        fit.plot_diagnostics(idata, keys='mass', output_dir=tmp_path)
    monkeypatch.setattr(MistModel, '__call__', lambda *a, **kw: pytest.fail('grid-free plots must not evaluate the model'))
    assert set(fit.plot_diagnostics(idata, keys='mass', show_tracks=False)) == expected-{'tracks'}
    assert set(plot_diagnostics(idata, keys='mass')) == expected-{'tracks'}


def test_diagnostics_no_observed_band_and_explicit_axes(posterior):
    fit, idata = posterior
    record = get_run_metadata(idata)
    record['model']['observations'] = dict(keys=['teff'], values=[5800.], errors=[80.])
    idata.attrs['jaxstar_run_metadata'] = json.dumps(record)
    assert set(fit.plot_diagnostics(idata, keys='mass')) == {'trace', 'corner'}
    with pytest.raises(ValueError, match='y keys'):
        fit.plot_posterior_tracks(idata)
    figs = fit.plot_diagnostics(idata, keys='mass', track_kwargs={'y': 'logg'})
    assert set(figs) == {'trace', 'corner', 'tracks'}
    with pytest.raises(ValueError, match='track model'):
        plot_diagnostics(idata, keys='mass', track_kwargs={'y': 'logg'})
