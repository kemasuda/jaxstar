"""Connected-region coverage, threshold sensitivity and MIST density agreement."""

import numpy as np
import pytest
from numpyro.infer.util import log_density

from jaxstar.mistfit import (
    InitialGridSearch, check_missed_modes, recheck_missed_modes, plot_missed_modes,
)
from jaxstar.mistfit.diagnostics import _cell_edges, _region_coverage
from test_mist_initialization import fit_options
from test_mistmodel import grid_path
from test_mist_tracks import track_path, track_precision


def two_peaks():
    x, y = np.linspace(-5., 5., 41), np.linspace(200., 210., 21)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    scores = np.logaddexp(-.5 * ((xx + 3) / .5)**2,
                         -.5 * ((xx - 3) / .5)**2) - .5 * ((yy - 205) / .7)**2
    counts = np.zeros((2, len(x), len(y)), dtype=int)
    counts[:, 8, 10] = 20  # draws in only the left peak
    return (x, y), scores, counts


def test_separated_unvisited_peak_flags_and_both_peaks_clear():
    axes, scores, counts = two_peaks()
    # Preserve the original total-count-only diagnostic when explicitly selected.
    reports = [_region_coverage(scores, 0., counts, threshold, 5, 1, None)
               for threshold in (3., 5., 10.)]
    for report in reports:
        assert report['flag'] and len(report['regions']) == 2
        left, right = report['regions']
        assert left['samples'] == 40 and not left['flag']
        assert right['samples'] == 0 and right['flag']
        np.testing.assert_array_equal(left['samples_by_chain'], [20, 20])
        assert right['peak_index'] == (32, 10)
    counts[:, 32, 10] = 10
    assert not any(_region_coverage(scores, 0., counts, t, 5, 1, None)['flag'] for t in (3., 5., 10.))


def test_unvisited_fraction_detects_a_peak_hidden_by_a_high_threshold_bridge():
    # Like the track example: threshold 20 joins two peaks; 5 separates them.
    scores = np.full((5, 3), -100.)
    scores[:, 1] = [0., -10., -10., -10., -2.]
    counts = np.zeros((2, 5, 3), dtype=int)
    counts[:, 0, 1] = 20
    count_only = _region_coverage(scores, 0., counts, 20., 5, 1, None)
    assert len(count_only['regions']) == 1 and not count_only['flag']
    joined = _region_coverage(scores, 0., counts, 20., 5, 1)
    assert len(joined['regions']) == 1 and joined['flag']
    region = joined['regions'][0]
    assert region['nearby_samples'] == 40
    assert region['n_cells'] == 5 and region['n_unvisited_cells'] == 3
    assert region['unvisited_fraction'] == .6
    assert region['flag_reasons'] == ('high_unvisited_fraction',)
    assert region['peak_index'] == (0, 1)  # the higher peak was visited
    assert region['flag_peak_index'] == (4, 1)  # point to the missed peak
    assert region['flag_peak_delta_logp'] == 2.
    np.testing.assert_array_equal(joined['labels'], count_only['labels'])
    split = _region_coverage(scores, 0., counts, 5., 5, 1)
    assert len(split['regions']) == 2 and split['flag']
    right = split['regions'][1]
    assert right['peak_index'] == (4, 1)
    assert right['nearby_samples'] == 0 and right['flag']
    assert right['unvisited_fraction'] == 1.
    assert right['flag_reasons'] == ('low_sample_count', 'high_unvisited_fraction')


def test_fraction_cutoff_is_strict_and_ignores_cells_outside_candidates():
    scores = np.full((2, 12), -100.)
    scores[0, 1:11] = 0.  # ten candidate cells
    counts = np.zeros((1, 2, 12), dtype=int)
    counts[0, 0, 1:10] = 2  # nine visited cells; one empty
    report = _region_coverage(scores, 0., counts, 5., 5, 0)
    region = report['regions'][0]
    assert region['unvisited_fraction'] == .1 and not region['flag']
    assert region['flag_reasons'] == () and region['flag_peak_index'] is None
    counts[0, 0, 9] = 0  # two empty cells: strictly more than ten percent
    report = _region_coverage(scores, 0., counts, 5., 5, 0)
    assert report['flag'] and report['regions'][0]['unvisited_fraction'] == .2
    assert not _region_coverage(scores, 0., counts, 5., 5, 0, .25)['flag']
    assert not _region_coverage(scores, 0., counts, 5., 5, 0, 1.)['flag']
    assert _region_coverage(scores, 0., counts, 5., 5, 0, 0.)['flag']
    # Duplicating draws in visited cells does not cover missing cells.
    repeated = _region_coverage(scores, 0., counts * 1000, 5., 5, 0)
    assert repeated['flag'] and repeated['regions'][0]['unvisited_fraction'] == .2


def test_neighbor_allowance_applies_to_each_candidate_cell():
    scores = np.zeros((3, 3))
    counts = np.zeros((1, 3, 3), dtype=int)
    counts[0, 1, 1] = 20
    strict = _region_coverage(scores, 0., counts, 5., 5, 0)
    nearby = _region_coverage(scores, 0., counts, 5., 5, 1)
    assert strict['regions'][0]['n_unvisited_cells'] == 8 and strict['flag']
    assert nearby['regions'][0]['n_unvisited_cells'] == 0 and not nearby['flag']
    np.testing.assert_array_equal(strict['labels'], nearby['labels'])
    assert not nearby['unvisited_cells'].any()


def test_diagonal_candidates_are_separate_but_count_as_neighbors():
    scores = np.array([[0., -100.], [-100., -1.]])
    counts = np.array([[[5, 0], [0, 0]]])
    strict = _region_coverage(scores, 0., counts, 5., 5, 0)
    buffered = _region_coverage(scores, 0., counts, 5., 5, 1)
    assert len(strict['regions']) == len(buffered['regions']) == 2
    assert strict['regions'][1]['flag'] and not buffered['regions'][1]['flag']
    assert buffered['regions'][1]['boundary_sensitive']
    # Counting a border must not change the candidate boundary.
    np.testing.assert_array_equal(strict['labels'], buffered['labels'])


def test_minimum_samples_is_an_absolute_count_across_chains():
    scores = np.array([[0., -100.]])
    counts = np.array([[[2, 10000]], [[2, 10000]]])
    few = _region_coverage(scores, 0., counts, 5., 5, 0)
    assert few['flag'] and few['regions'][0]['unvisited_fraction'] == 0.
    assert few['regions'][0]['flag_reasons'] == ('low_sample_count',)
    assert few['regions'][0]['flag_peak_index'] == (0, 0)
    counts[1, 0, 0] = 3  # five total visits, although neither chain has five
    report = _region_coverage(scores, 0., counts, 5., 5, 0)
    assert not report['flag']
    assert report['regions'][0]['samples'] == 5
    np.testing.assert_array_equal(report['regions'][0]['samples_by_chain'], [2, 3])


def test_midpoint_bins_and_border_counts_do_not_clip_outside_draws():
    edges = _cell_edges(np.array([0., 1., 4.]), (-.5, 5.))
    np.testing.assert_array_equal(edges, [-.5, .5, 2.5, 5.])
    counts, _, _ = np.histogram2d([.4, 1.8, 2.7, 5., 6.], [0.] * 5,
                                 bins=(edges, [-1., 1.]))
    np.testing.assert_array_equal(counts.ravel(), [1, 1, 2])  # 6 is outside
    counts = np.array([[[0], [0], [10]]])
    scores = np.array([[-100.], [0.], [-100.]])
    raw = _region_coverage(scores, 0., counts, 5., 5, 0)
    nearby = _region_coverage(scores, 0., counts, 5., 5, 1)
    assert raw['flag'] and not nearby['flag']
    region = nearby['regions'][0]
    assert region['samples'] == 0 and region['nearby_samples'] == 10
    assert region['boundary_sensitive']


def test_reference_and_thresholds_exclude_weak_or_invalid_grid_regions():
    # A posterior reference well above this entire slice produces no candidates.
    axes, scores, counts = two_peaks()
    assert not _region_coverage(scores, 100., counts, 10., 5, 1)['regions']
    scores = np.array([[0., np.nan, -7., -np.inf]])
    counts = np.array([[[10, 0, 0, 0]]])
    reports = [_region_coverage(scores, 0., counts, t, 5, 0) for t in (3., 5., 10.)]
    assert not reports[0]['flag'] and not reports[1]['flag'] and reports[2]['flag']
    assert len(reports[2]['regions']) == 2


@pytest.mark.parametrize('linear_age', [True, False])
@pytest.mark.parametrize('feh_step', [None, .1])
def test_draw_scores_match_scalar_joint_density_and_chain_counts(fit_options, linear_age, feh_step):
    fit, options = fit_options
    options.update(linear_age=linear_age, prot=15., rho_obs=.8, rho_err=.2)
    initial, search = fit.find_initial(model_kwargs=options, return_grid=True, feh_step=feh_step)
    assert isinstance(search, InitialGridSearch)
    samples = {key: np.full((2, 3), value) for key, value in initial.items()}
    samples['eep'] += np.array([[-3., -2., -1.], [1., 2., 3.]])
    samples['feh_init'] = samples['feh_init'] * .9  # stay inside even when the best slice is near a bound
    samples['distance'] *= 1.01
    # Saved-only quantities, including invalid ones, must not condition density.
    samples['teff'] = np.full((2, 3), np.nan)
    result = check_missed_modes(search, samples, batch_size=4)
    assert result['grid_search'] is search
    assert result['threshold'] == 5.
    assert result['min_samples'] == 5 and result['neighbor_cells'] == 1
    assert result['max_unvisited_fraction'] == .1
    for region in result['regions']:
        peak = region['flag_peak_index']
        if peak is None:
            assert region['flag_peak_feh'] is None
        else:
            assert region['flag_peak_feh'] == search.best_feh[peak]
    for i in range(2):
        for j in range(3):
            point = {key: samples[key][i, j] for key in initial}
            expected = log_density(fit.model, (), options, point)[0]
            np.testing.assert_allclose(result['posterior_log_density'][i, j], expected, rtol=1e-12)
    np.testing.assert_array_equal(result['counts_by_chain'].sum(axis=(1, 2)), [3, 3])
    assert result['samples_outside_grid'] == 0
    assert result['reference_log_density'] == max(
        search.log_density.max(), result['posterior_log_density'].max())
    np.testing.assert_array_equal(result['best_feh'], search.best_feh)
    if fit.physical_model.kind == 'isochrone' and linear_age:
        np.testing.assert_allclose(result['sample_positions'][:, 0], np.log10(samples['age'].ravel()) + 9)
    # One draw dimension is accepted without pretending to know its chains.
    flat = check_missed_modes(search, {k: v.ravel() for k, v in samples.items()},
                               threshold=5., batch_size=4)
    np.testing.assert_array_equal(flat['counts_by_chain'][0], result['counts_by_chain'].sum(axis=0))
    np.testing.assert_array_equal(flat['labels'], result['labels'])
    assert flat['flag'] == result['flag']


def test_recheck_reuses_scores_and_counts_without_model_evaluation(fit_options, monkeypatch):
    import jax
    import jaxstar.mistfit.diagnostics as diagnostics
    fit, options = fit_options
    initial, search = fit.find_initial(model_kwargs=options, return_grid=True, feh_step=.1)
    samples = {key: np.full((2, 3), value) for key, value in initial.items()}
    original = check_missed_modes(search, samples, threshold=3., min_samples=2,
                                   neighbor_cells=0, max_unvisited_fraction=.25)
    expected = check_missed_modes(search, samples, threshold=12., min_samples=2,
                                   neighbor_cells=1, max_unvisited_fraction=None)
    def unexpected(*args, **kwargs):
        raise AssertionError('rechecking must not evaluate the model or scan')
    monkeypatch.setattr(diagnostics, '_search_grid', unexpected)
    monkeypatch.setattr(diagnostics, 'log_density', unexpected)
    monkeypatch.setattr(jax, 'jit', unexpected)
    changed = recheck_missed_modes(original, threshold=12., neighbor_cells=1,
                                    max_unvisited_fraction=None)
    np.testing.assert_array_equal(changed['labels'], expected['labels'])
    np.testing.assert_array_equal(changed['flagged_cells'], expected['flagged_cells'])
    np.testing.assert_array_equal(changed['unvisited_cells'], expected['unvisited_cells'])
    assert changed['flag'] == expected['flag']
    assert changed['min_samples'] == 2 and changed['max_unvisited_fraction'] is None
    for key in ('grid_search', 'grid_log_density', 'posterior_log_density', 'counts_by_chain'):
        assert changed[key] is original[key]
    assert original['threshold'] == 3. and original['neighbor_cells'] == 0
    assert original['max_unvisited_fraction'] == .25
    unchanged = recheck_missed_modes(original)
    np.testing.assert_array_equal(unchanged['labels'], original['labels'])
    for settings in (dict(threshold=-1.), dict(neighbor_cells=True),
                     dict(min_samples=0), dict(max_unvisited_fraction=2.)):
        with pytest.raises(ValueError):
            recheck_missed_modes(original, **settings)


@pytest.mark.parametrize('linear_age', [True, False])
def test_fit_coverage_recovers_target_and_scans_without_initializing(fit_options, linear_age, monkeypatch):
    import json
    from types import SimpleNamespace
    import jax
    from jaxstar._arviz import from_dict
    from jaxstar.mistfit import MistFit
    from jaxstar.mistfit.results import _new_run_record
    fit, options = fit_options
    options.update(linear_age=linear_age, prot=15., rho_obs=.8, rho_err=.2,
                   dist_scale=np.asarray(.7), output_keys=('kmag_abs',))
    initial, expected_search = fit.find_initial(model_kwargs=options, return_grid=True, feh_step=.1)
    samples = {key: np.full((2, 3), value) for key, value in initial.items()}
    record = _new_run_record('nuts', fit.model, (), options)
    result = SimpleNamespace(_jaxstar_run_metadata=record,
                             get_samples=lambda group_by_chain: samples)
    def unexpected(*args, **kwargs):
        raise AssertionError('diagnostic scan must not choose a NUTS start or calculate gradients')
    monkeypatch.setattr(MistFit, 'find_initial', unexpected)
    monkeypatch.setattr(jax, 'value_and_grad', unexpected)
    report = fit.check_missed_modes(result)
    search = report['grid_search']
    assert search.feh_step == .1 and len(search.feh_values) > 1
    assert search.model_kwargs['linear_age'] == linear_age
    assert search.model_kwargs['prot'] == 15.
    assert search.model_kwargs['dist_scale'] == .7
    np.testing.assert_allclose(search.log_density_by_feh, expected_search.log_density_by_feh)
    np.testing.assert_array_equal(report['counts_by_chain'].sum(axis=(1, 2)), [3, 3])
    for k in initial:
        np.testing.assert_array_equal(samples[k], np.full((2, 3), initial[k]))
    # Portable results retain the same target without the original sampler.
    idata = from_dict({'posterior': samples})
    idata.attrs['jaxstar_run_metadata'] = json.dumps(record)
    portable = fit.check_missed_modes(idata, feh_step=None, feh_init=-.1, distance=.15)
    assert portable['grid_search'].fixed_values == dict(feh_init=-.1, distance=.15)
    np.testing.assert_allclose(portable['posterior_log_density'], report['posterior_log_density'])


def test_fit_coverage_rejects_missing_or_mismatched_provenance(fit_options, monkeypatch):
    import copy
    from types import SimpleNamespace
    import jaxstar.mistfit.diagnostics as diagnostics
    from jaxstar.mistfit.results import _new_run_record
    fit, options = fit_options
    original = _new_run_record('nuts', fit.model, (), options)
    def unexpected(*args, **kwargs):
        raise AssertionError('reject a mismatched target before scanning')
    monkeypatch.setattr(diagnostics, '_search_grid', unexpected)
    with pytest.raises(ValueError, match='run metadata'):
        fit.check_missed_modes(SimpleNamespace())
    for key, value, message in (
        ('observations', {}, 'observations'),
        ('physical_model', {'grid_sha256': 'another grid'}, 'physical grid'),
        ('callable', {}, 'probability model'),
        ('arguments', {'age_obs': {'callable': 'unsupported'}}, 'not restorable'),
    ):
        record = copy.deepcopy(original)
        record['model'][key] = value
        with pytest.raises(ValueError, match=message):
            fit.check_missed_modes(SimpleNamespace(_jaxstar_run_metadata=record))


def test_invalid_samples_and_settings_fail_clearly(fit_options):
    fit, options = fit_options
    initial, search = fit.find_initial(model_kwargs=options, return_grid=True)
    samples = {key: np.full((1, 2), value) for key, value in initial.items()}
    for invalid in ({}, dict(samples, eep=np.empty((1, 0))),
                    dict(samples, eep=np.array([[np.nan, 400.]]))):
        with pytest.raises(ValueError, match='missing|shape|finite'):
            check_missed_modes(search, invalid, threshold=5.)
    with pytest.raises(ValueError, match='nonfinite joint density'):
        check_missed_modes(search, dict(samples, feh_init=np.full((1, 2), 100.)), threshold=5.)
    for kwargs in [dict(threshold=[]), dict(threshold=np.nan), dict(threshold=[3., 3.]), dict(threshold=0.),
                   dict(min_samples=0), dict(neighbor_cells=-1), dict(batch_size=0),
                   *[dict(max_unvisited_fraction=value) for value in
                     (-.1, 1.1, np.nan, np.inf, [], [.1, .2], True)]]:
        with pytest.raises(ValueError):
            check_missed_modes(search, samples, **dict({'threshold': 5.}, **kwargs))
    count_only = check_missed_modes(search, samples, max_unvisited_fraction=None)
    assert count_only['max_unvisited_fraction'] is None
    assert all('high_unvisited_fraction' not in r['flag_reasons'] for r in count_only['regions'])


@pytest.mark.parametrize('single_cell', [False, True])
def test_plot_renders_flagged_peaks_and_single_cell_grid(tmp_path, single_cell):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    axes, scores, counts = two_peaks()
    if single_cell:
        axes, scores, counts = (np.array([1.]), np.array([205.])), np.zeros((1, 1)), np.zeros((2, 1, 1))
    check = dict(
        axes=axes, axis_names=('mass', 'eep'), grid_log_density=scores,
        reference_log_density=0., fixed_values=dict(feh_init=-.2, distance=.125),
        sample_positions=np.array([[-3., 205.]]),
        edges=tuple(_cell_edges(axis, (axis[0] - .1, axis[-1] + .1)) for axis in axes),
        **_region_coverage(scores, 0., counts, 5., 5, 1),
        feh_step=.1,
    )
    fig = plot_missed_modes(check)
    fig.savefig(tmp_path / 'coverage.png')
    assert (tmp_path / 'coverage.png').stat().st_size > 1000
    assert len(fig.axes) == 2  # one mass/EEP plane and its color bar
    crosses = [line for line in fig.axes[0].lines if line.get_marker() == 'x']
    flagged = [region for region in check['regions'] if region['flag']]
    assert len(crosses) == len(flagged)
    for cross, region in zip(crosses, flagged):
        i, j = region['flag_peak_index']
        np.testing.assert_array_equal(cross.get_xdata(), [axes[0][i]])
        np.testing.assert_array_equal(cross.get_ydata(), [axes[1][j]])
    plt.close(fig)
