"""Offline result persistence, provenance and diagnostics for both samplers."""

from hashlib import sha256
import json
from pathlib import Path

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jaxstar.mistfit import (MistFit, MistModel, to_inference_data, get_run_metadata,
                             save_result, load_result, summarize, run_nuts,
                             plot_trace, plot_corner, plot_diagnostics)
from test_mistmodel import grid_path, make_fit, OPTIONS, PARAMS


@pytest.fixture
def nuts_run(grid_path, x64_context):
    with x64_context():
        fit = make_fit(MistFit, grid_path, 'named')
        result = fit.run_nuts(jax.random.PRNGKey(41), model_kwargs=dict(OPTIONS, linear_age=False),
                              output_keys=('kmag_abs', 'mass_current', 'rho'),
                              init_strategy=init_to_value(values=PARAMS), num_chains=2,
                              chain_method='vectorized', num_warmup=24, num_samples=32,
                              max_tree_depth=4, progress_bar=False)
        yield fit, result
    plt.close('all')


def test_nuts_portable_roundtrip_and_grid_free_redraw(nuts_run, tmp_path, monkeypatch):
    fit, mcmc = nuts_run
    assert isinstance(mcmc, MCMC)
    record = get_run_metadata(mcmc)
    assert record['random']['key_data'] == [0, 41]
    assert record['sampling']['chain_method_used'] == 'vectorized'
    assert record['sampling']['num_chains'] == 2
    assert record['model']['arguments']['linear_age'] is False
    assert record['model']['arguments']['prot'] is None  # Effective model defaults recorded too.
    assert record['model']['arguments']['output_keys'] == ['kmag_abs', 'mass_current', 'rho']
    assert record['model']['observations']['keys'] == list(fit.observations.keys)
    source = record['model']['physical_model']['sources'][0]
    assert source['sha256'] == sha256(Path(source['path']).read_bytes()).hexdigest()
    assert len(record['model']['physical_model']['grid_sha256']) == 64
    changed = get_run_metadata(mcmc)
    changed['sampling']['num_chains'] = 100
    assert get_run_metadata(mcmc)['sampling']['num_chains'] == 2
    monkeypatch.setattr(MistModel, '__call__', lambda *a, **kw: pytest.fail('NUTS postprocessing must not re-evaluate the grid'))
    idata = fit.to_inference_data(mcmc)
    assert set(idata.posterior) == set(mcmc.get_samples())
    assert idata.posterior.sizes['chain'] == 2 and idata.posterior.sizes['draw'] == 32
    assert {'diverging', 'potential_energy', 'n_steps', 'acceptance_rate', 'step_size'} <= set(idata.sample_stats)
    assert 'lp' not in idata.sample_stats  # Potential energy is not mislabeled as log probability.
    assert idata.posterior['mass'].attrs['long_name'] == 'Initial mass'
    assert idata.posterior['mass_current'].attrs['long_name'] == 'Current mass'
    directory = fit.save_result(idata, tmp_path / 'star')
    assert {'posterior.nc', 'run.json', 'summary.csv'} <= {p.name for p in directory.iterdir()}
    # Reload after removing the source file: neither loading nor plotting needs a grid.
    Path(source['path']).unlink()
    loaded = load_result(directory)
    groups = idata.groups() if callable(idata.groups) else (g.strip('/') for g in idata.groups if g != '/')
    for group in groups:
        for name in getattr(idata, group).data_vars:
            a, b = getattr(idata, group)[name], getattr(loaded, group)[name]
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
    assert get_run_metadata(loaded) == get_run_metadata(idata)
    np.testing.assert_array_equal(loaded.observed_data.observable, fit.observations.keys)
    np.testing.assert_array_equal(loaded.constant_data.error, fit.observations.errors)
    table = summarize(loaded, keys=('age', 'mass'))
    assert {'r_hat', 'ess_bulk', 'ess_tail'} <= set(table.columns)
    figures = plot_diagnostics(loaded, keys=['mass', 'radius', 'feh'], output_dir=directory)
    assert set(figures) == {'trace', 'corner'}
    assert all(isinstance(fig, matplotlib.figure.Figure) for fig in figures.values())
    assert (directory / 'trace.png').stat().st_size > 1000
    assert (directory / 'corner.png').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        save_result(loaded, directory)
    with pytest.raises(FileExistsError):
        plot_diagnostics(loaded, keys='mass', output_dir=directory)
    save_result(loaded, directory, overwrite=True)
    assert load_result(directory / 'posterior.nc').posterior.sizes == loaded.posterior.sizes
    assert not hasattr(fit, 'samples') and not hasattr(fit, 'mcmc')


@pytest.mark.parametrize('sampler', ['nuts', 'multinest'])
def test_summary_accepts_string_list_and_tuple_keys(sampler, arviz_from_dict):
    rng = np.random.default_rng(5)
    idata = arviz_from_dict(posterior={key: rng.normal(size=(4, 64))
                                    for key in ('mass', 'radius', 'age', 'eep')})
    idata.attrs['jaxstar_run_metadata'] = json.dumps({'sampler': sampler})
    keys = ('mass', 'radius', 'age')  # The selection shown in the notebook.
    expected = summarize(idata, keys=list(keys))
    actual = summarize(idata, keys=keys)
    assert tuple(actual.index) == keys
    assert actual.equals(expected)
    assert tuple(summarize(idata, keys='mass').index) == ('mass',)
    assert tuple(summarize(idata).index) == tuple(idata.posterior.data_vars)
    assert ('r_hat' in actual.columns) == (sampler == 'nuts')
    assert len([key for key in actual.columns if key.startswith('hdi')]) == 2
    assert not any(key.startswith('eti') for key in actual.columns)


def test_corrupt_bundle_is_rejected(nuts_run, tmp_path):
    directory = save_result(nuts_run[1], tmp_path / 'star')
    path = directory / 'posterior.nc'
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='does not match'):
        load_result(directory)


def test_metadata_and_diagnostic_collection_do_not_change_seeded_sampling():
    def gaussian():
        numpyro.sample('x', dist.Normal(0., 1.))
    key = jax.random.PRNGKey(9)
    new = run_nuts(gaussian, key, num_chains=1, num_warmup=8, num_samples=16,
                   target_accept_prob=.95, dense_mass=False, max_tree_depth=3, progress_bar=False)
    old = MCMC(NUTS(gaussian, target_accept_prob=.95, dense_mass=False, max_tree_depth=3),
               num_chains=1, num_warmup=8, num_samples=16, progress_bar=False)
    old.run(key)
    np.testing.assert_array_equal(new.get_samples()['x'], old.get_samples()['x'])
    np.testing.assert_array_equal(new.get_extra_fields()['diverging'], old.get_extra_fields()['diverging'])
    assert get_run_metadata(old)['metadata_complete'] is False
    assert to_inference_data(old).posterior.sizes['draw'] == 16


def test_record_typed_rng_key_and_default_sampling_options():
    def gaussian():
        numpyro.sample('x', dist.Normal(0., 1.))
    result = run_nuts(gaussian, jax.random.key(23), num_chains=1, num_warmup=4,
                      num_samples=4, extra_fields=(), progress_bar=False)
    assert get_run_metadata(result)['random']['key_data'] == [0, 23]
    assert set(to_inference_data(result).sample_stats) == {'diverging'}


@pytest.fixture
def nested_run(grid_path, tmp_path, monkeypatch, x64_context):
    mn = pytest.importorskip('numpyro_inferutils.experimental.multinest')
    prefix = str(tmp_path / 'native') + '/'
    def solve(**kwargs):
        rng = np.random.default_rng(19)
        samples = np.stack([kwargs['Prior'](u) for u in rng.uniform(.25, .75, size=(64, kwargs['n_dims']))])
        name = kwargs['outputfiles_basename']
        np.savetxt(name + 'post_equal_weights.dat', samples)
        np.savetxt(name + '.txt', np.column_stack([np.full(64, 1/64), np.zeros(64), samples]))
        Path(name + 'stats.dat').write_text('native evidence record\n')
        Path(name + 'notes.ipynb').write_text('unrelated user work\n')
        return dict(samples=samples, logZ=-123.45, logZerr=.12)
    monkeypatch.setattr(mn, '_import_pymultinest', lambda: (solve, None))
    with x64_context():
        fit = make_fit(MistFit, grid_path, 'named')
        runner = fit.run_multinest(outputfiles_basename=prefix, model_kwargs=OPTIONS,
                                   output_keys=('rho', 'kmag_abs'), n_live_points=25,
                                   seed=19, resume=False)
        yield fit, runner, Path(prefix)
    plt.close('all')


def test_multinest_equal_weight_draws_evidence_native_files_and_reload(nested_run, tmp_path, monkeypatch):
    fit, runner, native = nested_run
    record = get_run_metadata(runner)
    assert record['sampling']['seed'] == 19
    assert record['sampling']['resolved_n_live_points'] == 25
    assert record['sampling']['resolved_sampling_efficiency'] == .8
    idata = to_inference_data(runner)
    assert idata.posterior.sizes['chain'] == 1
    assert idata.posterior.sizes['draw'] == 64
    assert {'rho', 'kmag_abs', *runner.site_names} <= set(idata.posterior)
    for i, name in enumerate(runner.site_names):
        np.testing.assert_array_equal(idata.posterior[name].values[0], runner.result['samples'][:, i])
    np.testing.assert_allclose(idata.posterior.rho, idata.posterior.star_mass / idata.posterior.radius**3)
    assert get_run_metadata(idata)['evidence'] == {'logZ': -123.45, 'logZerr': .12}
    assert not hasattr(idata, 'sample_stats')
    table = summarize(idata)
    assert not {'r_hat', 'ess_bulk', 'ess_tail'} & set(table.columns)
    directory = save_result(idata, tmp_path / 'nested')
    assert not (directory / 'multinest').exists()
    assert {'.txt', 'stats.dat', 'post_equal_weights.dat'} <= {p.name for p in directory.iterdir()}
    assert (native / 'notes.ipynb').read_text() == 'unrelated user work\n'
    for name in ('.txt', 'stats.dat', 'post_equal_weights.dat'):
        assert (directory / name).read_bytes() == (native / name).read_bytes()
    monkeypatch.setattr(MistModel, '__call__', lambda *a, **kw: pytest.fail('reload/plot must not evaluate the model'))
    loaded = load_result(directory)
    with pytest.raises(ValueError, match='no MCMC chain trace'):
        fit.plot_trace(loaded)
    figures = fit.plot_diagnostics(loaded, keys=('age', 'mass', 'kmag_abs'), output_dir=directory)
    assert set(figures) == {'corner'} and not (directory / 'trace.png').exists()
    second = save_result(loaded, tmp_path / 'nested-copy')
    assert (second / '.txt').read_bytes() == (directory / '.txt').read_bytes()
    assert get_run_metadata(load_result(second)) == get_run_metadata(loaded)


def test_plot_references_respect_physical_keys_and_allow_constant_output(arviz_from_dict):
    rng = np.random.default_rng(1)
    draws = {k: rng.normal(center, .01, (2, 40)) for k, center in {'feh': 0., 'feh_init': .05, 'kmag': 8., 'kmag_abs': 3.}.items()}
    draws['constant'] = np.ones((2, 40))
    idata = arviz_from_dict(posterior=draws, sample_stats={'diverging': np.zeros((2,40), dtype=bool)})
    record = {'sampler': 'nuts', 'model': {'observations': {'keys': ['feh', 'kmag'], 'values': [0., 8.], 'errors': [.1, .03]}}}
    idata.attrs['jaxstar_run_metadata'] = json.dumps(record)
    keys = ['feh', 'feh_init', 'kmag', 'kmag_abs']
    with_refs = plot_trace(idata, keys=keys)
    without = plot_trace(idata, keys=keys, show_observations=False)
    for i, key in enumerate(keys):
        for panel in (0,1):
            assert len(with_refs.axes[2*i+panel].lines) - len(without.axes[2*i+panel].lines) == (1 if key in ('feh','kmag') else 0)
    corner = plot_corner(idata, keys=['feh', 'constant'])
    assert len(corner.axes) == 4
    with pytest.raises(KeyError, match='was not saved'):
        plot_corner(idata, keys='rho')
    with pytest.raises(ValueError, match='distinct'):
        plot_trace(idata, keys=['feh', 'feh'])
    with pytest.raises(ValueError, match='equal-weight'):
        plot_corner(idata, keys='feh', weights=np.ones(80))
    bad = arviz_from_dict(posterior={'vector': np.ones((2, 40, 3)), 'bad': np.full((2,40), np.nan)})
    with pytest.raises(ValueError, match='scalar'):
        plot_trace(bad, keys='vector')
    with pytest.raises(ValueError, match='nonfinite'):
        plot_corner(bad, keys='bad')
    plt.close('all')


def test_multinest_archive_refuses_files_overwritten_by_another_run(nested_run, tmp_path):
    _, runner, native = nested_run
    (native / 'stats.dat').write_text('different run')
    with pytest.raises(ValueError, match='changed since this run'):
        save_result(runner, tmp_path / 'mixed')
    assert not (tmp_path / 'mixed' / 'posterior.nc').exists()
    directory = save_result(runner, tmp_path / 'draws-only', copy_native=False)
    assert not (directory / 'multinest').exists()
    assert get_run_metadata(load_result(directory))['evidence']['logZ'] == -123.45


def test_multinest_relative_prefix_survives_long_cwd_and_later_directory_change(
        grid_path, tmp_path, monkeypatch, x64_context):
    mn = pytest.importorskip('numpyro_inferutils.experimental.multinest')
    work = tmp_path / ('long-working-directory-' * 5)
    work.mkdir()
    monkeypatch.chdir(work)
    prefix = 'results/native/run-'

    def solve(**kwargs):
        # Model the filename buffer in older native MultiNest libraries.
        name = kwargs['outputfiles_basename']
        assert name == prefix
        theta = kwargs['Prior'](np.full(kwargs['n_dims'], .5))
        samples = np.stack([theta, theta])
        np.savetxt((name + 'post_equal_weights.dat')[:100], samples)
        Path((name + 'stats.dat')[:100]).write_text('native stats\n')
        # Python analysis expects the full suffix, as PyMultiNest does.
        return dict(samples=np.loadtxt(name + 'post_equal_weights.dat'),
                    logZ=-10., logZerr=.1)

    monkeypatch.setattr(mn, '_import_pymultinest', lambda: (solve, None))
    with x64_context():
        fit = make_fit(MistFit, grid_path, 'named')
        runner = fit.run_multinest(outputfiles_basename=prefix, model_kwargs=OPTIONS)
        record = get_run_metadata(runner)
        absolute = str(work / prefix)
        assert record['native_output_prefix'] == runner.outputfiles_basename == absolute
        assert record['sampling']['outputfiles_basename'] == prefix
        assert set(record['native_files']) == {'run-post_equal_weights.dat', 'run-stats.dat'}
        monkeypatch.chdir(tmp_path)
        saved = save_result(runner, tmp_path / 'portable')
        np.testing.assert_array_equal(
            load_result(saved).posterior.age.values[0],
            runner.result['samples'][:, runner.site_names.index('age')])
        assert (saved / 'run-stats.dat').read_text() == 'native stats\n'


def test_multinest_saves_and_plots_in_its_output_directory(nested_run):
    fit, runner, directory = nested_run
    names = ('.txt', 'stats.dat', 'post_equal_weights.dat')
    original = {name: ((directory / name).read_bytes(), (directory / name).stat().st_mtime_ns)
                for name in names}
    idata = fit.to_inference_data(runner)
    fit.save_result(idata, directory)
    figures = fit.plot_diagnostics(idata, output_dir=directory, keys=('mass', 'radius', 'age'))
    assert set(figures) == {'corner'}
    assert not (directory / 'multinest').exists()
    assert {'posterior.nc', 'run.json', 'summary.csv', 'corner.png', *names} <= {p.name for p in directory.iterdir()}
    for name, (contents, modified_at) in original.items():
        assert (directory / name).read_bytes() == contents
        assert (directory / name).stat().st_mtime_ns == modified_at
    assert (directory / 'notes.ipynb').read_text() == 'unrelated user work\n'
    loaded = load_result(directory)
    assert get_run_metadata(loaded) == get_run_metadata(idata)
    assert set(json.loads((directory / 'run.json').read_text())['native_files']) == set(names)


def test_previous_multinest_bundle_layout_can_be_loaded_and_resaved_flat(nested_run, tmp_path):
    _, runner, _ = nested_run
    directory = save_result(runner, tmp_path / 'previous-layout')
    manifest = json.loads((directory / 'run.json').read_text())
    previous = directory / 'multinest'
    previous.mkdir()
    for name in manifest.pop('native_files'):
        (directory / name).rename(previous / name)
        manifest['files']['multinest/' + name] = manifest['files'].pop(name)
    (directory / 'run.json').write_text(json.dumps(manifest))
    loaded = load_result(directory)
    flat = save_result(loaded, tmp_path / 'flat-layout')
    assert not (flat / 'multinest').exists()
    for name in ('.txt', 'stats.dat', 'post_equal_weights.dat'):
        assert (flat / name).read_bytes() == (previous / name).read_bytes()
    assert get_run_metadata(load_result(flat)) == get_run_metadata(loaded)
