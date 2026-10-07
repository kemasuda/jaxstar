"""Compare production track integration with the unchanged inference prototype.

Offline: reads the validated NPZ and previously saved synthetic observations.
Checks physical predictions, both prior choices on common age support (with
the known normalization shift), soft age penalties, and finite-difference
unconstrained NUTS gradients. Short sampler runs check API/result plumbing,
not posterior convergence or physical between-node interpolation accuracy.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density
from jax.flatten_util import ravel_pytree
from scipy.special import expit

from jaxstar.mistfit import MistFit, MistModel, MistObservations, get_run_metadata, load_result
from mist_track_inference import MistTrackGrid, MistTrackFit, LATENTS
from validate_mist_bolometric_corrections import file_record


CASES = dict(main_sequence=(1.03, -.08, 330.4, .15),
             turnoff=(1.16, 0., 435.5, .22), red_giant=(1.12, -.08, 500.4, .30))


def difference(actual, expected, *, rtol=3e-10, atol=3e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return {"max_absolute": float(np.max(np.abs(actual-expected))),
            "max_scaled": float(np.max(np.abs(actual-expected)/(1+np.abs(expected))))}


def check_case(name, truth, model, prototype_grid, args, index):
    source = args.reference_dir/f'{name}.npz'
    with np.load(source, allow_pickle=False) as data:
        observations = MistObservations(tuple(data['obskeys'].tolist()), tuple(data['obsvals']), tuple(data['obserrs']))
        saved = np.column_stack([data[key].ravel() for key in LATENTS])
    rng = np.random.default_rng(827+index)
    selected = saved[rng.choice(len(saved), 32, replace=False)]
    # Include newly admitted masses as well as prior saved posterior locations.
    wide = rng.uniform([.81, -.24, 203., .1], [1.49, .24, 549., .4], size=(64, 4))
    # Use actual grid bounds if passed a different validated rectangular NPZ.
    for i, axis in enumerate(prototype_grid.axes):
        wide[:, i] = rng.uniform(float(axis[0])+.01*(axis[-1]-axis[0]),
                                 float(axis[-1])-.01*(axis[-1]-axis[0]), size=len(wide))
    age = np.asarray(jax.jit(jax.vmap(lambda p: prototype_grid.values(*p)['logage']))(wide))
    wide = wide[(age >= 8.) & (age <= 10.14)][:16]
    points = np.vstack((np.asarray(truth)[None], selected, wide))
    output_keys = tuple(prototype_grid.values(*truth))

    def prediction(m, p):
        values = m(dict(zip(('mass_init', 'feh_init', 'eep', 'distance'), p)), keys=output_keys)
        return jnp.stack([values[k] for k in output_keys])

    def old_prediction(p):
        values = prototype_grid.values(*p)
        return jnp.stack([values[k] for k in output_keys])

    new_predictions = jax.jit(jax.vmap(prediction, in_axes=(None, 0)))(model, points)
    old_predictions = jax.jit(jax.vmap(old_prediction))(points)
    report = {'observations_source': file_record(source), 'points': len(points),
              'predictions': difference(new_predictions, old_predictions), 'priors': {}}
    fit = MistFit(model=model, observations=observations)
    for linear_age in (True, False):
        old = MistTrackFit(prototype_grid, observations.keys, observations.values, observations.errors,
                           linear_age=linear_age)
        options = {'linear_age': linear_age, 'age_max': None}
        def new_density(p):
            return log_density(fit.model, (), options, dict(zip(LATENTS, p)))[0]
        def old_density(p):
            return log_density(old.model, (), {}, dict(zip(LATENTS, p)))[0]
        new_value, new_gradient = jax.jit(jax.vmap(jax.value_and_grad(new_density)))(points)
        old_value, old_gradient = jax.jit(jax.vmap(jax.value_and_grad(old_density)))(points)
        age_width = 10**(10.14-9)-10**(8.-9) if linear_age else 10.14-8.
        shift = np.log(age_width) - np.log(float(prototype_grid.axes[2][-1]-prototype_grid.axes[2][0]))
        prior_report = {'common_support_log_density_after_constant_shift': difference(new_value, old_value+shift),
                        'expected_new_minus_old': float(shift),
                        'constrained_gradient': difference(new_gradient, old_gradient, atol=3e-7)}
        def bounded_density(p):
            return log_density(fit.model, (), dict(options, age_max=13.8), dict(zip(LATENTS, p)))[0]
        bounded_value = jax.jit(jax.vmap(bounded_density))(points)
        ages = np.asarray(new_predictions[:, output_keys.index('age')])
        prior_report['soft_upper_log_penalty'] = difference(
            bounded_value-new_value, -30.*expit((ages-13.8)/.1))
        # Some truth coordinates lie exactly on mass/FeH grid knots. A centered
        # finite difference straddles two linear cells there; use a random
        # interior point to check the derivative within a single cell.
        initial = dict(zip(LATENTS, points[-1]))
        new_info = initialize_model(jax.random.PRNGKey(19), fit.model, model_kwargs=options,
                                    init_strategy=init_to_value(values=initial))
        z, unravel = ravel_pytree(new_info.param_info.z)
        potential = jax.jit(lambda flat: new_info.potential_fn(unravel(flat)))
        h = 1e-5
        finite = np.array([(potential(z+d)-potential(z-d))/(2*h) for d in np.eye(len(z))*h])
        prior_report['potential_gradient'] = difference(
            jax.grad(potential)(z), finite, rtol=3e-6, atol=3e-6)
        report['priors']['age' if linear_age else 'logage'] = prior_report

    if args.samples:
        started = perf_counter()
        result = fit.run_nuts(jax.random.PRNGKey(913+index),
            model_kwargs={'age_max': args.age_max},
            init_strategy=init_to_value(values=dict(zip(LATENTS, truth))),
            num_chains=2, chain_method='sequential', num_warmup=args.warmup,
            num_samples=args.samples, target_accept_prob=.99, max_tree_depth=11,
            output_keys=('rho', 'kmag_abs'), progress_bar=False)
        samples = result.get_samples(group_by_chain=True)
        assert all(np.all(np.isfinite(v)) for v in samples.values())
        assert np.all(np.asarray(samples['dtdeep']) > 0)
        np.testing.assert_array_equal(samples['eep_lower'], np.full_like(samples['eep_lower'], prototype_grid.axes[2][0]))
        np.testing.assert_array_equal(samples['eep_upper'], np.full_like(samples['eep_upper'], prototype_grid.axes[2][-1]))
        assert np.all((samples['eep'] >= samples['eep_lower']) & (samples['eep'] <= samples['eep_upper']))
        metadata = get_run_metadata(result)
        assert metadata['model']['physical_model']['kind'] == 'track'
        assert metadata['model']['callable']['callable'].endswith('.model_track')
        directory = fit.save_result(result, args.output_dir/name)
        reloaded = load_result(directory)
        for key in samples:
            np.testing.assert_array_equal(reloaded.posterior[key].values, samples[key])
        report['nuts_smoke'] = {'seconds': perf_counter()-started, 'chains': 2,
            'age_max_gyr': args.age_max,
            'warmup_per_chain': args.warmup, 'samples_per_chain': args.samples,
            'divergences': int(np.asarray(result.get_extra_fields()['diverging']).sum()),
            'save_reload_exact': True, 'result_dir': str(directory),
            'interpretation': 'API/sampler integration smoke test, not a convergence assessment.'}
    print(name, json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, default=Path('benchmark-results/mist-track-grid-extended/grid.npz'))
    parser.add_argument('--reference-dir', type=Path, default=Path('benchmark-results/mist-track-inference'))
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark-results/mist-track-soft-age-bounds'))
    parser.add_argument('--age-max', type=float, default=13.8, help='smooth Gyr bound for the NUTS smoke runs (default: 13.8)')
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--samples', type=int, default=100, help='per chain; 0 skips sampling')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'summary.json').exists():
        raise FileExistsError('Choose a fresh output directory to preserve previous results')
    model, prototype = MistModel.from_tracks(args.grid), MistTrackGrid(args.grid)
    report = {'grid': file_record(args.grid), 'sources': [file_record(Path(p)) for p in (
        __file__, 'benchmarks/mist_track_inference.py', 'src/jaxstar/mistfit/model.py',
        'src/jaxstar/mistfit/_data.py', 'src/jaxstar/mistfit/inference.py',
        'src/jaxstar/mistfit/mistgrid/tracks.py', 'src/jaxstar/mistfit/mistfit.py',
        'src/jaxstar/mistfit/results.py')], 'cases': {}}
    # Extreme native ages remain supported; compare the default soft bound
    # to explicitly disabling it, including the old-age tail.
    prior_fit = MistFit(model=model, observations={})
    extremes = []
    for index in (np.argmin(prototype.native['logage']), np.argmax(prototype.native['logage'])):
        indices = np.unravel_index(index, prototype.native['logage'].shape)
        point = dict(zip(LATENTS, [float(axis[i]) for axis, i in zip(prototype.axes, indices)]+[.2]))
        base, trace = log_density(prior_fit.model, (), {'age_max': None}, point)
        limited, _ = log_density(prior_fit.model, (), {}, point)
        age = float(trace['age']['value'])
        assert np.isfinite(base) and np.isfinite(limited)
        difference(limited-base, -30.*expit((age-13.8)/.1))
        extremes.append({'age_gyr': age, 'unbounded_log_density': float(base),
                         'default_log_density': float(limited),
                         'soft_upper_log_penalty': float(limited-base)})
    report['native_age_extremes'] = extremes
    for i, (name, truth) in enumerate(CASES.items()):
        report['cases'][name] = check_case(name, truth, model, prototype, args, i)
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
