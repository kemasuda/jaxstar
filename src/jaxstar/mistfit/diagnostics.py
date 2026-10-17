"""Flag candidate regions that may contain posterior modes missed by sampling.

This checks a fixed-distance slice, maximizing over any scanned FeH values. Neither
an absent flag nor occupied cells certify that every posterior mode was found.
No region probability is inferred from the grid's joint-density heights.
"""

__all__ = ["check_missed_modes", "recheck_missed_modes", "plot_missed_modes"]

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density
from scipy.ndimage import binary_dilation, label

from .inference import InitialGridSearch, _default_model, _search_grid


def _restore_option(value):
    """Restore numeric model options recorded by results._json_value."""
    if isinstance(value, dict):
        if set(value) == {'values', 'dtype', 'shape'}:
            return np.asarray(value['values'], dtype=value['dtype']).reshape(value['shape'])
        raise ValueError('recorded model options are not restorable; use '
                         'check_missed_modes(search, samples) with an explicit target')
    if isinstance(value, list):
        return [_restore_option(item) for item in value]
    return value


def _check_fit_missed_modes(fit, result, *, feh_step, feh_init, distance, **settings):
    """Recover the sampling target, then perform an independent diagnostic scan."""
    from .results import get_run_metadata, _grid_record, _json_value
    from jaxstar._arviz import is_inference_data

    record = get_run_metadata(result).get('model', {})
    if not {'callable', 'arguments', 'observations', 'physical_model'} <= record.keys():
        raise ValueError('coverage requires jaxstar run metadata; use '
                         'check_missed_modes(search, samples) for an unrecorded result')
    if record['callable'] != _json_value(_default_model(fit.physical_model)):
        raise ValueError('result uses a different probability model from this MistFit')
    observed = dict(keys=list(fit.observations.keys), values=list(fit.observations.values),
                    errors=list(fit.observations.errors))
    if record['observations'] != _json_value(observed):
        raise ValueError('result observations do not match this MistFit')
    fingerprint = _grid_record(fit.physical_model)['grid_sha256']
    if record['physical_model'].get('grid_sha256') != fingerprint:
        raise ValueError('result physical grid does not match this MistFit')
    options = {key: _restore_option(value) for key, value in record['arguments'].items()}
    if is_inference_data(result):
        samples = {key: np.asarray(value.transpose('chain', 'draw').values)
                   for key, value in result.posterior.data_vars.items()
                   if set(value.dims) == {'chain', 'draw'}}
    else:
        samples = result.get_samples(group_by_chain=True)
    search = _search_grid(fit.observations, fit.physical_model, model_kwargs=options,
                          feh_step=feh_step, feh_init=feh_init, distance=distance)
    return check_missed_modes(search, samples, **settings)


def _coverage_settings(threshold, min_samples, neighbor_cells, max_unvisited_fraction):
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be a finite positive scalar')
    if max_unvisited_fraction is not None:
        fraction = np.asarray(max_unvisited_fraction, dtype=float)
        if (isinstance(max_unvisited_fraction, (bool, np.bool_)) or fraction.ndim
                or not np.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError('max_unvisited_fraction must be None or a finite scalar in [0, 1]')
        max_unvisited_fraction = float(fraction)
    for name, value, minimum in [('min_samples', min_samples, 1), ('neighbor_cells', neighbor_cells, 0)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    return dict(threshold=float(threshold), min_samples=int(min_samples),
                neighbor_cells=int(neighbor_cells), max_unvisited_fraction=max_unvisited_fraction)


def _cell_edges(axis, bounds):
    """Voronoi cells on a possibly nonuniform axis, clipped to search support."""
    return np.r_[bounds[0], axis[:-1] + np.diff(axis) / 2, bounds[1]]


def _region_coverage(scores, reference, counts, threshold, min_samples, neighbor_cells,
                     max_unvisited_fraction=0.1):
    """Check total visits and unvisited cells inside each candidate region."""
    delta = np.where(np.isfinite(scores), reference - scores, np.inf)
    # Face connectivity avoids joining islands that only touch diagonally.
    labels, number = label(delta <= threshold)
    visited = counts.sum(axis=0) > 0
    if neighbor_cells:
        visited = binary_dilation(visited, structure=np.ones((3, 3), dtype=bool),
                                  iterations=neighbor_cells)
    unvisited_cells = (labels > 0) & ~visited
    regions = []
    flagged_cells = np.zeros_like(labels, dtype=bool)
    for region_id in range(1, number + 1):
        region = labels == region_id
        neighborhood = (binary_dilation(region, structure=np.ones((3, 3), dtype=bool),
                                        iterations=neighbor_cells)
                        if neighbor_cells else region)
        per_chain = counts[:, region].sum(axis=1)
        nearby = counts[:, neighborhood].sum(axis=1)
        unvisited = region & unvisited_cells
        n_cells, n_unvisited = int(region.sum()), int(unvisited.sum())
        unvisited_fraction = n_unvisited / n_cells
        count_flag = bool(nearby.sum() < min_samples)
        fraction_flag = (max_unvisited_fraction is not None
                         and unvisited_fraction > max_unvisited_fraction)
        reasons = tuple(reason for reason, applies in (
            ('low_sample_count', count_flag),
            ('high_unvisited_fraction', fraction_flag),
        ) if applies)
        flagged = bool(reasons)
        if flagged:
            flagged_cells |= region
        peak = np.unravel_index(np.argmax(np.where(region, scores, -np.inf)), scores.shape)
        # Point to the missed part, even if the overall region peak was visited.
        flag_peak = (np.unravel_index(np.argmax(np.where(unvisited, scores, -np.inf)), scores.shape)
                     if fraction_flag else peak if flagged else None)
        regions.append(dict(
            id=region_id, n_cells=n_cells, peak_index=peak,
            peak_delta_logp=float(delta[peak]),
            samples=int(per_chain.sum()), samples_by_chain=per_chain,
            nearby_samples=int(nearby.sum()), nearby_samples_by_chain=nearby,
            n_unvisited_cells=n_unvisited, unvisited_fraction=unvisited_fraction,
            flag=flagged, flag_reasons=reasons, flag_peak_index=flag_peak,
            flag_peak_delta_logp=None if flag_peak is None else float(delta[flag_peak]),
            boundary_sensitive=bool(per_chain.sum() < min_samples and not count_flag),
        ))
    return dict(threshold=float(threshold), labels=labels, regions=regions,
                flagged_cells=flagged_cells, unvisited_cells=unvisited_cells,
                max_unvisited_fraction=max_unvisited_fraction,
                flag=any(region['flag'] for region in regions))


def check_missed_modes(search, samples, *, threshold=5., min_samples=5,
                        neighbor_cells=1, max_unvisited_fraction=0.1, batch_size=512):
    """Flag possible missed modes using candidate-region visits.

    ``search`` comes from ``fit.find_initial(..., return_grid=True)``. Supply
    constrained samples from the SAME model, observations and scientific options,
    e.g. ``mcmc.get_samples(group_by_chain=True)``. Scalar latent arrays must have
    shape (chain, draw) or (draw,); deterministic outputs are ignored. Only this
    diagnostic runs: no sampler or additional model search is launched.

    Grid and draws use the same constrained joint density (NOT NUTS potential
    energy, which includes transform Jacobians). The reference is the largest
    finite score among both. Posterior scores are evaluated in bounded batches.

    The grid map takes the maximum over scanned initial FeH, not a marginal
    probability. Counts project samples onto the same two spatial axes.
    threshold=5. selects cells with delta_logp <= threshold, relative to the
    same global reference. Four-neighbor (face-connected) cells form a candidate
    region; samples are binned at native-node midpoints.
    A region is flagged when fewer than min_samples draws fall inside it or its
    neighbor_cells-wide border (including diagonal neighbors). min_samples is
    an absolute count summed over all chains, not a fraction or an effective
    sample size. Nearby draws are allowed without enlarging the candidate
    boundary. Raw and buffered counts are both returned, also by
    chain. Border counts may overlap between regions and must not be added
    as disjoint probabilities. Flagging uses raw
    counts when neighbor_cells=0. No thinning or ESS-based significance is used.

    Independently, a cell is visited if at least one draw falls in it or within
    neighbor_cells (including diagonals). For each candidate region, count the
    fraction of its cells still unvisited. A fraction strictly greater than
    max_unvisited_fraction=0.1 also flags the region, even when its total count
    is large. Each cell has equal weight: this is neither a coordinate-area
    fraction nor a posterior probability. It depends on grid spacing, sample
    count and neighbor_cells. Set max_unvisited_fraction=None to disable only
    this condition; otherwise it must be a finite scalar in [0, 1].

    Each check uses one positive scalar threshold for both candidate selection
    and connectivity. Override the defaults to explore sensitivity; these
    settings are diagnostic thresholds, not calibrated tests.
    No candidate region means this slice supplies no high-score candidates; it
    is not evidence of unimodality. Connectivity may merge peaks joined by a
    high-score ridge. The cell check can detect unvisited parts of that region,
    but small missing parts below the fraction cutoff can still be missed.
    Projection can hide separation in FeH/distance, and modes between grid
    nodes or outside this slice can be missed entirely.

    Returns a dict: regions contains each component's peak_index, peak_feh,
    peak_delta_logp, raw/nearby counts (also by chain), n_unvisited_cells,
    unvisited_fraction, flag_reasons, flag and boundary_sensitive. Reasons are
    'low_sample_count' and/or 'high_unvisited_fraction'. boundary_sensitive
    describes only a raw-count shortfall rescued by neighboring draws.
    flag_peak_index/flag_peak_feh/flag_peak_delta_logp locate the plotted cross:
    the highest-scoring unvisited cell for a fraction flag, otherwise the region
    peak for a count flag, or None without a flag. peak_index always retains
    the overall region peak.
    flag is true if any component is flagged. labels is the integer component
    map (zero outside candidates), while flagged_cells marks flagged components.
    unvisited_cells marks unvisited candidate cells after the neighbor allowance.
    Grid/draw scores, axes, edges and sample_positions are included for plotting.
    grid_search retains the full InitialGridSearch, including every FeH slice
    and its target. recheck_missed_modes can change the diagnostic settings
    without another grid search, density evaluation or sampling run.
    """
    if not isinstance(search, InitialGridSearch):
        raise TypeError('search must be an InitialGridSearch')
    settings = _coverage_settings(threshold, min_samples, neighbor_cells, max_unvisited_fraction)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be an integer >= 1')

    first = search.axis_names[0]
    if first == 'logage' and search.model_kwargs.get('linear_age', True):
        first = 'age'
    names = (first, 'eep', 'feh_init', 'distance')
    missing = set(names) - set(samples)
    if missing:
        raise ValueError(f'missing posterior latent samples: {sorted(missing)}')
    values = {name: np.asarray(samples[name]) for name in names}
    shape = values[first].shape
    if len(shape) not in (1, 2) or not all(shape):
        raise ValueError('posterior latents must have nonempty (draw,) or (chain, draw) shape')
    if any(v.shape != shape or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('posterior latents must have matching shapes and finite values')
    values = {key: value.reshape((-1, shape[-1])) for key, value in values.items()}
    model = _default_model(search.physical_model)

    def score(physical_model, point):
        return log_density(model, (search.observations, physical_model), search.model_kwargs, point)[0]

    evaluate = jax.jit(jax.vmap(score, in_axes=(None, 0)))
    flat = {key: value.ravel() for key, value in values.items()}
    size = flat[first].size
    batch_size = min(batch_size, size)
    scores = []
    for start in range(0, size, batch_size):
        count = min(batch_size, size - start)
        batch = {key: jnp.asarray(np.pad(value[start:start + count], (0, batch_size - count),
                                         mode='edge')) for key, value in flat.items()}
        scores.append(np.asarray(evaluate(search.physical_model, batch))[:count])
    posterior_scores = np.concatenate(scores).reshape(values[first].shape)
    if not np.isfinite(posterior_scores).all():
        raise ValueError('posterior has nonfinite joint density under the search target; '
                         'check that samples, model and search options match')

    x = values[first]
    if first == 'age':
        x = np.log10(x) + 9.  # physical Gyr samples -> native log10(year) grid
    edges = tuple(_cell_edges(axis, bounds) for axis, bounds in zip(search.axes, search.bounds))
    counts = np.stack([np.histogram2d(a, b, bins=edges)[0].astype(np.int64)
                       for a, b in zip(x, values['eep'])])
    finite = search.log_density[np.isfinite(search.log_density)]
    if not finite.size:
        raise ValueError('search has no finite grid scores')
    reference = float(max(finite.max(), posterior_scores.max()))
    return recheck_missed_modes(dict(
        **settings, grid_search=search,
        axis_names=search.axis_names, axes=search.axes, edges=edges,
        fixed_values=search.fixed_values.copy(), grid_log_density=search.log_density,
        feh_values=search.feh_values.copy(), feh_step=search.feh_step, best_feh=search.best_feh,
        reference_log_density=reference, posterior_log_density=posterior_scores,
        sample_positions=np.stack((x.ravel(), values['eep'].ravel()), axis=-1),
        counts_by_chain=counts,
        samples_outside_grid=int(size - counts.sum()),
    ))


_UNCHANGED = object()


def recheck_missed_modes(check, *, threshold=None, min_samples=None, neighbor_cells=None,
                          max_unvisited_fraction=_UNCHANGED):
    """Recompute regions/flags from an existing coverage report, without scoring.

    Omitted settings retain their previous values. Explicit None disables the
    unvisited-fraction condition. Returns a new dict; the input is unchanged.
    Search, scores, counts and plotting arrays are shared without copying.
    A different FeH scan, fixed distance, target or posterior needs a new check.
    """
    settings = _coverage_settings(
        check['threshold'] if threshold is None else threshold,
        check['min_samples'] if min_samples is None else min_samples,
        check['neighbor_cells'] if neighbor_cells is None else neighbor_cells,
        check['max_unvisited_fraction'] if max_unvisited_fraction is _UNCHANGED else max_unvisited_fraction,
    )
    report = _region_coverage(check['grid_log_density'], check['reference_log_density'],
                              check['counts_by_chain'], **settings)
    for region in report['regions']:
        region['peak_feh'] = float(check['best_feh'][region['peak_index']])
        point = region['flag_peak_index']
        region['flag_peak_feh'] = None if point is None else float(check['best_feh'][point])
    return {**check, **settings, **report}


def plot_missed_modes(check):
    """Return one Figure from check_missed_modes, with no density re-evaluation.

    White outlines enclose candidate regions. A red cross marks each flagged
    region's highest-scoring unvisited cell for a fraction flag, or its overall
    peak for a count-only flag. The title counts flagged / total regions.
    The color scale saturates above threshold. Track axes are mass/EEP;
    isochrone axes are log10(age/yr)/EEP. All draws are projected onto that plane.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    delta = check['reference_log_density'] - check['grid_log_density']
    positions = check['sample_positions']
    mesh = ax.pcolormesh(*check['edges'], np.ma.masked_invalid(delta).T,
                         vmin=0, vmax=check['threshold'], cmap='viridis_r',
                         shading='flat', rasterized=True)
    ax.scatter(*positions.T, s=2, c='0.75', alpha=.2, rasterized=True)
    candidate = check['labels'] > 0
    if min(candidate.shape) > 1 and candidate.any() and not candidate.all():
        ax.contour(*check['axes'], candidate.T, levels=[.5], colors='white', linewidths=1)
    flagged = [region for region in check['regions'] if region['flag']]
    for region in flagged:
        i, j = region.get('flag_peak_index', region['peak_index'])
        ax.plot(check['axes'][0][i], check['axes'][1][j], 'rx', ms=9, mew=2)
    status = (f"{len(flagged)}/{len(check['regions'])} regions flagged"
              if check['regions'] else 'no high-score grid regions')
    ax.set_title(rf"$\Delta\log p\leq {check['threshold']:g}$" + '\n' + status)
    ax.set_xlabel(r'$M_{\rm initial}/M_\odot$' if check['axis_names'][0] == 'mass'
                  else r'$\log_{10}(\mathrm{age/yr})$')
    ax.set_xlim(check['edges'][0][[0, -1]])
    ax.set_ylim(check['edges'][1][[0, -1]])
    ax.set_ylabel('EEP')
    ax.legend(handles=[
        Line2D([], [], color='0.65', marker='.', ls='', label='NUTS samples'),
        Line2D([], [], color='white', ls='-', label='candidate boundary'),
        Line2D([], [], color='red', marker='x', ls='', label='undersampled region'),
    ], loc='lower right', facecolor='0.3', labelcolor='white', fontsize='small')
    fig.colorbar(mesh, ax=ax, label=r'$\log p_{\rm ref}-\max_{\rm FeH}\log p_{\rm grid}$', extend='max')
    fixed = check['fixed_values']
    if check['feh_step'] is None:
        description = f"Initial FeH={fixed['feh_init']:.3g}"
    else:
        description = f"Max over initial FeH ({check['feh_step']:g} dex scan)"
    fig.suptitle('Potentially missed modes\n' + description
                 + f"; distance={fixed['distance']:.3g} kpc")
    return fig
