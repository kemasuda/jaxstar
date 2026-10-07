"""Diagnostic plots and evolutionary tracks evaluated with an explicit model."""

__all__ = ["plot_trace", "plot_corner", "plot_track", "plot_posterior_tracks", "plot_diagnostics"]

from pathlib import Path

import numpy as np

from .results import to_inference_data, get_run_metadata, _quantity_info
from .model import MistModel, _MAGNITUDES


def _keys(idata, keys, *, trace=False):
    available = tuple(idata.posterior.data_vars)
    if keys is None:
        record = get_run_metadata(idata)
        if trace:
            candidates = record.get("latent_names", ())
        else:
            observed = record.get("model", {}).get("observations", {}).get("keys", ())
            candidates = (*observed, "feh_init", "mass", "radius", "age", "eep")
        keys = tuple(dict.fromkeys(k for k in candidates if k in available))
        if not keys:
            keys = tuple(k for k in available if idata.posterior[k].dims == ("chain", "draw"))
    else:
        keys = (keys,) if isinstance(keys, str) else tuple(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("keys must contain distinct posterior variable names")
    for key in keys:
        if key not in available:
            raise KeyError(f"{key!r} was not saved; available outputs: {available}. Request it with output_keys when sampling.")
        data = idata.posterior[key]
        if data.dims != ("chain", "draw"):
            raise ValueError(f"plotting requires scalar outputs; {key!r} has dimensions {data.dims}")
        if not np.all(np.isfinite(data.values)):
            raise ValueError(f"{key!r} contains nonfinite posterior values")
    return tuple(keys)


def _labels(idata, keys, labels):
    supplied = {} if labels is None else dict(labels)
    return {k: supplied.get(k, idata.posterior[k].attrs.get("label", _quantity_info(k)["label"])) for k in keys}


def _references(idata):
    obs = get_run_metadata(idata).get("model", {}).get("observations", {})
    # Exact key matching: surface FeH is never drawn as initial FeH, nor current
    # mass as initial mass, nor apparent magnitude as absolute magnitude.
    return dict(zip(obs.get("keys", ()), obs.get("values", ())))


def _save(fig, path, overwrite):
    if path is not None:
        path = Path(path).expanduser()
        if path.exists() and not overwrite:
            raise FileExistsError(f"figure exists: {path}; use overwrite=True")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_trace(result, *, keys=None, labels=None, show_observations=True,
               save_path=None, overwrite=False, **kwargs):
    """Return a Matplotlib Figure with separate-chain traces and marginals.

    Defaults to sampled latent sites; divergence marks use saved sample_stats.
    Accepts MCMC or reloaded InferenceData. MultiNest samples have no MCMC trace.
    Additional kwargs are forwarded to arviz.plot_trace (Matplotlib backend).
    """
    import arviz as az
    idata = to_inference_data(result)
    if get_run_metadata(idata).get("sampler") == "multinest":
        raise ValueError("MultiNest posterior draws have no MCMC chain trace; use plot_corner")
    keys = _keys(idata, keys, trace=True)
    if "var_names" in kwargs or "backend" in kwargs:
        raise ValueError("use keys for variable selection; plots use the Matplotlib backend")
    options = {"compact": False, "combined": False, "legend": True,
               "divergences": "bottom", "show": False,
               "labeller": az.labels.MapLabeller(var_name_map=_labels(idata, keys, labels))}
    options.update(kwargs)
    axes = np.asarray(az.plot_trace(idata, var_names=list(keys), backend="matplotlib", **options))
    fig = axes.flat[0].figure
    if show_observations:
        reference = _references(idata)
        for i, key in enumerate(keys):
            if key in reference:
                axes[i, 0].axvline(reference[key], color="C3", ls="--", lw=1)
                axes[i, 1].axhline(reference[key], color="C3", ls="--", lw=1)
    fig.tight_layout()
    return _save(fig, save_path, overwrite)


def plot_corner(result, *, keys=None, labels=None, show_observations=True,
                save_path=None, overwrite=False, **kwargs):
    """Return a corner Figure, combining chains only for this correlation plot.

    Defaults to observed outputs plus feh_init, mass, radius, age and eep. A
    missing requested key is an error, not a request to re-evaluate a grid.
    Titles use short keys with the median and 16th/84th percentiles; full quantity
    labels are on the axes. Constant variables get a finite plotting range.
    Other kwargs go to corner.
    """
    try:
        import corner
    except ImportError as exc:
        raise ImportError('Corner plots require corner; install jaxstar[mistfit-diagnostics]') from exc
    idata = to_inference_data(result)
    keys = _keys(idata, keys)
    if "weights" in kwargs:
        raise ValueError("these results contain equal-weight draws; weights must not be applied again")
    samples = np.column_stack([idata.posterior[k].values.reshape(-1) for k in keys])
    if len(samples) < 2:
        raise ValueError("corner plots need at least two posterior draws")
    reference = _references(idata) if show_observations else {}
    display = _labels(idata, keys, labels)
    # Corner rejects zero-width intervals. Fixed outputs are valid saved data.
    ranges = []
    for values in samples.T:
        lo, hi = np.min(values), np.max(values)
        if hi == lo:
            width = max(abs(float(lo)), 1.) * 1e-3
            lo, hi = lo - width, hi + width
        ranges.append((lo, hi))
    options = {"labels": [display[k] for k in keys], "titles": list(keys),
               "truths": [reference.get(k) for k in keys],
               "show_titles": True, "title_fmt": ".3g", "quantiles": (.16, .5, .84),
               "range": ranges, "label_kwargs": {"fontsize": 10}, "title_kwargs": {"fontsize": 10}}
    options.update(kwargs)
    fig = corner.corner(samples, **options)
    return _save(fig, save_path, overwrite)


def _magnitude(key):
    return key.removesuffix("_abs") in _MAGNITUDES


def _observed_bands(idata):
    observed = get_run_metadata(idata).get("model", {}).get("observations", {})
    return tuple(key for key in observed.get("keys", ()) if _magnitude(key))


def _range(value, name):
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)) or value[0] >= value[1]:
        raise ValueError(f"{name} must be a finite increasing (lower, upper) pair")
    return value


def _track_curves(model, params, keys, eep_range, age_range, n_points):
    """Evaluate a batch of tracks, including native knots and exact endpoints.

    Age limits invert the model's piecewise-linear logage curve for drawing
    only. They do not alter the model's derivative field or inference prior.
    """
    if not isinstance(model, MistModel) or model.kind != "track":
        raise ValueError("track plots require MistModel.from_tracks(...)")
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    if any(not isinstance(k, str) or k not in model.output_keys for k in keys):
        raise ValueError("x and y must be available MistModel output keys")
    eep_range, age_range = _range(eep_range, "eep_range"), _range(age_range, "age_range")
    if eep_range is not None and age_range is not None:
        raise ValueError("specify only one of eep_range or age_range")
    for key, axis in (("mass_init", "initial_mass"), ("feh_init", "initial_feh")):
        bounds = np.asarray(model.grid.axis(axis))
        if not np.all(np.isfinite(params[key]) & (params[key] >= bounds[0]) & (params[key] <= bounds[-1])):
            raise ValueError(f"{key} must lie inside the track grid")
    if "distance" in params and not np.all(np.isfinite(params["distance"]) & (params["distance"] > 0)):
        raise ValueError("distance must be finite and positive (kpc)")
    native = np.asarray(model.grid.axis("eep"))
    if eep_range is None:
        eep_range = native[[0, -1]]
    elif eep_range[0] < native[0] or eep_range[1] > native[-1]:
        raise ValueError("eep_range must lie inside the track grid")
    count = len(params["mass_init"])
    low, high = np.full(count, eep_range[0]), np.full(count, eep_range[1])
    coordinates = {key: values[:, None] for key, values in params.items()}
    if age_range is not None:
        if age_range[0] < 0:
            raise ValueError("age_range must be nonnegative (Gyr)")
        curves = np.asarray(model({**coordinates, "eep": native[None, :]}, keys="logage")["logage"])
        log_limits = (np.log10(age_range[0])+9 if age_range[0] > 0 else -np.inf,
                      np.log10(age_range[1])+9)
        limits = np.array([np.interp(log_limits, curve, native) for curve in curves])
        low, high = limits.T
    active = low < high
    if not np.any(active):
        raise ValueError("the requested age range does not overlap these tracks")
    # Fixed-size batches avoid recompiling interpolation for each track. Native
    # knots outside the interval duplicate its endpoints, removed when drawing.
    eeps = np.sort(np.concatenate((np.linspace(low, high, n_points).T,
                                  np.clip(native[None, :], low[:, None], high[:, None])), axis=1), axis=1)
    values = model({**coordinates, "eep": eeps}, keys=keys)
    values = {key: np.asarray(value) for key, value in values.items()}
    if any(not np.all(np.isfinite(value[active])) for value in values.values()):
        raise ValueError("nonfinite track predictions in the requested range")
    return values, eeps, active


def _track_axes(ax, x, y, labels):
    labels = {} if labels is None else labels
    ax.set_xlabel(labels.get(x, _quantity_info(x)["label"]))
    ax.set_ylabel(labels.get(y, _quantity_info(y)["label"]))
    # HR-diagram convention; set rather than toggle for overlaid tracks.
    if x in ("teff", "logt", "logTeff") or _magnitude(x):
        ax.xaxis.set_inverted(True)
    if _magnitude(y):
        ax.yaxis.set_inverted(True)


def plot_track(model, mass_init, feh_init, *, distance=None, x="teff", y="kmag_abs",
               eep_range=None, age_range=None, n_points=400, ax=None, labels=None,
               save_path=None, overwrite=False, **line_kwargs):
    """Return a Figure showing one track at fixed initial mass and initial FeH.

    x/y are scalar MistModel output keys. The default is absolute Ks vs Teff;
    apparent magnitudes require distance in kpc. Specify either eep_range or
    age_range (Gyr); otherwise show the full grid EEP extent. An age interval
    is intersected with the available track, independently of inference priors.
    n_points samples EEP, with native knots also included. Supply ax to overlay
    tracks. Additional kwargs go to ax.plot. Teff and magnitude axes use the
    HR-diagram direction. No observations or sampler results are required.
    """
    import matplotlib.pyplot as plt
    params = {"mass_init": mass_init, "feh_init": feh_init}
    if distance is not None:
        params["distance"] = distance
    if any(np.ndim(value) != 0 for value in params.values()):
        raise ValueError("mass_init, feh_init and distance must be scalars")
    params = {key: np.asarray([value], dtype=float) for key, value in params.items()}
    values, eeps, _ = _track_curves(model, params, tuple(dict.fromkeys((x, y))),
                                   eep_range, age_range, n_points)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    keep = np.r_[True, np.diff(eeps[0]) > 0]
    ax.plot(values[x][0, keep], values[y][0, keep], **line_kwargs)
    _track_axes(ax, x, y, labels)
    return _save(ax.figure, save_path, overwrite)


def _track_observation(ax, x, y, observations):
    observed = dict(zip(observations.get("keys", ()),
                        zip(observations.get("values", ()), observations.get("errors", ()))))
    if x in observed and y in observed:
        xv, xe = observed[x]
        yv, ye = observed[y]
        ax.errorbar(xv, yv, xerr=xe, yerr=ye, fmt="*", ms=10, color="C3",
                    ecolor="C3", capsize=3, zorder=5, label="Observed")
    elif x in observed:
        value, error = observed[x]
        ax.axvspan(value-error, value+error, color="C3", alpha=.12)
        ax.axvline(value, color="C3", ls="--", lw=1, label="Observed")
    elif y in observed:
        value, error = observed[y]
        ax.axhspan(value-error, value+error, color="C3", alpha=.12)
        ax.axhline(value, color="C3", ls="--", lw=1, label="Observed")


def plot_posterior_tracks(result, model, *, x="teff", y=None, n_samples=20, seed=0,
                          eep_range=None, age_range=None, n_points=400, labels=None,
                          show_observations=True, save_path=None, overwrite=False):
    """Return a Figure with posterior tracks and sampled locations in each panel.

    y is one key or a sequence; None selects all observed magnitude keys in
    recorded order. x defaults to Teff. Uniformly choose up to n_samples joint
    draws without replacement across chains, using a local RNG (seed=0 by
    default). The same draws are used for every panel. Each draw's initial
    mass/FeH and distance define a curve; its EEP defines the marked position.
    Outputs are recomputed using the supplied track model, which must match
    the fitted grid. Thus axes need not have been saved as posterior outputs.

    eep_range/age_range follow plot_track. Locations outside the drawing range
    are not marked; tracks with no age overlap are omitted without resampling.
    Observations/errors come from the result, with exact key matching. If only
    one axis is observed, show its value and error as a line and shaded band.
    Apparent and absolute magnitudes are never substituted for one another.
    """
    import matplotlib.pyplot as plt
    idata = to_inference_data(result)
    record = get_run_metadata(idata)
    if record.get("model", {}).get("physical_model", {}).get("kind") == "isochrone":
        raise ValueError("posterior tracks require a track-model result")
    ys = _observed_bands(idata) if y is None else ((y,) if isinstance(y, str) else tuple(y))
    if not ys or any(not isinstance(key, str) for key in ys) or len(set(ys)) != len(ys):
        raise ValueError("specify distinct y keys, or provide results with observed magnitudes")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    mass_key = "mass_init" if "mass_init" in idata.posterior else "mass"
    required = [mass_key, "feh_init", "eep"]
    if "distance" in idata.posterior:
        required.append("distance")
    _keys(idata, required)
    size = idata.posterior["eep"].size
    if size == 0:
        raise ValueError("posterior tracks need at least one draw")
    indices = np.random.default_rng(seed).choice(size, min(n_samples, size), replace=False)
    selected = {key: np.asarray(idata.posterior[key]).reshape(-1)[indices] for key in required}
    selected["mass_init"] = selected.pop(mass_key)
    sample_eep = selected.pop("eep")
    keys = tuple(dict.fromkeys((x, *ys)))
    curves, eeps, active = _track_curves(model, selected, keys, eep_range, age_range, n_points)
    positions = {key: np.asarray(value) for key, value in model({**selected, "eep": sample_eep}, keys=keys).items()}
    if any(not np.all(np.isfinite(value)) for value in positions.values()):
        raise ValueError("posterior locations lie outside the supplied track model")
    marked = active & (sample_eep >= eeps[:, 0]) & (sample_eep <= eeps[:, -1])
    columns = min(3, len(ys))
    fig, axes = plt.subplots((len(ys)+columns-1)//columns, columns,
                             figsize=(5*columns, 4.3*((len(ys)+columns-1)//columns)), squeeze=False)
    try:
        for ax, key in zip(axes.flat, ys):
            for number, i in enumerate(np.flatnonzero(active)):
                keep = np.r_[True, np.diff(eeps[i]) > 0]
                ax.plot(curves[x][i, keep], curves[key][i, keep], color="C0", alpha=.25, lw=1,
                        label="Posterior tracks" if number == 0 else None)
            if np.any(marked):
                ax.plot(positions[x][marked], positions[key][marked], "o", color="C0", ms=4,
                        alpha=.8, zorder=4, label="Posterior locations")
            if show_observations:
                _track_observation(ax, x, key, record.get("model", {}).get("observations", {}))
            _track_axes(ax, x, key, labels)
            ax.set_title(key)
            ax.legend(fontsize="small")
        for ax in list(axes.flat)[len(ys):]:
            fig.delaxes(ax)
        if not np.all(active):
            fig.suptitle(f"{active.sum()}/{len(active)} selected tracks overlap the age range")
        fig.tight_layout()
        return _save(fig, save_path, overwrite)
    except Exception:
        plt.close(fig)
        raise


def plot_diagnostics(result, *, keys=None, trace_kwargs=None, corner_kwargs=None,
                     model=None, track_kwargs=None, show_tracks=True,
                     output_dir=None, overwrite=False):
    """Return diagnostic Figures: trace, corner, and optionally posterior tracks.

    MultiNest omits trace. Trace/corner need no grid. With a track model and
    observed magnitudes (or explicit track_kwargs['y']), add 'tracks', one
    panel per observed band. MistFit supplies its model automatically. Use
    show_tracks=False for grid-free diagnostics. Per-plot dictionaries override
    plotting options; keys selects only trace/corner variables. No fitter state
    is changed and no files are written unless output_dir is given.
    """
    import matplotlib.pyplot as plt
    idata = to_inference_data(result)
    methods = {"corner": (plot_corner, {"keys": keys, **dict(corner_kwargs or {})})}
    if get_run_metadata(idata).get("sampler") != "multinest":
        methods = {"trace": (plot_trace, {"keys": keys, **dict(trace_kwargs or {})}), **methods}
    options = dict(track_kwargs or {})
    if show_tracks:
        if options and (model is None or model.kind != "track"):
            raise ValueError("track_kwargs requires a track model; pass model=...")
        if model is not None and model.kind == "track" and (options.get("y") is not None or _observed_bands(idata)):
            methods["tracks"] = (plot_posterior_tracks, {"model": model, **options})
    directory = Path(output_dir).expanduser() if output_dir is not None else None
    if directory is not None and not overwrite:
        for name in methods:
            if (directory / f"{name}.png").exists():
                raise FileExistsError(f"figure exists: {directory / (name + '.png')}; use overwrite=True")
    figures = {}
    try:
        for name, (method, options) in methods.items():
            if "save_path" in options or "overwrite" in options:
                raise ValueError("use output_dir and overwrite on plot_diagnostics to save figures")
            figures[name] = method(idata, **options)
        if directory is not None:
            for name, figure in figures.items():
                _save(figure, directory / f"{name}.png", overwrite)
    except Exception:
        for figure in figures.values():
            plt.close(figure)
        raise
    return figures
