"""Diagnostic plots from native results or portable InferenceData."""

__all__ = ["plot_trace", "plot_corner", "plot_diagnostics"]

from pathlib import Path

import numpy as np

from .results import to_inference_data, get_run_metadata, _quantity_info


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


def plot_diagnostics(result, *, keys=None, trace_kwargs=None, corner_kwargs=None,
                     output_dir=None, overwrite=False):
    """Return {'trace': Figure, 'corner': Figure}, or only corner for MultiNest.

    Convert once, then draw/save without accessing a grid. Per-plot dictionaries
    can override keys and plotting options. No fitter state is changed and no
    files are written unless output_dir is given.
    """
    import matplotlib.pyplot as plt
    idata = to_inference_data(result)
    methods = {"corner": (plot_corner, dict(corner_kwargs or {}))}
    if get_run_metadata(idata).get("sampler") != "multinest":
        methods = {"trace": (plot_trace, dict(trace_kwargs or {})), **methods}
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
            figures[name] = method(idata, **{"keys": keys, **options})
        if directory is not None:
            for name, figure in figures.items():
                _save(figure, directory / f"{name}.png", overwrite)
    except Exception:
        for figure in figures.values():
            plt.close(figure)
        raise
    return figures
