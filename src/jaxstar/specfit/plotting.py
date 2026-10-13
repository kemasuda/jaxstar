"""Grid-free diagnostic plots of stored spectral-fit posterior draws."""

import json

import numpy as np

from .results import (
    FitResult, _LATENT_NAMES, _arviz_uses_datatree, _is_inference_data,
    _mcmc_to_inference_data, _validate_posterior,
)

__all__ = ["plot_mcmc_diagnostics"]


def _plot_data(result, regions):
    latent_names = None
    if isinstance(result, FitResult):
        if result.idata is None:
            raise ValueError("no posterior samples were saved")
        latent_names = (result.metadata.get("sampler") or {}).get("latent_names")
        idata = result.idata
    elif _is_inference_data(result):
        idata = result
    else:
        idata = _mcmc_to_inference_data(result, regions=regions)
    _validate_posterior(idata)
    if _LATENT_NAMES in idata.posterior.attrs:
        latent_names = json.loads(idata.posterior.attrs[_LATENT_NAMES])
    if latent_names is None:
        # External InferenceData need not record which outputs were sampled.
        latent_names = list(idata.posterior.data_vars)
    return idata, latent_names


def _dataset(data, group):
    group_data = data[group]
    return group_data.to_dataset() if _arviz_uses_datatree() else group_data


def _plot_trace(data, names, options):
    """Return a Matplotlib Figure using the installed ArviZ plotting API."""
    import arviz as az

    if not _arviz_uses_datatree():
        options = {"legend": True, "divergences": "bottom", "show": False,
                   **options}
        axes = az.plot_trace(data, var_names=names, backend="matplotlib", **options)
        return np.asarray(axes).flat[0].figure

    # 1.x defaults to ECDFs; keep the marginal density shown by the 0.x plot.
    options = {"kind": "kde", **options}
    legend = options.pop("legend", True)
    collection = az.plot_trace_dist(data, var_names=names,
                                    backend="matplotlib", **options)
    if legend:
        collection.add_legend("chain")
    return collection.get_viz("figure")


def _variables(posterior, names, defaults, argument):
    names = defaults if names is None else names
    names = [names] if isinstance(names, str) else list(names)
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{argument} must contain posterior variable names")
    if len(set(names)) != len(names):
        raise ValueError(f"{argument} must contain distinct variable names")
    for name in names:
        if name not in posterior:
            raise KeyError(f"{name!r} is not in the saved posterior")
        if not np.all(np.isfinite(posterior[name].values)):
            raise ValueError(f"{name!r} contains nonfinite posterior values")
    return names


def _corner_ranges(posterior, names, coords):
    """Keep explicitly selected constant outputs plottable, including vectors."""
    posterior = posterior.sel(coords or {})
    ranges = []
    for name in names:
        values = posterior[name].transpose("chain", "draw", ...).values
        samples = values.reshape((values.shape[0] * values.shape[1], -1))
        if samples.shape[0] < 2:
            raise ValueError("corner plots need at least two posterior draws")
        for column in samples.T:
            low, high = float(np.min(column)), float(np.max(column))
            if high == low:
                width = max(abs(low), 1.) * 1e-3
                low, high = low - width, high + width
            ranges.append((low, high))
    return ranges


def plot_mcmc_diagnostics(result, *, trace_vars=None, corner_vars=None,
                          log_vars=(), labels=None, trace_kwargs=None,
                          corner_kwargs=None, regions=None):
    """Return {'trace': Figure, 'corner': Figure} without evaluating a model.

    Accept a completed NumPyro MCMC, InferenceData (ArviZ 0.x), DataTree
    (ArviZ 1.x), or saved FitResult. Both plots default to all sampled sites,
    excluding fixed/derived deterministic outputs when latent-site metadata
    is available. External ArviZ data
    without that metadata defaults to all posterior variables; select names
    explicitly to exclude deterministic outputs. trace_vars and corner_vars
    independently select native site names, including deterministic outputs.

    Trace plots retain separate chains and expand every vector component
    (including region RVs), without ArviZ's default subplot limit. Both plots
    mark saved divergences. regions optionally labels a MCMC's vector rv site;
    SpecFit supplies its labels automatically. Existing ArviZ data retains
    its own coordinates.

    log_vars selects positive variables to display as natural logs in both
    plots, using a copy. Selection always uses native names. labels is a
    native-name-to-display-label mapping; log variables default to ln(name).
    Per-plot kwargs override display settings for arviz.plot_trace (0.x),
    arviz.plot_trace_dist (1.x), and corner.corner. legend is supported in both
    versions. Figures remain open for plt.show() or further editing;
    neither the supplied result nor the fit is modified.
    """
    import arviz as az
    import corner
    import matplotlib.pyplot as plt

    idata, defaults = _plot_data(result, regions)
    trace_names = _variables(idata.posterior, trace_vars, defaults, "trace_vars")
    corner_names = _variables(idata.posterior, corner_vars, defaults, "corner_vars")
    log_names = ([log_vars] if isinstance(log_vars, str) else list(log_vars))
    if log_names:
        log_names = _variables(idata.posterior, log_names, (), "log_vars")
    data = idata.copy()
    for group in ("posterior", "sample_stats"):
        if hasattr(data, group):
            values = _dataset(data, group).transpose("chain", "draw", ...)
            if _arviz_uses_datatree():
                data[group] = values
            else:
                setattr(data, group, values)
    display_labels = {}
    for name in log_names:
        if np.any(data.posterior[name].values <= 0):
            raise ValueError(f"log_vars requires positive samples for {name!r}")
        data.posterior[name] = np.log(data.posterior[name])
        display_labels[name] = f"ln({name})"
    display_labels.update(labels or {})
    labeller = az.labels.MapLabeller(var_name_map=display_labels)
    trace_options = {"compact": False, "combined": False, "labeller": labeller,
                     **dict(trace_kwargs or {})}
    corner_options = {"show_titles": True, "title_fmt": ".2f",
                      "quantiles": (.16, .5, .84), "divergences": True,
                      "labeller": labeller, **dict(corner_kwargs or {})}
    for argument, options, reserved in (
            ("trace_kwargs", trace_options, ("var_names", "filter_vars", "backend", "group")),
            ("corner_kwargs", corner_options, ("var_names", "filter_vars", "group"))):
        if any(name in options for name in reserved):
            raise ValueError(
                f"{argument} cannot set {reserved}; use trace_vars/corner_vars "
                "to select posterior variables (Matplotlib backend)")
    if corner_options.get("range") is None:
        corner_options["range"] = _corner_ranges(
            _dataset(data, "posterior"), corner_names, corner_options.get("coords"))
    figures = {}
    before = set(plt.get_fignums())
    try:
        with az.rc_context({"plot.max_subplots": None}):
            figures["trace"] = _plot_trace(data, trace_names, trace_options)
        figures["trace"].tight_layout()
        figures["corner"] = corner.corner(data, var_names=corner_names,
                                           **corner_options)
    except Exception:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        raise
    return figures
