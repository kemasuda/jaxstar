"""Host-side contamination-mixture fitting and plots, used only for masking."""

import numpy as np


def _max_broadening_fwhm(specmodel, params):
    """Largest kernel FWHM over component/region pairs with nonzero light."""
    try:
        components = specmodel._components(params)
        weights, _, _, _ = specmodel._fractions(
            params, len(components), specmodel.spectra.wavelength.dtype)
        active = np.asarray(weights) > 0
        # Entirely dark components need not even have measurable kernels.
        selected = np.any(active, axis=1)
        kernel_params = {
            "components": tuple(c for c, keep in zip(components, selected) if keep),
            "instrument": params.get("instrument", {}),
        }
        widths = np.asarray(specmodel.broadening_fwhm(kernel_params))
        active = active[selected]
        if widths.shape != active.shape:
            raise ValueError("kernel FWHM must have shape (component, region)")
        widths = widths[active]
        if widths.size == 0 or not np.all(np.isfinite(widths) & (widths >= 0)):
            raise ValueError("contributing kernel FWHMs must be finite and nonnegative")
        return float(np.max(widths))
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"cannot determine automatic mask_velocity: {exc}; "
                         "supply an explicit mask_velocity") from exc


def _normal_residual_variance(observation, prediction, model, trace):
    """Read the fitted normal covariance only when mixture masking is requested."""
    from .continuum import _region_jitter
    from .likelihood import _gp_setup, _region_gp
    from .numpyro_model import model_single

    if "normal_residual_variance" in prediction:
        return np.asarray(prediction["normal_residual_variance"])
    if getattr(model, "function", None) is not model_single:
        raise ValueError("custom masking models must predict normal_residual_variance = diag(C)")
    jitter = trace["jitter"]["value"]
    gp = {name: trace[name]["value"] for name in ("gp_amplitude", "gp_scale") if name in trace}
    variance = np.full(observation.shape, np.nan)
    if not gp:
        jitters = np.asarray(_region_jitter(jitter, observation.n_regions))
        jitters = jitters[0] if observation.ndim == 1 else jitters[:, None]
        error = np.where(observation.valid, observation.uncertainty, 1.)
        return np.where(observation.valid, np.hypot(error, jitters)**2, np.nan)
    prepared, _, amplitude, scale, jitters, dtype, solver = _gp_setup(
        observation, prediction["model_flux"], gp["gp_amplitude"], gp["gp_scale"],
        jitter, model.settings["gp_solver"])
    for region, indices in enumerate(prepared.indices):
        if indices:
            normal_gp, _, _, _ = _region_gp(
                observation, region, indices, amplitude, scale, jitters[region], dtype, solver)
            # variance already includes kernel, observation noise, and jitter.
            np.atleast_2d(variance)[region, list(indices)] = np.asarray(normal_gp.variance)
    return variance


def _fit_gaussian_mixture(residual, variance):
    """Fit one global fraction and extra flux scale to independent residuals.

    Normalize flux units for optimization; the returned scale is absolute.
    Finite bounds on transformed parameters only protect numerical limits.
    Prefer the no-contamination limit when the mixture cannot improve on it.
    """
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp

    residual, variance = np.asarray(residual, float), np.asarray(variance, float)
    if (residual.ndim != 1 or residual.size == 0 or variance.shape != residual.shape
            or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(variance) & (variance > 0))):
        raise ValueError("mixture fitting requires finite residuals and positive variances")
    unit = np.median(np.sqrt(variance))
    r, v = residual / unit, variance / unit**2
    narrow = -.5 * (np.log(2 * np.pi * v) + r**2 / v)

    def terms(parameters):
        eta, log_scale = parameters
        extra = np.exp(2 * log_scale)
        broad_variance = v + extra
        broad = -.5 * (np.log(2 * np.pi * broad_variance) + r**2 / broad_variance)
        weights = np.stack((narrow - np.logaddexp(0., eta),
                            broad - np.logaddexp(0., -eta)))
        logp = logsumexp(weights, axis=0)
        probability = np.exp(weights[1] - logp)
        return logp, probability, broad_variance, extra

    def objective(parameters):
        logp, probability, broad_variance, extra = terms(parameters)
        gradient = [-np.mean(probability - expit(parameters[0])),
                    -np.mean(probability * extra / broad_variance
                             * (r**2 / broad_variance - 1))]
        return -np.mean(logp), np.asarray(gradient)

    max_scale = max(1., np.max(np.abs(r)))
    bounds = [(-20., 20.), (-20., np.log(max_scale) + 2.)]
    # Multiple starts reduce sensitivity to the collapsed-mixture boundary.
    candidates = []
    for fraction in (.01, .1, .5):
        for scale in (1., max_scale / 2.):
            result = minimize(
                objective, [np.log(fraction / (1 - fraction)), np.log(scale)],
                jac=True, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            if result.success and np.isfinite(result.fun):
                candidates.append(result)
    if not candidates:
        raise RuntimeError("Gaussian-mixture masking optimization did not converge; mask unchanged")
    best = min(candidates, key=lambda result: result.fun)
    collapsed = best.fun >= -np.mean(narrow) - 1e-10
    parameters = np.array([-20., -20.]) if collapsed else best.x
    logp, probability, _, _ = terms(parameters)
    return {
        "mismatch_fraction": float(expit(parameters[0])),
        "mismatch_scale": float(unit * np.exp(parameters[1])),
        "log_likelihood": float(np.sum(logp) - residual.size * np.log(unit)),
        "collapsed": bool(collapsed),
    }, probability


def _mixture_residual_densities(x, variance, fraction, mismatch):
    """Weighted component densities in residual/sqrt(V).

    All usable pixels contribute, independently of their final mask labels.
    The outlier width is pixel-dependent because mismatch is in flux units.
    """
    normal = (1. - fraction) * np.exp(-.5 * x**2) / np.sqrt(2 * np.pi)
    widths = np.hypot(1., mismatch / np.sqrt(variance))
    outlier = np.zeros_like(x, dtype=float)
    # Bound temporary storage when plotting long spectral regions.
    for start in range(0, widths.size, 2048):
        width = widths[start:start + 2048]
        outlier += (np.exp(-.5 * (x[:, None] / width)**2) / width).sum(axis=1)
    outlier *= fraction / (widths.size * np.sqrt(2 * np.pi))
    return normal, outlier


def _plot_mixture_outliers(observation, baseline, fit_mask, regions, diagnostics,
                           *, save_path=None, keep_open=True):
    """Match the legacy layout, with physical residuals and outlier probability."""
    from pathlib import Path
    from urllib.parse import quote

    import matplotlib.pyplot as plt

    output = Path(save_path) if save_path is not None else None
    velocity_source = diagnostics["mask_velocity_source"]
    if velocity_source in ("auto", "explicit"):
        padding_label = (f"mask padding: ±{diagnostics['mask_velocity']:.2g} km/s "
                         f"({velocity_source})")
    elif velocity_source == "run_length":
        padding_label = "mask padding: run-length extension"
    else:
        padding_label = "mask padding: disabled"
    arrays = [np.atleast_2d(value) for value in (
        observation.wavelength, observation.flux, observation.mask, baseline,
        fit_mask, diagnostics["normal_residual_variance"], diagnostics["mismatch_probability"])]
    for region, wave, flux, fixed, model_flux, flags, variance, probability in zip(regions, *arrays):
        fig = plt.figure(figsize=(14, 6))
        grid = fig.add_gridspec(3, 2, width_ratios=(3, 1))
        upper = fig.add_subplot(grid[0, 0])
        middle = fig.add_subplot(grid[1, 0], sharex=upper)
        lower = fig.add_subplot(grid[2, 0], sharex=upper)
        histogram = fig.add_subplot(grid[1:, 1])
        upper.tick_params(labelbottom=False)
        middle.tick_params(labelbottom=False)
        valid = ~fixed
        residual = np.full(wave.shape, np.nan)
        residual[valid] = flux[valid] - model_flux[valid]
        scale = np.sqrt(np.where(valid, variance, np.nan))
        for ax, values in ((upper, flux), (middle, residual), (lower, probability)):
            ax.plot(wave, np.where(valid, values, np.nan), "o", color="gray",
                    mfc="none", mew=.5, ms=3, label="data")
            ax.plot(wave[flags], values[flags], "o", color="salmon", ms=3, label="fit mask")
        upper.plot(wave, model_flux, color="C0", lw=1,
                    label="physical spectrum × continuum")
        upper.set(title=f"Region {region}: Gaussian-mixture outlier check\n{padding_label}",
                  ylabel="Flux")
        upper.legend(fontsize=8)
        middle.axhline(0., color="C0", lw=.5)
        normal_band = middle.fill_between(wave, -scale, scale, color="C0", alpha=.15,
                                          label=r"normal residual scale $\sqrt{V}$")
        middle.set_ylabel("Physical residual")
        middle.legend(handles=[normal_band], fontsize=8)
        threshold_line = lower.axhline(
            diagnostics["p_outlier"], color="C3", ls="--", lw=.8,
            label=f"p_outlier = {diagnostics['p_outlier']:g}")
        lower.set(xlabel="Wavelength [Å]", ylabel="Outlier probability", ylim=(-.03, 1.03))
        lower.set_xlim(wave[0], wave[-1])
        lower.legend(handles=[threshold_line], fontsize=8)
        histogram.set(xlabel=r"Physical residual / $\sqrt{V}$", ylabel="Pixel count")
        if np.any(valid):
            standardized = residual / scale
            bins = np.histogram_bin_edges(standardized[valid], bins=50)
            bottom = np.zeros(len(bins) - 1, dtype=int)
            for selected, color, label in ((valid & ~flags, "gray", "retained"),
                                           (flags, "salmon", "fit mask")):
                counts, _ = np.histogram(standardized[selected], bins=bins)
                histogram.stairs(bottom + counts, bins, baseline=bottom,
                                 fill=True, color=color, alpha=.5, label=label)
                bottom = bottom + counts
            fraction, mismatch = (diagnostics[name] for name in ("mismatch_fraction", "mismatch_scale"))
            if fraction is not None and mismatch is not None:
                curve_x = np.unique(np.concatenate((
                    np.linspace(-6., 6., 401),
                    np.linspace(min(-6., bins[0]), max(6., bins[-1]), 601))))
                normal, outlier = _mixture_residual_densities(
                    curve_x, variance[valid], fraction, mismatch)
                count_scale = np.count_nonzero(valid) * (bins[1] - bins[0])
                normal *= count_scale
                outlier *= count_scale
                # Keep the normal reference's original +/-6 range, while
                # following the broad component across the observed tails.
                normal_range = np.abs(curve_x) <= 6.
                for x, counts, color, style, label in (
                        (curve_x[normal_range], normal[normal_range], "C0", "--", "normal component"),
                        (curve_x, outlier, "C3", "-.", "outlier component"),
                        (curve_x, normal + outlier, "black", "-", "total mixture")):
                    histogram.plot(x, counts, color=color, ls=style, lw=.8, label=label)
            histogram.set_yscale("log")
            histogram.set_ylim(bottom=.5)
            histogram.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02),
                             ncol=2, handlelength=1.5, columnspacing=1., borderaxespad=0, frameon=False)
            fraction = "not supplied" if fraction is None else f"{fraction:.4g}"
            mismatch = "not supplied" if mismatch is None else f"{mismatch:.4g}"
            histogram.text(0, 1.22,
                           f"f_mismatch = {fraction}\n"
                           f"s_mismatch = {mismatch}\n"
                           f"median(sqrt(V)) = {np.median(scale[valid]):.4g}\n"
                           "(scales in flux units)",
                           ha="left", va="bottom", transform=histogram.transAxes,
                           fontsize=10, linespacing=1.5, in_layout=False)
        else:
            for ax in (middle, lower, histogram):
                ax.text(.5, .5, "No usable pixels", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        try:
            if output is not None:
                suffix = output.suffix or f".{plt.rcParams['savefig.format']}"
                label = quote(str(region), safe="")
                fig.savefig(output.with_name(f"{output.stem}_region{label}{suffix}"))
        finally:
            if not keep_open:
                plt.close(fig)
