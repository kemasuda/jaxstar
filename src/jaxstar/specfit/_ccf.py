"""Paired post-fit CCFs on the original observed pixels, without gap filling."""

from collections.abc import Mapping

import numpy as np

from .reconstruction import _posterior_draws
from .sampling import _C_KMS


def _region_ccfs(wave, data, model, error, source, template, rv, velocity, weighting):
    """Weighted, mean-subtracted cross products with one shifted template."""
    shifts = rv + velocity
    if not np.all(np.isfinite(shifts) & (np.abs(shifts) < _C_KMS)):
        raise ValueError("CCF trial RVs must be finite with abs(rv) < c")
    log_shifts = np.arctanh(shifts / _C_KMS)
    # Center log wavelengths before interpolation to preserve small shifts.
    log_source = np.log(source / source[0])
    log_wave = np.log(wave / source[0])
    if (log_wave[0] - log_shifts.max() < log_source[0]
            or log_wave[-1] - log_shifts.min() > log_source[-1]):
        raise ValueError("insufficient template wavelength coverage for CCF shifts; "
                         "reduce ccfvmax or prepare wider spectral regions")
    weights = np.ones_like(data, dtype=float)
    if weighting == "inverse_variance":
        weights = (np.min(error) / error)**2
    weights /= weights.sum()
    centered = np.stack((data, model))
    centered = centered - centered[:, :1]
    centered = (centered - (centered @ weights)[:, None]) * weights
    ccfs = np.empty((2, len(velocity)))
    for i, shift in enumerate(log_shifts):
        shifted = np.interp(log_wave - shift, log_source, template)
        shifted -= shifted[0]
        shifted -= weights @ shifted
        ccfs[:, i] = centered @ shifted
    return ccfs


def ccf_diagnostics(fit, *, point, samples, model, max_draws, ccfvmax,
                    velocity_step, weighting):
    """Compute paired curves per draw, then average before common scaling."""
    if (point is None) == (samples is None):
        raise ValueError("supply exactly one of point or samples")
    if weighting not in ("uniform", "inverse_variance"):
        raise ValueError("weighting must be 'uniform' or 'inverse_variance'")
    count = int(np.ceil(ccfvmax / velocity_step))
    velocity = np.linspace(-ccfvmax, ccfvmax, 2 * count + 1)
    if samples is None:
        indices = None
        points = (point,)
    else:
        samples, indices = _posterior_draws(samples, max_draws=max_draws)
        points = ({name: value[i] for name, value in samples.items()} for i in indices)
    source = np.asarray(fit.specmodel.spectra.wavelength, dtype=float)
    waves, fluxes, errors, masks = (np.atleast_2d(np.asarray(value)) for value in (
        fit.observation.wavelength, fit.observation.flux, fit.observation.uncertainty,
        fit.effective_mask))
    if np.any(np.sum(~masks, axis=1) < 2):
        raise ValueError("CCF comparison requires at least two usable pixels in every region")
    sums = np.zeros((len(fit.regions), 2, len(velocity)))
    draws = 1 if indices is None else len(indices)
    for values in points:
        prediction = fit.predict(point=values, model=model)
        required = ("params", "continuum", "model_flux")
        if not isinstance(prediction, Mapping) or any(key not in prediction for key in required):
            raise ValueError("CCF comparison requires prediction fields params, continuum and model_flux")
        params = prediction["params"]
        components = params.get("components", ()) if isinstance(params, Mapping) else ()
        if len(components) != 1:
            raise ValueError("CCF comparison requires one stellar component")
        rv = np.asarray(components[0].get("rv", np.nan), dtype=float)
        if rv.ndim == 0:
            rv = np.full(len(fit.regions), rv)
        if rv.shape != (len(fit.regions),) or not np.all(np.isfinite(rv)):
            raise ValueError("CCF comparison requires finite scalar or per-region RV")
        continuum = np.atleast_2d(fit._prediction(prediction["continuum"], fit.effective_mask))
        baseline = np.atleast_2d(fit._prediction(prediction["model_flux"], fit.effective_mask))
        if np.any(continuum[~masks] <= 0):
            raise ValueError("CCF continuum must be positive on usable pixels")
        templates = np.asarray(fit.specmodel.intrinsic(params, source), dtype=float)
        for i, label in enumerate(fit.regions):
            valid = ~masks[i]
            correction = continuum[i, valid]
            try:
                ccfs = _region_ccfs(
                    waves[i, valid], fluxes[i, valid] / correction,
                    baseline[i, valid] / correction, errors[i, valid] / correction,
                    source[i], templates[i], rv[i], velocity, weighting)
            except ValueError as exc:
                raise ValueError(f"Region {label}: {exc}") from exc
            sums[i] += ccfs / draws
    results = []
    for label, pair, mask in zip(fit.regions, sums, masks):
        scale = np.max(pair[1])
        if not np.all(np.isfinite(pair)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Region {label}: model CCF must have a finite positive peak; "
                             "check the template, fitted spectrum and velocity window")
        data_ccf, model_ccf = pair / scale
        results.append(dict(region=label, velocity=velocity.copy(), data_ccf=data_ccf,
                            model_ccf=model_ccf, residual=data_ccf - model_ccf,
                            normalization=float(scale), n_pixels=int(np.sum(~mask)),
                            draw_indices=None if indices is None else indices.copy()))
    return results


def plot_ccf(diagnostics, *, save_path):
    """Draw region CCFs and data-minus-model residuals on a shared scale."""
    from pathlib import Path
    from urllib.parse import quote

    import matplotlib.pyplot as plt

    output = Path(save_path) if save_path is not None else None
    plots = []
    try:
        for result in diagnostics:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5),
                                     gridspec_kw={"height_ratios": (7, 3)})
            plots.append((fig, axes))
            upper, lower = axes
            velocity = result["velocity"]
            upper.plot(velocity, result["data_ccf"], color="gray", label="data")
            upper.plot(velocity, result["model_ccf"], color="C0", label="model")
            lower.plot(velocity, result["residual"], color="gray")
            lower.axhline(0, color="C0", lw=.5)
            for ax in axes:
                ax.axvline(0, color="0.7", ls=":", lw=.7)
            upper.set(title=f"Region {result['region']}: post-fit CCF", ylabel="Scaled CCF")
            upper.legend()
            lower.set(xlabel="RV offset from fitted RV [km/s]", ylabel="Data − model",
                      xlim=(velocity[0], velocity[-1]))
            fig.tight_layout(h_pad=.2)
            if output is not None:
                suffix = output.suffix or f".{plt.rcParams['savefig.format']}"
                label = quote(str(result["region"]), safe="")
                fig.savefig(output.with_name(f"{output.stem}_region{label}{suffix}"))
    except Exception:
        for fig, _ in plots:
            plt.close(fig)
        raise
    return plots
