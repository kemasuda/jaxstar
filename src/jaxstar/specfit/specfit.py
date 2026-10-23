"""Optional orchestration over observations and ordinary NumPyro models.

Inference, continuum and GP mathematics remain in their existing modules.
Mask updates and diagnostics are host-side operations, outside JAX tracing.
"""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import numpy as np

from .broadening import default_broadening
from .model import SpecModel
from .observation import Observation
from .sampling import _C_KMS


def _rows(array):
    return np.atleast_2d(np.asarray(array))


def _positive(value, name, *, allow_zero=False):
    array = np.asarray(value)
    if (array.ndim != 0 or not np.isfinite(array)
            or (array < 0 if allow_zero else array <= 0)):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(array)


def _compute_ccf(wave, flux, template_wave, template_flux, oversample_factor):
    """Frozen jaxspec's mean-subtracted, unweighted wavelength CCF."""
    from scipy.interpolate import interp1d
    from scipy.signal import correlate

    low, high = np.log10(wave.min()) + 1e-4, np.log10(wave.max()) - 1e-4
    if low >= high:
        raise ValueError(
            "CCF wavelength span is too small for the legacy 1e-4 dex edge margins")
    grid = np.logspace(low, high, len(wave) * oversample_factor)
    observed = interp1d(wave, flux)(grid) - np.mean(flux)
    template = interp1d(template_wave, template_flux)(
        grid) - np.mean(template_flux)
    ccf = correlate(observed, template)
    loggrid = np.log(grid)
    velocity = (np.arange(len(ccf)) * np.diff(loggrid)[0]
                - (loggrid[-1] - loggrid[0])) * _C_KMS
    return velocity, ccf


def _plot_ccf_diagnostics(ccf, regions, *, save_path=None, keep_open=True):
    """Plot the computed CCFs and their RV/width summary without recomputing."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for region, velocity, values in zip(regions, ccf["velocities"], ccf["ccfs"]):
        ax.plot(velocity, values / np.max(values),
                alpha=.6, label=f"Region {region}")
    ax.plot(ccf["velocity"], ccf["combined"] / np.max(ccf["combined"]),
            color="black", label="Median CCF")
    ax.axvline(ccf["rv"], color="0.5", ls="--")
    ax.set(xlim=(ccf["velocity"][0], ccf["velocity"][-1]),
           xlabel="Velocity [km/s]", ylabel="Normalized CCF")
    ax.set_title(f"Median RV = {ccf['rv']:.3f} km/s\n"
                 f"CCF FWHM = {ccf['width']:.3f} km/s", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.), borderaxespad=0)
    fig.tight_layout()
    try:
        if save_path is not None:
            fig.savefig(save_path)
    finally:
        if not keep_open:
            plt.close(fig)


def _extend_mask(flag, factor):
    """Extend each original True run by ceil(factor * run_length) on both sides."""
    result = flag.copy()
    changes = np.diff(np.r_[False, flag, False].astype(int))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        padding = int(np.ceil(factor * (end - start)))
        result[max(0, start - padding):min(len(flag), end + padding)] = True
    return result


def _plot_outlier_diagnostics(observation, baseline, fit_mask, regions,
                              diagnostics, sigma_threshold, *, save_path=None,
                              keep_open=True):
    """Draw the quantities used for clipping, without recomputing them."""
    from pathlib import Path
    from urllib.parse import quote

    import matplotlib.pyplot as plt

    output = Path(save_path) if save_path is not None else None
    for region, wave, flux, error, fixed, model_flux, flagged, diagnostic in zip(
            regions, _rows(observation.wavelength), _rows(observation.flux),
            _rows(observation.uncertainty), _rows(
                observation.mask), _rows(baseline),
            _rows(fit_mask), diagnostics):
        fig = plt.figure(figsize=(14, 6))
        grid = fig.add_gridspec(3, 2, width_ratios=(3, 1))
        upper = fig.add_subplot(grid[0, 0])
        middle = fig.add_subplot(grid[1, 0], sharex=upper)
        lower = fig.add_subplot(grid[2, 0], sharex=upper)
        histogram = fig.add_subplot(grid[1:, 1])
        upper.tick_params(labelbottom=False)
        middle.tick_params(labelbottom=False)
        usable = ~fixed
        upper.plot(wave[usable], flux[usable], "o", color="gray", mfc="none",
                   mew=.5, ms=3, label="data")
        upper.plot(wave[usable], model_flux[usable],
                   color="C0", lw=1, label="baseline")
        upper.plot(wave[flagged], flux[flagged], "o", color="salmon", ms=3,
                   label="fit mask")
        upper.set(title=f"Region {region}: outlier check", ylabel="Flux")
        middle.set_ylabel("Residual")
        lower.set(xlabel="Wavelength [Å]",
                  ylabel=r"Detrended residual / $\sigma$")
        histogram.set(xlabel=r"Detrended residual / $\sigma$",
                      ylabel="Pixel count")
        upper.legend(fontsize=8)
        if diagnostic is None:
            for ax in (middle, lower, histogram):
                ax.text(.5, .5, "No usable pixels", ha="center", va="center",
                        transform=ax.transAxes)
        else:
            residual, trend, highpass, scatter = diagnostic
            if scatter > 0:
                standardized = highpass / scatter
                limit = sigma_threshold
            else:
                # A zero MAD has no normalized residual; keep spikes visible
                # in flux units with the actual (zero) clipping threshold.
                standardized = highpass
                limit = 0.
                lower.set_ylabel("Detrended residual")
                histogram.set_xlabel("Detrended residual")
                lower.text(.02, .95, "MAD scale = 0; shown in flux units",
                           va="top", transform=lower.transAxes, fontsize=8)
            for ax, values in ((middle, residual), (lower, standardized)):
                ax.plot(wave[usable], values[usable], "o", color="gray", mfc="none",
                        mew=.5, ms=3, label="data")
                ax.plot(wave[flagged], values[flagged], "o", color="salmon", ms=3,
                        label="fit mask")
            trend_line, = middle.plot(
                wave, trend, color="black", lw=1, label="median-filter trend")
            middle.legend(handles=[trend_line], fontsize=8)
            threshold_line = lower.axhline(
                limit, color="C3", ls="--", lw=0.5, label="clipping threshold")
            lower.axhline(-limit, color="C3", ls="--", lw=0.5)
            lower.legend(handles=[threshold_line], fontsize=8)
            bins = np.histogram_bin_edges(standardized[usable], bins=50)
            for selected, color, label in ((usable & ~flagged, "gray", "retained"),
                                           (flagged, "salmon", "fit mask")):
                counts, _ = np.histogram(standardized[selected], bins=bins)
                histogram.stairs(counts, bins, fill=True,
                                 color=color, alpha=.5, label=label)
            if scatter > 0:
                normal_x = np.linspace(-6., 6., 401)
                count_scale = np.count_nonzero(usable) * (bins[1] - bins[0])
                normal_counts = count_scale * \
                    np.exp(-.5 * normal_x**2) / np.sqrt(2 * np.pi)
                histogram.plot(normal_x, normal_counts,
                               color="C0", lw=0.8, label="N(0, 1)")
            histogram.set_yscale("log")
            histogram.set_ylim(bottom=.5)
            histogram.axvline(limit, color="C3", ls="--", lw=0.5)
            histogram.axvline(-limit, color="C3", ls="--", lw=0.5)
            histogram.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.02),
                             ncol=3, handlelength=1.5, columnspacing=1.,
                             borderaxespad=0, frameon=False)
            median_error = np.median(error[usable])
            histogram.text(0, 1.15,
                           f"1.4826 x MAD = {scatter:.4g}\n"
                           f"median(error_obs) = {median_error:.4g}\n"
                           f"ratio = {scatter / median_error:.3f}\n"
                           "(scales in flux units)",
                           ha="left", va="bottom", transform=histogram.transAxes,
                           fontsize=10, linespacing=1.5, in_layout=False)
        lower.set_xlim(wave[0], wave[-1])
        fig.tight_layout()
        try:
            if output is not None:
                suffix = output.suffix or f".{plt.rcParams['savefig.format']}"
                label = quote(str(region), safe="")
                fig.savefig(output.with_name(
                    f"{output.stem}_region{label}{suffix}"))
        finally:
            if not keep_open:
                plt.close(fig)


class SpecFit:
    """Convenience workflow for already associated Observation and SpecModel.

    The constructor accepts matching rows; ``from_grid`` also prepares a model
    from legacy NPZ files. Neither entry point chooses priors. Persistent state
    is the two inputs, setup-time RV bounds, a replaceable Boolean fit mask,
    and the last CCF diagnostics. Samples/results are returned
    to the caller. ``observation.mask`` is never changed. Display labels come
    from ``observation.region``, falling back to ``specmodel.spectra.regions``.

    Inference requires an explicit ``model(observation, specmodel)`` callable.
    Use make_model_single or bind a custom function and settings. Prediction
    delegates to the configured model's callback. No probability model is chosen
    implicitly, and each run uses the current fit mask.
    """

    @classmethod
    def from_grid(cls, gridpath, observation, *, library="coelho", pattern="*.npz",
                  wav_margin=3., pixels=None, velocity_step=None, vmax=50.,
                  wavelength_medium="unknown", flux_kind="as_stored",
                  broadening_operator=None, fixed_ip=None):
        """Prepare a SpecModel from legacy NPZs and associate an Observation.

        Delegates to ``SpecModel.from_grid`` using the observation's wavelengths
        and region labels. Its loading/resampling options have identical
        meanings here; specify exactly one of pixels or velocity_step. Arrays,
        fixed masks and labels in the supplied observation are preserved.
        """
        if not isinstance(observation, Observation):
            raise TypeError("SpecFit.from_grid requires an Observation")
        model = SpecModel.from_grid(
            gridpath, observation.wavelength, regions=observation.region,
            library=library, pattern=pattern, wav_margin=wav_margin,
            pixels=pixels, velocity_step=velocity_step, vmax=vmax,
            wavelength_medium=wavelength_medium, flux_kind=flux_kind,
            broadening_operator=broadening_operator, fixed_ip=fixed_ip)
        return cls(observation, model)

    def __init__(self, observation, specmodel):
        if not isinstance(observation, Observation) or not isinstance(specmodel, SpecModel):
            raise TypeError("SpecFit requires Observation and SpecModel")
        if observation.n_regions != len(specmodel.spectra.regions):
            raise ValueError(
                "Observation and SpecModel must have the same number of regions")
        if observation.region is not None and observation.region != specmodel.spectra.regions:
            raise ValueError(
                "Observation region labels must match the spectral library row order")
        wave, source = _rows(observation.wavelength), np.asarray(
            specmodel.spectra.wavelength)
        if np.any(wave[:, 0] < source[:, 0]) or np.any(wave[:, -1] > source[:, -1]):
            raise ValueError(
                "insufficient model wavelength coverage; prepare wider spectral regions")
        self.observation = observation
        self.specmodel = specmodel
        self._rv_bounds = None
        if specmodel.broadening_operator is default_broadening:
            half = specmodel.velocity_grid.shape[-1] // 2
            safe = source[:, half:-half].astype(np.float64)
            # Positive RV redshifts the safe model interval. Invert its Doppler
            # factors at the observed red/blue endpoints to get lower/upper RV.
            factors = np.column_stack((wave[:, -1] / safe[:, -1],
                                       wave[:, 0] / safe[:, 0]))
            bounds = _C_KMS * (factors - 1) * (factors + 1) / (factors**2 + 1)
            bounds.setflags(write=False)
            self._rv_bounds = bounds
        self.reset_fit_mask()
        self.ccf = None

    @property
    def rv_bounds(self):
        """Read-only (region, 2) lower/upper effective RV bounds in km/s.

        Computed once for the full observation wavelengths, including masked
        pixels, after discarding the default broadening kernel's half-support
        at both edges. Uses SpecModel's relativistic Doppler convention. Bounds
        describe each component/region separately; a shared RV must satisfy all
        rows. Lower > upper means no RV can cover that entire region. Endpoint
        roundoff can affect exact-boundary evaluation, especially in float32.

        Priors are not clipped and evaluation retains SpecModel's coverage
        checks. None for a custom broadening operator, whose safe wavelength
        range may depend on its parameters and is checked during evaluation.
        """
        return self._rv_bounds

    @property
    def regions(self):
        """Region labels inherited from the observation or spectral library."""
        return self.observation.region or self.specmodel.spectra.regions

    @property
    def fit_mask(self):
        """Read-only Boolean array; update through set_fit_mask/reset_fit_mask."""
        return self._fit_mask

    @property
    def effective_mask(self):
        return np.asarray(self.observation.mask) | self.fit_mask

    def mask_summary(self):
        """Return per-region fixed_mask, fit_mask and usable pixel counts.

        The two masks are counted separately and can overlap. Usable pixels
        are outside their union (effective_mask). Return a list of dictionaries
        with region labels; formatting and display are left to the caller.
        """
        return [{"region": label, "fixed_mask": int(np.count_nonzero(fixed)),
                 "fit_mask": int(np.count_nonzero(fitted)),
                 "usable": int(np.count_nonzero(~(fixed | fitted)))}
                for label, fixed, fitted in zip(
                    self.regions, _rows(self.observation.mask), _rows(self.fit_mask))]

    def fit_observation(self):
        """Current mask union with the original arrays, dtype and metadata."""
        return replace(self.observation, mask=self.effective_mask)

    def set_fit_mask(self, mask):
        mask = np.asarray(mask)
        if mask.shape != self.observation.shape:
            raise ValueError(
                f"fit mask must have shape {self.observation.shape}, got {mask.shape}")
        if mask.dtype != np.bool_:
            raise TypeError("fit mask must have Boolean dtype")
        self._fit_mask = mask.copy()
        self._fit_mask.setflags(write=False)

    def reset_fit_mask(self):
        self.set_fit_mask(np.zeros(self.observation.shape, dtype=bool))

    def check_ccf(self, atmosphere=None, *, v_limit=500., ccfvmax=100.,
                  oversample_factor=5, plot=False, save_path=None):
        """Return median region RV and combined half-maximum width, in km/s.

        ``atmosphere=None`` uses the Coelho template teff=5800 K, logg=4.4,
        feh=0 and alpha=0. Other libraries require explicit library-named
        coordinates. An explicit mapping replaces the entire template.
        Coordinates must lie inside the loaded grid; defaults are not clipped.
        Uses the frozen CCF definition, edge margins and median combination.
        Fewer than two usable pixels in a region, missing positive peaks or
        half-maximum crossings fail clearly. No priors are set automatically.
        The diagnostic arrays are available in ``fit.ccf``.

        ``plot=True`` plots peak-normalized region CCFs and the combined median
        CCF in one figure, with a median-RV line and Median RV / CCF FWHM
        annotations in km/s. The displayed range is median RV +/- ccfvmax.
        ``save_path="ccf.pdf"`` saves that figure without a region suffix;
        without an extension, Matplotlib's default save format is used. Saving
        with ``plot=False`` closes the figure afterward. ``plot=True`` leaves it
        open for notebook display or ``plt.show()``. The return value remains
        ``(rv, width)`` in all cases, and stored CCF arrays remain unnormalized.
        """
        from scipy.interpolate import interp1d

        v_limit = _positive(v_limit, "v_limit")
        ccfvmax = _positive(ccfvmax, "ccfvmax")
        if not isinstance(oversample_factor, (int, np.integer)) or oversample_factor < 1:
            raise ValueError("oversample_factor must be a positive integer")
        if atmosphere is None:
            library = self.specmodel.spectra.library
            if library != "coelho":
                raise ValueError("the default CCF template is only available for Coelho; "
                                 f"supply atmosphere explicitly for library {library!r}")
            atmosphere = {"teff": 5800., "logg": 4.4, "feh": 0., "alpha": 0.}
        source = np.asarray(self.specmodel.spectra.wavelength)
        params = {"components": ({"atmosphere": atmosphere},)}
        templates = np.asarray(self.specmodel.intrinsic(params, source))
        velocities, ccfs, rvs = [], [], []
        for wave, flux, mask, swave, template in zip(
                _rows(self.observation.wavelength), _rows(
                    self.observation.flux),
                _rows(self.effective_mask), source, templates):
            if np.count_nonzero(~mask) < 2:
                raise ValueError(
                    "CCF requires at least two usable pixels in every region")
            velocity, ccf = _compute_ccf(
                wave[~mask], flux[~mask], swave, template, oversample_factor)
            allowed = np.abs(velocity) < v_limit
            if not np.any(allowed) or np.max(ccf[allowed]) <= 0 or not np.all(np.isfinite(ccf)):
                raise ValueError(
                    "CCF has no finite positive peak inside v_limit")
            rvs.append(velocity[np.argmax(np.where(allowed, ccf, -np.inf))])
            velocities.append(velocity)
            ccfs.append(ccf)
        rv = float(np.median(rvs))
        common = np.linspace(rv - ccfvmax, rv + ccfvmax, 10000)
        if any(common[0] < velocity[0] or common[-1] > velocity[-1] for velocity in velocities):
            raise ValueError(
                "CCF width window exceeds available lags; reduce ccfvmax")
        combined = np.median([interp1d(velocity, ccf)(common)
                              for velocity, ccf in zip(velocities, ccfs)], axis=0)
        if np.max(combined) <= 0:
            raise ValueError("combined CCF has no positive peak")
        half = combined / np.max(combined) - .5
        crossings = common[1:][half[1:] * half[:-1] < 0]
        if len(crossings) < 2:
            raise ValueError(
                "CCF width needs two half-maximum crossings; adjust ccfvmax")
        width = float(crossings[-1] - crossings[0])
        self.ccf = {"rv": rv, "width": width, "region_rv": np.asarray(rvs),
                    "velocities": tuple(velocities), "ccfs": tuple(ccfs),
                    "velocity": common, "combined": combined}
        if plot or save_path is not None:
            _plot_ccf_diagnostics(self.ccf, self.regions, save_path=save_path,
                                  keep_open=plot)
        return rv, width

    def run_svi(self, *, model, rng_key, step_size=.01, num_steps=1000,
                p_initial=None, progress_bar=True):
        """Return numpyro-inferutils.find_map_svi's native guide-median dict.

        Supply a configured model or a plain callable of (observation, specmodel).
        With p_initial=None, initialization draws from the model's priors.
        """
        from numpyro_inferutils import find_map_svi

        bound = partial(model, self.fit_observation(), self.specmodel)
        return find_map_svi(bound, step_size, num_steps, rng_key=rng_key,
                            p_initial=p_initial, progress_bar=progress_bar)

    def run_nuts(self, *, model, rng_key, num_warmup=500, num_samples=500,
                 num_chains=1, progress_bar=True, nuts_kwargs=None,
                 mcmc_kwargs=None):
        """Run NumPyro NUTS/MCMC with an explicit model; return MCMC.

        The current observation and mask are bound before inference. Model settings
        are carried by the supplied callable.
        """
        from numpyro.infer import MCMC, NUTS

        controls = dict(mcmc_kwargs or {})
        if controls.get("jit_model_args", False):
            raise ValueError(
                "SpecFit binds its observation; use jit_model_args=False, including for GP")
        bound = partial(model, self.fit_observation(), self.specmodel)
        mcmc = MCMC(NUTS(bound, **(nuts_kwargs or {})), num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains,
                    progress_bar=progress_bar, **controls)
        mcmc.run(rng_key)
        return mcmc

    def predict(self, *, point=None, samples=None, model, max_draws=50):
        """Predict at a native point or average predictions over posterior draws.

        Supply exactly one of point or samples. All physical/continuum/GP
        calculations delegate to model.predict using the current fit mask.
        A point returns the model's full prediction dict. The built-in model
        returns model_flux = physical_flux * continuum and mean_flux equal to
        model_flux plus the GP conditional mean, when available. With vecchia1,
        mean_flux equals model_flux, gp_mean is omitted, and a point also returns
        normal_residual_variance = diag(C) for mixture masking. Samples must have
        a leading, flattened draw axis, as in MCMC.get_samples(). At most max_draws
        evenly spaced draws are used; None uses all draws. Sample predictions
        return the mean curves (physical_flux, continuum, model_flux,
        mean_flux and gp_mean when supplied by the model) and draw_indices.
        When model_flux is supplied, model_flux_quantiles and
        gp_mean_quantiles contain pointwise 16th/84th percentiles of
        model_flux and mean_flux - model_flux, respectively, with shape
        (2, *observation.shape), using the same draws. The difference is taken
        within each draw. These describe the spread of conditional means across
        draws, excluding within-draw continuum/GP uncertainty and fresh noise.
        Parameter dictionaries and conditional covariance objects are not
        averaged. These are conditional-mean predictions, not predictive draws.
        """
        from .reconstruction import _mean_prediction

        if (point is None) == (samples is None):
            raise ValueError("supply exactly one of point or samples")
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise TypeError(
                "prediction requires a configured model with a predict method")
        observation = self.fit_observation()

        def predict_point(values):
            return predict(values, observation, self.specmodel)

        if point is not None:
            return predict_point(point)
        return _mean_prediction(predict_point, samples, max_draws=max_draws)

    def _prediction(self, prediction, mask):
        array = np.asarray(prediction)
        if array.shape != self.observation.shape:
            raise ValueError(
                f"prediction must have shape {self.observation.shape}")
        if not np.all(np.isfinite(array[~np.asarray(mask)])):
            raise ValueError("prediction must be finite on usable pixels")
        return array

    def mask_outliers(self, *, point, model, sigma_threshold=5.,
                      extend_outlier_mask=True, extension_factor=1., mask_v=None,
                      plot=False, save_path=None, method="gaussian_mixture", p_outlier=.5,
                      mask_velocity=10.):
        """Replace the fit mask; default to Gaussian-mixture classification.

        ``method="gaussian_mixture"`` fits one global contamination fraction
        and extra absolute flux scale to data minus model_flux, without
        subtracting the GP mean or median filtering. The narrow variance is
        diag(K + observation variance + jitter variance) from the fitted normal
        model. Pixels with mismatch probability > p_outlier (default .5) are
        flagged, then extended. sigma_threshold and mask_v
        apply only to sigma_clip. The inference likelihood is never changed.
        For mixture masking, mask_velocity (default 10.) sets a fixed half-width
        in km/s around each original flagged pixel instead of extension_factor,
        using c * abs(log(wavelength / flagged_wavelength)) <= mask_velocity
        within each region. Zero adds no neighbors; None keeps the usual rule.
        extend_outlier_mask=False disables either kind of extension.
        After a mixture call, ``outlier_diagnostics`` contains
        mismatch_probability (NaN on fixed masks), mismatch_fraction,
        mismatch_scale, and normal_residual_variance from that masking pass.
        These diagnostics are a snapshot, not recomputed when the mask changes.
        Mixture plots use the same layout, with physical residuals in the middle,
        probabilities and their threshold at the bottom, and a histogram of
        residual/sqrt(V) on the right. Final extended flags are highlighted.

        ``method="sigma_clip"`` selects the legacy median-filter/MAD rule below.
        Predict once at ``point`` using ``model`` and the current fit mask,
        before updating the mask. The baseline is prediction['model_flux'],
        excluding GP mean. ``mask_v`` is in km/s; absent an override it is the
        largest predicted component vsini, matching the old shared filter-width
        choice. All pixels outside the fixed mask are then reconsidered for
        clipping, so previously flagged pixels can return. Fixed masked pixels
        are excluded from filtering, MAD and new flags.

        ``plot=True`` creates a figure per region with three panels on the left:
        data/baseline, residual/median-filter trend, and detrended residual
        divided by its MAD scale with +/-sigma_threshold lines. On the right,
        a histogram of the same detrended quantity separates retained and masked
        pixels and shows the thresholds and a standard-normal reference, scaled
        by the before-clipping pixel count and bin width. It reports 1.4826 * MAD,
        median observed uncertainty and their ratio, all evaluated outside the
        fixed mask before clipping. The
        final (possibly extended) mask is highlighted. For zero MAD, the last
        panel and histogram use flux units instead, without the normal reference.
        ``save_path="outliers.pdf"`` saves outliers_region8.pdf, etc., using
        plot_models' region suffix and default-format conventions. Saving also
        works with ``plot=False`` and closes each figure afterward; ``plot=True``
        leaves figures open for notebook display or ``plt.show()``. Plotting
        defaults to False; the return value is always the updated fit mask.
        """
        if method not in ("sigma_clip", "gaussian_mixture"):
            raise ValueError(
                "method must be 'sigma_clip' or 'gaussian_mixture'")
        if method == "gaussian_mixture":
            from numpyro import handlers
            from ._outliers import (
                _normal_residual_variance, _fit_gaussian_mixture, _plot_mixture_outliers,
            )

            threshold = np.asarray(p_outlier)
            if threshold.ndim != 0 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("p_outlier must be finite in [0, 1]")
            factor = _positive(
                extension_factor, "extension_factor", allow_zero=True)
            if mask_velocity is not None:
                mask_velocity = _positive(
                    mask_velocity, "mask_velocity", allow_zero=True)
            # Capture the already-replayed fitted noise sites; no new samples.
            with handlers.trace() as trace:
                prediction = self.predict(point=point, model=model)
            if not isinstance(prediction, Mapping) or "model_flux" not in prediction:
                raise ValueError(
                    "model prediction must contain model_flux")
            fixed = np.asarray(self.observation.mask)
            baseline = self._prediction(prediction["model_flux"], fixed)
            variance = self._prediction(
                _normal_residual_variance(self.observation, prediction, model, trace), fixed)
            if np.any(variance[~fixed] <= 0):
                raise ValueError(
                    "normal_residual_variance must be positive on usable pixels")
            probability = np.full(self.observation.shape, np.nan)
            fitted = dict(mismatch_fraction=None,
                          mismatch_scale=None, log_likelihood=0.)
            if np.any(~fixed):
                fitted, probability[~fixed] = _fit_gaussian_mixture(
                    self.observation.flux[~fixed] - baseline[~fixed], variance[~fixed])
            flags = probability > float(threshold)
            if extend_outlier_mask:
                if mask_velocity is None:
                    flags = np.stack([_extend_mask(row, factor)
                                     for row in _rows(flags)]).reshape(fixed.shape)
                else:
                    for wave, row in zip(_rows(self.observation.wavelength), _rows(flags)):
                        log_wave = np.log(np.asarray(wave, dtype=float))
                        # Original flags only; no recursive extension.
                        centers = log_wave[row]
                        left = np.searchsorted(
                            log_wave, centers - mask_velocity / _C_KMS)
                        right = np.searchsorted(
                            log_wave, centers + mask_velocity / _C_KMS, side="right")
                        for start, end in zip(left, right):
                            row[start:end] = True
                flags &= ~fixed
            diagnostics = dict(fitted, method=method, p_outlier=float(threshold),
                               mismatch_probability=probability,
                               normal_residual_variance=variance)
            self.set_fit_mask(flags)
            self.outlier_diagnostics = diagnostics
            if plot or save_path is not None:
                _plot_mixture_outliers(self.observation, baseline, self.fit_mask,
                                       self.regions, diagnostics, save_path=save_path, keep_open=plot)
            return self.fit_mask

        from scipy.stats import median_abs_deviation

        threshold = _positive(sigma_threshold, "sigma_threshold")
        factor = _positive(
            extension_factor, "extension_factor", allow_zero=True)
        make_plots = plot or save_path is not None
        prediction = self.predict(point=point, model=model)
        if not isinstance(prediction, Mapping) or "model_flux" not in prediction:
            raise ValueError(
                "model prediction must return a mapping containing model_flux")
        fixed = np.asarray(self.observation.mask)
        baseline = self._prediction(prediction["model_flux"], fixed)
        if mask_v is None:
            if "params" not in prediction:
                raise ValueError(
                    "supply mask_v when model prediction does not include physical params")
            mask_v = max(float(np.max(component["broadening"]["vsini"]))
                         for component in prediction["params"]["components"])
        velocities = np.asarray(mask_v)
        if velocities.ndim == 0:
            velocities = np.full(self.observation.n_regions, velocities)
        if velocities.shape != (self.observation.n_regions,) or not np.all(np.isfinite(velocities) & (velocities >= 0)):
            raise ValueError(
                "mask_v must be finite nonnegative scalar or one value per region")
        flags = np.zeros((self.observation.n_regions,
                         self.observation.n_pixels), dtype=bool)
        diagnostics = []
        for i, (wave, flux, clip, model_flux, velocity) in enumerate(zip(
                _rows(self.observation.wavelength), _rows(
                    self.observation.flux),
                _rows(fixed), _rows(baseline), velocities)):
            usable = ~clip
            if not np.any(usable):
                if make_plots:
                    diagnostics.append(None)
                continue
            if len(wave) < 2:
                raise ValueError(
                    "outlier filtering requires at least two pixels")
            kernel = int(np.median(wave) * velocity * 2 /
                         3e5 / np.median(np.diff(wave))) * 4 + 1
            kernel = min(kernel, len(wave) if len(wave) % 2 else len(wave) - 1)
            residual = np.full(wave.shape, np.nan)
            residual[usable] = flux[usable] - model_flux[usable]
            # Zero edge padding matches scipy.signal.medfilt. Fixed masked
            # NaN/Inf cannot contaminate its neighbors' local median.
            windows = np.lib.stride_tricks.sliding_window_view(
                np.pad(residual, kernel // 2, constant_values=0), kernel)
            trend = np.full(wave.shape, np.nan)
            trend[usable] = np.nanmedian(windows[usable], axis=-1)
            highpass = residual - trend
            scatter = 1.4826 * median_abs_deviation(highpass[usable])
            flags[i, usable] = np.abs(highpass[usable]) > threshold * scatter
            if extend_outlier_mask:
                flags[i] = _extend_mask(flags[i], factor) & usable
            if make_plots:
                diagnostics.append((residual, trend, highpass, scatter))
        self.set_fit_mask(flags.reshape(self.observation.shape))
        if make_plots:
            _plot_outlier_diagnostics(self.observation, baseline, self.fit_mask,
                                      self.regions, diagnostics, threshold,
                                      save_path=save_path, keep_open=plot)
        return self.fit_mask

    def residual_diagnostics(self, *, point, model, baseline="mean_flux"):
        """Per-region RMS, MAD scale, standardized RMS/counts and lag-1 correlation.

        Predict once at ``point`` using ``model`` and the current effective mask.
        Residuals are observation.flux minus the selected prediction field:
        ``mean_flux`` includes the GP mean; ``model_flux`` excludes it.
        Jitter comes from the prediction (zero if omitted by a custom model),
        in additive flux units, scalar or one value per region. Standardization
        uses sqrt(uncertainty**2 + jitter**2), without GP whitening.

        Only currently usable pixels contribute. Lag-1 pairs are adjacent in
        the original wavelength array; masked gaps are never bridged.
        Empty/undefined statistics are NaN and threshold counts are zero.
        Return a list of dictionaries; formatting/display are left to the caller.
        """
        from scipy.stats import median_abs_deviation

        if baseline not in ("mean_flux", "model_flux"):
            raise ValueError(
                "baseline must be 'mean_flux' or 'model_flux'")
        prediction = self.predict(point=point, model=model)
        if not isinstance(prediction, Mapping) or baseline not in prediction:
            raise ValueError(
                f"model prediction must return a mapping containing {baseline}")
        mask = self.effective_mask
        model_flux = self._prediction(prediction[baseline], mask)
        jitter = np.asarray(prediction.get("jitter", 0.))
        if jitter.ndim == 0:
            jitter = np.full(self.observation.n_regions, jitter)
        if jitter.shape != (self.observation.n_regions,) or not np.all(np.isfinite(jitter) & (jitter >= 0)):
            raise ValueError(
                "jitter must be finite nonnegative scalar or one value per region")
        stats = []
        for label, flux, error, excluded, predicted, extra in zip(
                self.regions, _rows(self.observation.flux), _rows(
                    self.observation.uncertainty),
                _rows(mask), _rows(model_flux), jitter):
            valid = ~excluded
            residual = flux[valid] - predicted[valid]
            standardized = residual / np.hypot(error[valid], extra)
            pairs = valid[:-1] & valid[1:]
            left = flux[:-1][pairs] - predicted[:-1][pairs]
            right = flux[1:][pairs] - predicted[1:][pairs]
            correlation = (float(np.corrcoef(left, right)[0, 1])
                           if len(left) >= 2 and np.std(left) > 0 and np.std(right) > 0 else np.nan)
            stats.append({"region": label, "n_usable": len(residual),
                          "rms": float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan,
                          "mad_scale": float(1.4826 * median_abs_deviation(residual)) if len(residual) else np.nan,
                          "standardized_rms": float(np.sqrt(np.mean(standardized**2))) if len(residual) else np.nan,
                          "n_gt_3sigma": int(np.count_nonzero(np.abs(standardized) > 3)),
                          "n_gt_5sigma": int(np.count_nonzero(np.abs(standardized) > 5)),
                          "n_adjacent_pairs": len(left), "lag1_correlation": correlation})
        return stats

    def plot_models(self, *, point=None, samples=None, model, show_gp=False,
                    show_interval=False, max_draws=50, res_factor=1.5,
                    show_masked=True, save_path=None):
        """Predict and plot continuum-corrected spectra and baseline residuals.

        The upper panel shows data and model_flux (physical model including
        the continuum prediction). The lower shows data - model_flux;
        show_gp adds mean_flux - model_flux there. It controls display only,
        leaving the model's continuum/GP calculations to predict. Vecchia models
        omit the GP overlay, since they reconstruct no latent GP mean. Supply exactly
        one of point or samples; samples uses predict's max_draws convention.
        With samples, show_interval=True adds pointwise 16th-84th percentile
        bands for model_flux in the upper panel and, with show_gp, the GP
        mean in the lower panel. Bands are hidden by default. They
        show the spread of conditional means across draws, excluding within-draw
        continuum/GP uncertainty and fresh noise. The lower panel subtracts the
        mean model_flux; its zero line shares the upper model's C0 color.
        show_masked=False hides fixed- and fit-masked data points in both panels;
        model/GP curves and bands still cover all wavelengths. The default is True.
        Upper limits include unmasked data and all finite displayed model/band values.
        Residual limits are +/- res_factor times the largest absolute usable
        residual or finite displayed GP curve/band value. Each region
        has its own figure with closely spaced, shared-x panels and a common
        vertical legend to their right. Return a list
        of (figure, axes) pairs in region order; each axes array has shape (2,).
        save_path="models.png" saves models_region8.png, etc., even for a single
        region. Region labels are URL-escaped for filenames. If no extension
        is supplied, use Matplotlib's default save format.
        """
        from pathlib import Path
        from urllib.parse import quote

        import matplotlib.pyplot as plt

        res_factor = _positive(res_factor, "res_factor")
        prediction = self.predict(
            point=point, samples=samples, model=model, max_draws=max_draws)
        if getattr(model, "settings", {}).get("gp_solver") == "vecchia1":
            show_gp = False
        baseline = self._prediction(
            prediction["model_flux"], self.effective_mask)
        interval = (np.stack([_rows(self._prediction(bound, self.effective_mask))
                              for bound in prediction["model_flux_quantiles"]])
                    if samples is not None and show_interval else None)
        gp_mean = (self._prediction(prediction["mean_flux"], self.effective_mask) - baseline
                   if show_gp else None)
        gp_interval = (np.stack([_rows(self._prediction(bound, self.effective_mask))
                                 for bound in prediction["gp_mean_quantiles"]])
                       if samples is not None and show_gp and show_interval else None)
        plots = []
        output = Path(save_path) if save_path is not None else None
        for i, (wave, flux, fixed, fitted, model_flux) in enumerate(zip(
                _rows(self.observation.wavelength), _rows(
                    self.observation.flux),
                _rows(self.observation.mask), _rows(self.fit_mask), _rows(baseline))):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 5),
                                     gridspec_kw={"height_ratios": (7, 3)})
            upper, lower = axes
            for select, style, color, label in ((~(fixed | fitted), "o", "gray", "data"),
                                                (fixed, "x", "lightgray",
                                                 "fixed mask"),
                                                (fitted & ~fixed, "o", "salmon", "fit mask")):
                if not np.any(select) or (not show_masked and label != "data"):
                    continue
                marker_options = dict(
                    mfc="none", mew=.5) if label == "data" else {}
                upper.plot(wave[select], flux[select], style, color=color, ms=3,
                           label=label, **marker_options)
                lower.plot(wave[select], (flux - model_flux)[select], style,
                           color=color, ms=3, **marker_options)
            upper.plot(wave, model_flux, color="C0", ls="-", lw=1,
                       label="model\n(incl. continuum)")
            if interval is not None:
                upper.fill_between(wave, interval[0, i], interval[1, i],
                                   color="C0", alpha=.2, linewidth=0,
                                   label="model 16–84%\n(across draws)")
            if show_gp:
                lower.plot(wave, _rows(gp_mean)[
                           i], color="C1", lw=1, label="GP mean")
                if gp_interval is not None:
                    lower.fill_between(wave, gp_interval[0, i], gp_interval[1, i],
                                       color="C1", alpha=.2, linewidth=0,
                                       label="GP 16–84%\n(across draws)")
            valid = ~(fixed | fitted)
            shown = np.concatenate(
                (flux[valid], model_flux[np.isfinite(model_flux)]))
            if interval is not None:
                bounds = interval[:, i]
                shown = np.concatenate((shown, bounds[np.isfinite(bounds)]))
            if shown.size:
                low, high = np.min(shown), np.max(shown)
                margin = .05 * (high - low if high >
                                low else max(abs(high), 1.))
                upper.set_ylim(low - margin, high + margin)
            if np.any(valid):
                maximum = np.max(np.abs(flux[valid] - model_flux[valid]))
                if maximum == 0:
                    maximum = np.median(
                        _rows(self.observation.uncertainty)[i, valid])
                if show_gp:
                    gp_shown = _rows(gp_mean)[i]
                    if gp_interval is not None:
                        gp_shown = np.concatenate(
                            (gp_shown, gp_interval[:, i].ravel()))
                    gp_shown = gp_shown[np.isfinite(gp_shown)]
                    if gp_shown.size:
                        maximum = max(maximum, np.max(np.abs(gp_shown)))
                lower.set_ylim(-res_factor * maximum, res_factor * maximum)
            lower.axhline(0, color="C0", lw=.5)
            upper.set_ylabel("Normalized flux")
            upper.set_title(f"Region {self.regions[i]}")
            lower.set(xlabel="Wavelength [Å]", ylabel="Residual")
            lower.set_xlim(wave[0], wave[-1])
            fig.tight_layout(rect=(0, 0, .81, .92), h_pad=0.2)
            fig.legend(loc="center left", bbox_to_anchor=(.81, .5),
                       ncol=1)
            if output is not None:
                suffix = output.suffix or f".{plt.rcParams['savefig.format']}"
                region = quote(str(self.regions[i]), safe="")
                fig.savefig(output.with_name(
                    f"{output.stem}_region{region}{suffix}"))
            plots.append((fig, axes))
        return plots

    def plot_mcmc_diagnostics(self, result, **kwargs):
        """Plot all sampled sites by default; return trace and corner Figures.

        Accept MCMC, InferenceData, DataTree or FitResult. See
        :func:`jaxstar.specfit.plot_mcmc_diagnostics` for variable selection,
        log transforms, labels and per-plot options. Only region labels are
        taken from this fitter; no model evaluation or prediction is needed.
        """
        from .plotting import plot_mcmc_diagnostics

        return plot_mcmc_diagnostics(result, regions=self.regions, **kwargs)
