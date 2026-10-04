"""Optional orchestration over observations and ordinary NumPyro models.

Inference, continuum and GP mathematics remain in their existing modules.
Mask updates and diagnostics are host-side operations, outside JAX tracing.
"""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import numpy as np

from .continuum import chebyshev_basis, continuum_posterior, evaluate_continuum
from .likelihood import gp_continuum_posterior, gp_conditional_mean
from .model import SpecModel
from .numpyro_model import model_single
from .observation import Observation, _identifiers
from .sampling import _C_KMS


def _rows(array):
    return np.atleast_2d(np.asarray(array))


def _positive(value, name, *, allow_zero=False):
    array = np.asarray(value)
    if (array.ndim != 0 or not np.isfinite(array)
            or (array < 0 if allow_zero else array <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(array)


def _compute_ccf(wave, flux, template_wave, template_flux, oversample_factor):
    """Frozen jaxspec's mean-subtracted, unweighted wavelength CCF."""
    from scipy.interpolate import interp1d
    from scipy.signal import correlate

    low, high = np.log10(wave.min()) + 1e-4, np.log10(wave.max()) - 1e-4
    if low >= high:
        raise ValueError("CCF wavelength span is too small for the legacy 1e-4 dex edge margins")
    grid = np.logspace(low, high, len(wave) * oversample_factor)
    observed = interp1d(wave, flux)(grid) - np.mean(flux)
    template = interp1d(template_wave, template_flux)(grid) - np.mean(template_flux)
    ccf = correlate(observed, template)
    loggrid = np.log(grid)
    velocity = (np.arange(len(ccf)) * np.diff(loggrid)[0]
                - (loggrid[-1] - loggrid[0])) * _C_KMS
    return velocity, ccf


def _extend_mask(flag, factor):
    """Extend each original True run by ceil(factor * run_length) on both sides."""
    result = flag.copy()
    changes = np.diff(np.r_[False, flag, False].astype(int))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        padding = int(np.ceil(factor * (end - start)))
        result[max(0, start - padding):min(len(flag), end + padding)] = True
    return result


class SpecFit:
    """Convenience workflow for already associated Observation and SpecModel.

    Rows must already match; this object neither selects libraries nor chooses
    priors. Persistent state is the two inputs, display labels, a replaceable
    Boolean fit mask, and the last CCF diagnostics. Samples/results are returned
    to the caller. ``observation.mask`` is never changed.

    Inference accepts ordinary ``model(observation, specmodel, **model_kwargs)``
    callables, defaulting to model_single. Direct low-level usage remains valid.
    """

    def __init__(self, observation, specmodel, orders=None):
        if not isinstance(observation, Observation) or not isinstance(specmodel, SpecModel):
            raise TypeError("SpecFit requires Observation and SpecModel")
        if observation.n_regions != len(specmodel.spectra.regions):
            raise ValueError("Observation and SpecModel must have the same number of regions")
        if observation.region is not None and observation.region != specmodel.spectra.regions:
            raise ValueError("Observation region labels must match the spectral library row order")
        wave, source = _rows(observation.wavelength), np.asarray(specmodel.spectra.wavelength)
        if np.any(wave[:, 0] < source[:, 0]) or np.any(wave[:, -1] > source[:, -1]):
            raise ValueError("insufficient model wavelength coverage; prepare wider spectral regions")
        labels = (observation.order or observation.region or specmodel.spectra.regions) if orders is None else orders
        labels = _identifiers(labels, "orders", observation.n_regions)
        self.observation = observation
        self.specmodel = specmodel
        self.orders = labels
        self.reset_fit_mask()
        self.ccf = None

    @property
    def fit_mask(self):
        """Read-only Boolean array; update through set_fit_mask/reset_fit_mask."""
        return self._fit_mask

    @property
    def effective_mask(self):
        return np.asarray(self.observation.mask) | self.fit_mask

    def fit_observation(self):
        """Current mask union with the original arrays, dtype and metadata."""
        return replace(self.observation, mask=self.effective_mask)

    def set_fit_mask(self, mask):
        mask = np.asarray(mask)
        if mask.shape != self.observation.shape:
            raise ValueError(f"fit mask must have shape {self.observation.shape}, got {mask.shape}")
        if mask.dtype != np.bool_:
            raise TypeError("fit mask must have Boolean dtype")
        self._fit_mask = mask.copy()
        self._fit_mask.setflags(write=False)

    def reset_fit_mask(self):
        self.set_fit_mask(np.zeros(self.observation.shape, dtype=bool))

    def check_ccf(self, atmosphere, *, v_limit=500., ccfvmax=100., oversample_factor=5):
        """Return median region RV and combined half-maximum width, in km/s.

        Supply the template's library-named atmosphere coordinates explicitly.
        Uses the frozen CCF definition, edge margins and median combination.
        Fewer than two usable pixels in a region, missing positive peaks or
        half-maximum crossings fail clearly. No priors are set automatically.
        The diagnostic arrays are available in ``fit.ccf``.
        """
        from scipy.interpolate import interp1d

        v_limit = _positive(v_limit, "v_limit")
        ccfvmax = _positive(ccfvmax, "ccfvmax")
        if not isinstance(oversample_factor, (int, np.integer)) or oversample_factor < 1:
            raise ValueError("oversample_factor must be a positive integer")
        source = np.asarray(self.specmodel.spectra.wavelength)
        params = {"components": ({"atmosphere": atmosphere},)}
        templates = np.asarray(self.specmodel.intrinsic(params, source))
        velocities, ccfs, rvs = [], [], []
        for wave, flux, mask, swave, template in zip(
                _rows(self.observation.wavelength), _rows(self.observation.flux),
                _rows(self.effective_mask), source, templates):
            if np.count_nonzero(~mask) < 2:
                raise ValueError("CCF requires at least two usable pixels in every region")
            velocity, ccf = _compute_ccf(wave[~mask], flux[~mask], swave, template, oversample_factor)
            allowed = np.abs(velocity) < v_limit
            if not np.any(allowed) or np.max(ccf[allowed]) <= 0 or not np.all(np.isfinite(ccf)):
                raise ValueError("CCF has no finite positive peak inside v_limit")
            rvs.append(velocity[np.argmax(np.where(allowed, ccf, -np.inf))])
            velocities.append(velocity)
            ccfs.append(ccf)
        rv = float(np.median(rvs))
        common = np.linspace(rv - ccfvmax, rv + ccfvmax, 10000)
        if any(common[0] < velocity[0] or common[-1] > velocity[-1] for velocity in velocities):
            raise ValueError("CCF width window exceeds available lags; reduce ccfvmax")
        combined = np.median([interp1d(velocity, ccf)(common)
                              for velocity, ccf in zip(velocities, ccfs)], axis=0)
        if np.max(combined) <= 0:
            raise ValueError("combined CCF has no positive peak")
        half = combined / np.max(combined) - .5
        crossings = common[1:][half[1:] * half[:-1] < 0]
        if len(crossings) < 2:
            raise ValueError("CCF width needs two half-maximum crossings; adjust ccfvmax")
        width = float(crossings[-1] - crossings[0])
        self.ccf = {"rv": rv, "width": width, "region_rv": np.asarray(rvs),
                    "velocities": tuple(velocities), "ccfs": tuple(ccfs),
                    "velocity": common, "combined": combined}
        return rv, width

    def run_svi(self, *, rng_key, model_kwargs, step_size=.01, num_steps=1000,
                p_initial=None, progress_bar=True, model=model_single):
        """Return numpyro-inferutils.find_map_svi's native guide-median dict."""
        from numpyro_inferutils import find_map_svi

        bound = partial(model, self.fit_observation(), self.specmodel, **model_kwargs)
        return find_map_svi(bound, step_size, num_steps, rng_key=rng_key,
                            p_initial=p_initial, progress_bar=progress_bar)

    def run_nuts(self, *, rng_key, model_kwargs, num_warmup=300, num_samples=400,
                 num_chains=1, progress_bar=True, nuts_kwargs=None,
                 mcmc_kwargs=None, model=model_single):
        """Run ordinary NumPyro NUTS/MCMC with a concrete current mask; return MCMC."""
        from numpyro.infer import MCMC, NUTS

        controls = dict(mcmc_kwargs or {})
        if controls.get("jit_model_args", False):
            raise ValueError("SpecFit binds its observation; use jit_model_args=False, including for GP")
        bound = partial(model, self.fit_observation(), self.specmodel, **model_kwargs)
        mcmc = MCMC(NUTS(bound, **(nuts_kwargs or {})), num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains,
                    progress_bar=progress_bar, **controls)
        mcmc.run(rng_key)
        return mcmc

    def reconstruct(self, point, *, model_kwargs, model=model_single):
        """Reconstruct one explicit native sample-site dict using the same model.

        All latent sites must be supplied; none are randomly drawn. Fixed and
        derived sites are recomputed by the model, including empirical stellar
        relations. The default model returns SpecModel parameters and records
        sigma_constant/sigma_continuum/jitter, plus paired GP sites if enabled.
        A custom model can use this same small reconstruction convention;
        arbitrary custom inference callables need not implement it.
        """
        from numpyro import handlers

        if not isinstance(point, Mapping):
            raise TypeError("point must be a dict of native sample-site values")

        def substitute(site):
            if site["type"] == "sample" and not site["is_observed"]:
                if site["name"] not in point:
                    raise ValueError(f"point must supply latent site {site['name']!r}")
                return point[site["name"]]
            return None

        obs = self.fit_observation()
        with handlers.trace() as trace, handlers.substitute(substitute_fn=substitute):
            params = model(obs, self.specmodel, **model_kwargs)
        names = ("sigma_constant", "sigma_continuum", "jitter")
        if not isinstance(params, Mapping) or "components" not in params or not all(name in trace for name in names):
            raise ValueError("reconstruction model must return SpecModel params and record "
                             "sigma_constant, sigma_continuum and jitter sites")
        physical = self.specmodel(params, obs.wavelength)
        basis = model_kwargs.get("basis")
        if basis is None:
            basis = chebyshev_basis(obs.wavelength, degree=model_kwargs.get("degree", 4))
        noise = {name: trace[name]["value"] for name in names}
        gp_names = ("gp_amplitude", "gp_scale")
        gp = {name: trace[name]["value"] for name in gp_names if name in trace}
        if gp and len(gp) != 2:
            raise ValueError("reconstruction requires paired gp_amplitude/gp_scale sites")
        gp_options = dict(gp, gp_solver=model_kwargs.get("gp_solver", "auto"))
        post = (gp_continuum_posterior(obs, physical, basis=basis, **noise, **gp_options)
                if gp else continuum_posterior(obs, physical, basis=basis, **noise))
        continuum = evaluate_continuum(basis, post.mean)
        baseline = physical * continuum
        result = {"params": params, "physical_flux": physical, "continuum": continuum,
                  "continuum_posterior": post, "continuum_flux": baseline,
                  "mean_flux": baseline, "jitter": noise["jitter"]}
        if gp:
            result["gp_mean"] = gp_conditional_mean(obs, baseline, jitter=noise["jitter"], **gp_options)
            result["mean_flux"] = baseline + result["gp_mean"]
        return result

    def _prediction(self, prediction, mask):
        array = np.asarray(prediction)
        if array.shape != self.observation.shape:
            raise ValueError(f"prediction must have shape {self.observation.shape}")
        if not np.all(np.isfinite(array[~np.asarray(mask)])):
            raise ValueError("prediction must be finite on usable pixels")
        return array

    def mask_outliers(self, reconstruction=None, *, prediction=None, sigma_threshold=5.,
                      extend_outlier_mask=True, extension_factor=1., mask_v=None):
        """Replace the fit mask using the legacy median-filter/MAD rule.

        Default baseline is reconstruction['continuum_flux'], excluding GP mean.
        Explicit ``prediction=...`` opts into any other baseline. ``mask_v`` is
        in km/s; absent an override it is the largest reconstructed component
        vsini, matching the old shared filter-width choice. The previous fit
        mask is ignored when recomputing; previously flagged pixels can return.
        Fixed masked pixels are excluded from filtering, MAD and new flags.
        """
        from scipy.stats import median_abs_deviation

        threshold = _positive(sigma_threshold, "sigma_threshold")
        factor = _positive(extension_factor, "extension_factor", allow_zero=True)
        if prediction is None:
            if not isinstance(reconstruction, Mapping) or "continuum_flux" not in reconstruction:
                raise ValueError("supply reconstruction with continuum_flux or an explicit prediction")
            prediction = reconstruction["continuum_flux"]
        fixed = np.asarray(self.observation.mask)
        baseline = self._prediction(prediction, fixed)
        if mask_v is None:
            if not isinstance(reconstruction, Mapping) or "params" not in reconstruction:
                raise ValueError("supply mask_v when reconstructed physical params are unavailable")
            mask_v = max(float(np.max(component["broadening"]["vsini"]))
                         for component in reconstruction["params"]["components"])
        velocities = np.asarray(mask_v)
        if velocities.ndim == 0:
            velocities = np.full(self.observation.n_regions, velocities)
        if velocities.shape != (self.observation.n_regions,) or not np.all(np.isfinite(velocities) & (velocities >= 0)):
            raise ValueError("mask_v must be finite nonnegative scalar or one value per region")
        flags = np.zeros((self.observation.n_regions, self.observation.n_pixels), dtype=bool)
        for i, (wave, flux, clip, model_flux, velocity) in enumerate(zip(
                _rows(self.observation.wavelength), _rows(self.observation.flux),
                _rows(fixed), _rows(baseline), velocities)):
            usable = ~clip
            if not np.any(usable):
                continue
            if len(wave) < 2:
                raise ValueError("outlier filtering requires at least two pixels")
            kernel = int(np.median(wave) * velocity * 2 / 3e5 / np.median(np.diff(wave))) * 4 + 1
            kernel = min(kernel, len(wave) if len(wave) % 2 else len(wave) - 1)
            residual = np.full(wave.shape, np.nan)
            residual[usable] = flux[usable] - model_flux[usable]
            # Zero edge padding matches scipy.signal.medfilt. Fixed masked
            # NaN/Inf cannot contaminate its neighbors' local median.
            windows = np.lib.stride_tricks.sliding_window_view(
                np.pad(residual, kernel // 2, constant_values=0), kernel)
            highpass = residual[usable] - np.nanmedian(windows[usable], axis=-1)
            scatter = 1.4826 * median_abs_deviation(highpass)
            flags[i, usable] = np.abs(highpass) > threshold * scatter
            if extend_outlier_mask:
                flags[i] = _extend_mask(flags[i], factor) & usable
        self.set_fit_mask(flags.reshape(self.observation.shape))
        return self.fit_mask

    def residual_diagnostics(self, prediction, *, jitter=0.):
        """Per-region RMS, MAD scale, standardized RMS/counts and lag-1 correlation.

        Only currently usable pixels contribute. Lag-1 pairs are adjacent in
        the original wavelength array; masked gaps are never bridged. Jitter
        is explicit additive flux-unit noise, scalar or one value per region.
        Empty/undefined statistics are NaN and threshold counts are zero.
        """
        from scipy.stats import median_abs_deviation

        mask = self.effective_mask
        prediction = self._prediction(prediction, mask)
        jitter = np.asarray(jitter)
        if jitter.ndim == 0:
            jitter = np.full(self.observation.n_regions, jitter)
        if jitter.shape != (self.observation.n_regions,) or not np.all(np.isfinite(jitter) & (jitter >= 0)):
            raise ValueError("jitter must be finite nonnegative scalar or one value per region")
        stats = []
        for label, flux, error, excluded, baseline, extra in zip(
                self.orders, _rows(self.observation.flux), _rows(self.observation.uncertainty),
                _rows(mask), _rows(prediction), jitter):
            valid = ~excluded
            residual = flux[valid] - baseline[valid]
            standardized = residual / np.hypot(error[valid], extra)
            pairs = valid[:-1] & valid[1:]
            left = flux[:-1][pairs] - baseline[:-1][pairs]
            right = flux[1:][pairs] - baseline[1:][pairs]
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

    def plot_models(self, reconstruction, *, show_physical=False, res_factor=1.5, save_path=None):
        """Plot existing arrays; return figure/axes, with limits based on usable data.

        As in legacy plots, residual limits are +/- res_factor times the largest
        usable residual. Masked extremes do not control the display; callers
        can change the returned axes limits to inspect those excluded values.
        """
        import matplotlib.pyplot as plt

        res_factor = _positive(res_factor, "res_factor")
        baseline = self._prediction(reconstruction["continuum_flux"], self.effective_mask)
        mean = self._prediction(reconstruction["mean_flux"], self.effective_mask)
        physical = (self._prediction(reconstruction["physical_flux"], self.effective_mask)
                    if show_physical else None)
        fig, axes = plt.subplots(self.observation.n_regions * 2, 1,
                                 figsize=(10, 4 * self.observation.n_regions), squeeze=False)
        axes = axes[:, 0].reshape(self.observation.n_regions, 2)
        for i, (wave, flux, fixed, fitted, continuum_flux, mean_flux) in enumerate(zip(
                _rows(self.observation.wavelength), _rows(self.observation.flux),
                _rows(self.observation.mask), _rows(self.fit_mask), _rows(baseline), _rows(mean))):
            upper, lower = axes[i]
            lower.sharex(upper)
            for select, style, label in ((~(fixed | fitted), ".", "usable"),
                                          (fixed, "x", "fixed mask"),
                                          (fitted & ~fixed, "o", "fit mask")):
                upper.plot(wave[select], flux[select], style, ms=3, label=label)
                lower.plot(wave[select], (flux - mean_flux)[select], style, ms=3)
            upper.plot(wave, continuum_flux, lw=1, label="continuum_flux")
            upper.plot(wave, mean_flux, lw=1, label="mean_flux")
            if physical is not None:
                upper.plot(wave, _rows(physical)[i], lw=.8, label="physical_flux")
            valid = ~(fixed | fitted)
            if np.any(valid):
                shown = [flux[valid], continuum_flux[valid], mean_flux[valid]]
                if physical is not None:
                    shown.append(_rows(physical)[i, valid])
                low, high = np.min(shown), np.max(shown)
                margin = .05 * (high - low if high > low else max(abs(high), 1.))
                upper.set_ylim(low - margin, high + margin)
                maximum = np.max(np.abs(flux[valid] - mean_flux[valid]))
                if maximum == 0:
                    maximum = np.median(_rows(self.observation.uncertainty)[i, valid])
                lower.set_ylim(-res_factor * maximum, res_factor * maximum)
            lower.axhline(0, color="gray", lw=.5)
            upper.set(title=f"Region {self.orders[i]}", ylabel="Flux")
            lower.set(xlabel="Wavelength [Å]", ylabel="Residual")
            upper.legend(fontsize=8)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        return fig, axes
