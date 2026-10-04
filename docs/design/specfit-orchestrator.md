# SpecFit orchestration

`jaxstar.specfit.SpecFit` is optional convenience over already constructed
`Observation` and `SpecModel`. It does not load libraries, select scientific
priors, define a second likelihood, or own posterior samples. Direct calls to
`model_single`, `find_map_svi`, ordinary NumPyro, continuum and GP helpers remain
unchanged. The constructor checks region count/order and basic wavelength
coverage; parameter-dependent broadening/RV coverage remains in `SpecModel`.
`orders` supplies display labels, not a data-selection/reordering operation.

Persistent state consists of the observation, physical model, display labels,
Boolean fit mask and the last CCF diagnostic arrays. There is one `SpecFit`, no
single/binary class hierarchy. Custom ordinary NumPyro callables remain usable
with either inference wrapper.

## Masks and scientific behavior

`Observation.mask` is the immutable fixed data mask. `fit.fit_mask` starts false
and is replaced via `set_fit_mask`, `reset_fit_mask` or `mask_outliers`.
`fit.fit_observation()` uses the current Observation constructor, preserving
array values/dtypes and metadata with mask `observation.mask | fit_mask`.
Mask management, CCF, clipping and diagnostics run outside JAX tracing.

Outlier clipping recomputes from the original fixed-mask usable pixels, ignoring
the previous fit mask. A better refit can therefore recover falsely excluded
pixels. It uses the frozen jaxspec residual-minus-median-filter statistic,
`1.4826 * MAD`, sigma threshold and optional contiguous-run extension by
`ceil(extension_factor * run_length)` on both sides. Filter width retains the
old `int(median(wave) * mask_v * 2 / 3e5 / median(diff(wave))) * 4 + 1`
rule. `mask_v` defaults to the largest reconstructed component vsini (km/s);
an explicit scalar or region vector may override it.

The default baseline is **`reconstruction['continuum_flux']`**, the physical
spectrum times the conditional continuum mean, as the analogue of old fitted
`fluxmodel`. The GP mean is deliberately excluded. Clipping against a different
baseline, including the full GP conditional mean, requires explicit
`prediction=...`. Clipping never reconstructs an ensemble of posterior draws.

Safety differences from frozen jaxspec: fixed masked values are omitted from
local medians as well as the MAD/flags, so NaN/Inf cannot contaminate neighbors;
zero edge padding is retained. Oversized median windows are capped to the
largest odd width no greater than the region size. Extended flags stay outside
the fixed mask. Ordinary finite unmasked spectra use the same statistic.

## CCF and inference

`check_ccf(atmosphere, v_limit=500, ccfvmax=100, oversample_factor=5)` requires
explicit library-named template coordinates, using `SpecModel.intrinsic` on its
prepared wavelength grid. The CCF preserves old jaxspec's unweighted,
mean-subtracted, log-grid interpolation/correlation and `1e-4` dex edge margins.
Currently usable pixels are selected before interpolation, including its legacy
linear interpolation across excluded gaps. Per-region peak RVs are combined
with a median; region CCFs are interpolated onto 10,000 common velocity samples,
then their median is used for the outermost half-maximum crossing width.
Returns `(rv, width)` in km/s, with arrays retained in `fit.ccf`.

Too few usable pixels, insufficient wavelength span/lag coverage, absent
positive peaks or absent half-maximum crossings fail explicitly. Peak search
uses negative infinity outside `v_limit` instead of legacy zero, avoiding an
out-of-window peak when correlations are negative. Width is a diagnostic, not
an automatic prior or assertion against the broadening kernel's `vmax`.

`run_svi` directly calls `numpyro_inferutils.find_map_svi`; its return is the
native AutoLaplace guide-median dict. Supply `rng_key`, `model_kwargs`, optionally
`step_size`, `num_steps`, `p_initial`, `progress_bar`, and `model`.
`run_nuts` directly returns NumPyro `MCMC(NUTS(...))`, exposing `rng_key`,
`model_kwargs`, warmup/sample/chain counts, progress, `nuts_kwargs` and
`mcmc_kwargs`. Both bind the current effective Observation and SpecModel before
inference. GP masks are concrete; `jit_model_args=True` is rejected. Both accept
custom `model(observation, specmodel, **model_kwargs)` callables; there is no
model registry or new optimizer.

## Reconstruction and diagnostics

`reconstruct(point, model_kwargs=..., model=model_single)` consumes one explicit
native sample-site dict. It replays the same external model, requiring every
latent site without random sampling. Fixed/derived sites are recomputed from
configuration, including empirical vmic/vmacro and physical-logg options.
A custom reconstruction model must return SpecModel params and record native
`sigma_constant`, `sigma_continuum`, `jitter`, and paired GP sites if enabled.
Custom inference itself has no such return/site requirement.

The returned ordinary dict contains `params`, `physical_flux`, `continuum`,
`continuum_posterior`, `continuum_flux`, `mean_flux`, `jitter`, plus `gp_mean`
only in GP mode. It uses existing iid/GP continuum posterior/evaluation and GP
conditional-mean helpers. Arrays are reconstructed once at this point, never
saved in each posterior draw by SpecFit.

`residual_diagnostics(prediction, jitter=0)` returns per-region usable counts,
RMS, robust MAD scale, standardized RMS, >3σ/>5σ counts and lag-1 correlation.
Standardization includes explicitly supplied additive jitter in flux units.
Lag-1 pairs must be adjacent and usable in the original array, never bridged
across gaps. Undefined/empty statistics are NaN, counts zero. For final GP
diagnostics use `mean_flux`; this is separate from the clipping baseline.
`plot_models(reconstruction, show_physical=False, res_factor=1.5, save_path=None)` uses existing
arrays, showing data/fixed/fit masks, continuum/full means and residuals; returns
Matplotlib figure and `(n_region, 2)` axes. Limits use usable pixels so masked
extremes do not squash the display, as in legacy residual plots. Change the
returned axes limits to inspect excluded extremes. Plot imports are lazy.

## Canonical workflow

```python
fit = SpecFit(obs, specmodel)
rv, width = fit.check_ccf(template_atmosphere)
# Caller chooses priors/initial values, possibly using the CCF diagnostics.
point = fit.run_svi(rng_key=key1, model_kwargs=config, p_initial=initial)
pred = fit.reconstruct(point, model_kwargs=config)
fit.mask_outliers(pred)  # continuum_flux, not GP conditional mean
point = fit.run_svi(rng_key=key2, model_kwargs=config, p_initial=point)
mcmc = fit.run_nuts(rng_key=key3, model_kwargs=config,
                   nuts_kwargs={"init_strategy": init_to_value(values=point)})
point = {name: np.median(draws, axis=0)
         for name, draws in mcmc.get_samples().items()}
pred = fit.reconstruct(point, model_kwargs=config)
stats = fit.residual_diagnostics(pred['mean_flux'], jitter=pred['jitter'])
fig, axes = fit.plot_models(pred)
```

The executed [development notebook](../../dev_notebooks/specfit/specfit_end_to_end.ipynb)
demonstrates synthetic CCF → SVI → mask replacement → refit → short NUTS →
point reconstruction and diagnostics, with a small optional GP section. Short
inference runs demonstrate plumbing, not scientific convergence. The separate
local real-order notebook stays ignored and is neither a package-test nor
development-notebook dependency.
