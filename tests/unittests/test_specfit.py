"""SpecFit connects existing physics/inference, preserving mask and CCF behavior."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, init_to_value
import pytest

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (
    Observation, SpecModel, SpecFit, single_star_params, chebyshev_basis,
    continuum_posterior, evaluate_continuum, gp_continuum_posterior, gp_conditional_mean,
)
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit.fit import _extend_mask


def case(single=False):
    count = 1 if single else 2
    centers = np.array([5000., 6000.])[:count, None]
    velocity = np.linspace(-250., 250., 601)
    wave = centers * np.exp(velocity / 299792.458)
    profile = np.exp(-.5 * (velocity / 4.)**2)
    nodes = np.stack([np.broadcast_to(1 - depth * profile, wave.shape) for depth in (.2, .5)])
    grid = RectilinearGrid(axes={"teff": np.array([5000., 6500.])},
        fields={"flux": Field(nodes, ("teff",), payload_dims=("region", "pixel"))})
    model = SpecModel(_SpectralLibrary(grid, wave, (8, 9)[:count], "fixture"), vmax=30.)
    wavelength = centers * np.exp(np.linspace(-120.37, 120.19, 96) / 299792.458)
    point = {"teff": 5800.}
    broadening = dict(vsini=6., vmacro=3., q1=.36, q2=.3)
    params = single_star_params(point, **broadening, rv=12., resolving_power=70000.)
    physical = np.asarray(model(params, wavelength))
    measured = physical * (1.01 + .01 * np.linspace(-1, 1, 96))
    mask = np.zeros_like(measured, dtype=bool)
    mask[:, 3] = True
    errors = np.full(measured.shape, .01)
    measured[mask], errors[mask] = np.nan, np.inf
    if single:
        wavelength, measured, errors, mask = (x[0] for x in (wavelength, measured, errors, mask))
    obs = Observation(wavelength, measured, errors, mask, region=(8, 9)[:count],
                      order=(108, 109)[:count], exposure="synthetic")
    config = dict(atmosphere={"teff": dist.Uniform(5000., 6500.)},
                  broadening=broadening, rv=12., resolving_power=70000.,
                  sigma_constant=.1, sigma_continuum=.03, jitter=.004,
                  basis=chebyshev_basis(obs.wavelength))
    return SpecFit(obs, model), config, point


@pytest.mark.parametrize("single", [False, True])
def test_masks_and_preserved_observation(single):
    fit, _, _ = case(single)
    original = np.asarray(fit.observation.mask).copy()
    assert not fit.fit_mask.any()
    flags = np.zeros(fit.observation.shape, bool)
    flags[..., 10] = True
    fit.set_fit_mask(flags)
    flags[...] = False
    effective = fit.fit_observation()
    assert effective.mask[..., 10].all()
    np.testing.assert_array_equal(effective.mask, original | fit.fit_mask)
    for name in ("wavelength", "flux", "uncertainty"):
        np.testing.assert_array_equal(getattr(effective, name), getattr(fit.observation, name))
        assert getattr(effective, name).dtype == getattr(fit.observation, name).dtype
    assert effective.region == fit.observation.region
    assert effective.order == fit.observation.order and effective.exposure == "synthetic"
    np.testing.assert_array_equal(fit.observation.mask, original)
    assert not fit.fit_mask.flags.writeable
    fit.reset_fit_mask()
    assert not fit.fit_mask.any()
    with pytest.raises(ValueError, match="shape"):
        fit.set_fit_mask(np.zeros(3, bool))
    with pytest.raises(TypeError, match="Boolean"):
        fit.set_fit_mask(np.zeros(fit.observation.shape))


def test_association():
    fit, _, _ = case()
    with pytest.raises(TypeError, match="Observation"):
        SpecFit(None, fit.specmodel)
    with pytest.raises(ValueError, match="row order"):
        SpecFit(replace(fit.observation, region=(9, 8)), fit.specmodel)
    with pytest.raises(ValueError, match="orders"):
        SpecFit(fit.observation, fit.specmodel, orders=(8,))
    with pytest.raises(ValueError, match="coverage"):
        SpecFit(replace(fit.observation, wavelength=fit.observation.wavelength + 100.), fit.specmodel)
    single = Observation(*[np.asarray(getattr(fit.observation, k))[0]
                           for k in ("wavelength", "flux", "uncertainty", "mask")])
    with pytest.raises(ValueError, match="number of regions"):
        SpecFit(single, fit.specmodel)


def test_ccf_recovers_rv_and_respects_both_masks():
    fit, _, _ = case()
    extra = np.zeros(fit.observation.shape, bool)
    extra[:, 48] = True
    fit.set_fit_mask(extra)
    rv, width = fit.check_ccf({"teff": 5800.}, v_limit=50., ccfvmax=40.)
    assert abs(rv - 12.) < 2.
    assert 5. < width < 30.
    np.testing.assert_allclose(fit.ccf["region_rv"], rv, atol=.2)
    contaminated = np.asarray(fit.observation.flux).copy()
    contaminated[fit.effective_mask] = 1e10
    alternate = SpecFit(replace(fit.observation, flux=contaminated), fit.specmodel)
    alternate.set_fit_mask(extra)
    np.testing.assert_allclose(alternate.check_ccf({"teff": 5800.}, v_limit=50., ccfvmax=40.), (rv, width))
    np.testing.assert_allclose(alternate.ccf["combined"], fit.ccf["combined"])


def test_ccf_invalid_geometry_and_missing_peak():
    fit, _, _ = case()
    with pytest.raises(ValueError, match="positive integer"):
        fit.check_ccf({"teff": 5800.}, oversample_factor=1.5)
    fit.set_fit_mask(np.ones(fit.observation.shape, bool))
    with pytest.raises(ValueError, match="usable pixels"):
        fit.check_ccf({"teff": 5800.})
    flat = SpecFit(replace(fit.observation, flux=np.ones(fit.observation.shape)), fit.specmodel)
    with pytest.raises(ValueError, match="positive peak"):
        flat.check_ccf({"teff": 5800.})


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("gp", [False, True])
def test_reconstruction_matches_helpers(single, gp, x64_context):
    with x64_context(True):
        fit, config, point = case(single)
        extra = np.zeros(fit.observation.shape, bool)
        extra[..., 48] = True
        fit.set_fit_mask(extra)
        if gp:
            config.update(gp_amplitude=.02, gp_scale=.15, gp_solver="direct")
        result = fit.reconstruct(point, model_kwargs=config)
        obs = fit.fit_observation()
        physical = fit.specmodel(result["params"], obs.wavelength)
        options = dict(sigma_constant=.1, sigma_continuum=.03, jitter=.004, basis=config["basis"])
        if gp:
            post = gp_continuum_posterior(obs, physical, **options,
                                         gp_amplitude=.02, gp_scale=.15, gp_solver="direct")
        else:
            post = continuum_posterior(obs, physical, **options)
        continuum = evaluate_continuum(config["basis"], post.mean)
        np.testing.assert_allclose(result["physical_flux"], physical, atol=1e-12)
        np.testing.assert_allclose(result["continuum"], continuum, atol=1e-12)
        np.testing.assert_allclose(result["continuum_flux"], physical * continuum, atol=1e-12)
        np.testing.assert_allclose(result["continuum_posterior"].covariance, post.covariance, atol=1e-12)
        if gp:
            expected = gp_conditional_mean(obs, physical * continuum, jitter=.004,
                                          gp_amplitude=.02, gp_scale=.15, gp_solver="direct")
            np.testing.assert_allclose(result["gp_mean"], expected, atol=1e-12)
            np.testing.assert_allclose(result["mean_flux"], physical * continuum + expected, atol=1e-12)
        else:
            assert "gp_mean" not in result
            np.testing.assert_array_equal(result["mean_flux"], result["continuum_flux"])
        with pytest.raises(ValueError, match="latent site 'teff'"):
            fit.reconstruct({}, model_kwargs=config)


def test_reconstruct_replays_empirical_model_and_ignores_derived_point_sites():
    fit, config, point = case()
    config["broadening"] = {key: value for key, value in config["broadening"].items() if key != "vmacro"}
    config["use_empirical_vmacro"] = True
    result = fit.reconstruct(dict(point, vmacro=4., u1=999.), model_kwargs=config)
    assert result["params"]["components"][0]["broadening"]["vmacro"] == 4.
    np.testing.assert_allclose(result["params"]["components"][0]["broadening"]["u1"], .36, atol=1e-6)
    assert np.all(np.isfinite(result["mean_flux"]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_region_jitter_and_fixed_physical_point(dtype, x64_context):
    with x64_context(dtype == np.float64):
        fit, config, point = case()
        obs = Observation(*(np.asarray(getattr(fit.observation, name)).astype(dtype)
                            for name in ("wavelength", "flux", "uncertainty")),
                          mask=fit.observation.mask, region=fit.observation.region)
        fit = SpecFit(obs, fit.specmodel)
        config.update(atmosphere=point, basis=chebyshev_basis(obs.wavelength),
                      jitter=np.array([.004, .008], dtype=dtype))
        result = fit.reconstruct({}, model_kwargs=config)
        assert result["mean_flux"].dtype == np.dtype(dtype)
        diagnostics = fit.residual_diagnostics(result["mean_flux"], jitter=result["jitter"])
        assert len(diagnostics) == 2
        assert all(np.isfinite(row["standardized_rms"]) for row in diagnostics)
        with pytest.raises(ValueError, match="prediction must have shape"):
            fit.residual_diagnostics(np.ones(96))


@pytest.mark.parametrize("gp", [False, True])
def test_default_svi_and_nuts_use_effective_observation(gp):
    pytest.importorskip("numpyro_inferutils")
    fit, config, _ = case(single=True)
    if gp:
        pytest.importorskip("tinygp")
        config.update(gp_amplitude=.015, gp_scale=.15)
    extra = np.zeros(fit.observation.shape, bool)
    extra[48] = True
    # A large contaminated pixel must not pull inference away from the truth.
    flux = np.asarray(fit.observation.flux).copy()
    flux[48] = 1e3
    fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
    fit.set_fit_mask(extra)
    estimate = fit.run_svi(rng_key=jax.random.PRNGKey(3), model_kwargs=config,
                          step_size=.04, num_steps=100, p_initial={"teff": 5600.}, progress_bar=False)
    assert np.isfinite(estimate["teff"])
    assert abs(float(estimate["teff"]) - 5800.) < 180.
    mcmc = fit.run_nuts(rng_key=jax.random.PRNGKey(4), model_kwargs=config,
                       num_warmup=12, num_samples=12, progress_bar=False,
                       nuts_kwargs={"init_strategy": init_to_value(values=estimate), "max_tree_depth": 3})
    assert isinstance(mcmc, MCMC)
    assert mcmc.get_samples()["teff"].shape == (12,)
    assert np.all(np.isfinite(mcmc.get_samples()["teff"]))
    assert not {"model_flux", "gp_mean", "continuum_flux"}.intersection(mcmc.get_samples())
    with pytest.raises(ValueError, match="jit_model_args"):
        fit.run_nuts(rng_key=jax.random.PRNGKey(0), model_kwargs=config,
                     mcmc_kwargs={"jit_model_args": True})


def test_custom_svi_nuts_callable():
    pytest.importorskip("numpyro_inferutils")
    fit, _, _ = case(single=True)
    fit.set_fit_mask(np.arange(fit.observation.n_pixels) == 48)
    effective = fit.fit_observation()

    def custom(observation, specmodel, *, center):
        np.testing.assert_array_equal(observation.mask, effective.mask)
        assert specmodel is fit.specmodel
        x = numpyro.sample("custom_parameter", dist.Normal(center, 1.))
        numpyro.factor("custom_density", -.5 * (x - center)**2)

    result = fit.run_svi(model=custom, model_kwargs={"center": 2.}, rng_key=jax.random.PRNGKey(1),
                         num_steps=30, step_size=.08, progress_bar=False,
                         p_initial={"custom_parameter": 1.})
    assert abs(float(result["custom_parameter"]) - 2.) < .5
    mcmc = fit.run_nuts(model=custom, model_kwargs={"center": 2.}, rng_key=jax.random.PRNGKey(2),
                        num_warmup=5, num_samples=5, progress_bar=False)
    assert np.isfinite(mcmc.get_samples()["custom_parameter"]).all()


def clipping_case():
    fit, _, _ = case(single=True)
    wave = np.linspace(5000., 5001., 96)
    noise = .001 * np.random.default_rng(19).normal(size=96)
    flux = np.ones(96) + noise
    flux[[10, 70]] += .2
    mask = np.zeros(96, bool)
    mask[70] = True
    flux[70] = np.nan
    error = np.full(96, .01)
    error[70] = np.inf
    fit = SpecFit(Observation(wave, flux, error, mask), fit.specmodel)
    return fit, np.ones(96)


def test_outliers_default_gp_baseline_replacement_override_and_fixed_nan():
    fit, baseline = clipping_case()
    pred = {"continuum_flux": baseline, "mean_flux": fit.observation.flux,
            "params": {"components": ({"broadening": {"vsini": 2.}},)}}
    flags = fit.mask_outliers(pred, extend_outlier_mask=False).copy()
    assert flags[10] and not flags[70]
    # Explicit GP-mean-like baseline absorbs the spike only when requested.
    fit.mask_outliers(pred, prediction=pred["mean_flux"], extend_outlier_mask=False)
    assert not fit.fit_mask.any()
    # Improved baseline fits the first spike but introduces a second discrepancy.
    improved = np.array(fit.observation.flux, copy=True)
    improved[20] -= .2
    fit.set_fit_mask(flags)
    fit.mask_outliers(prediction=improved, mask_v=2., extend_outlier_mask=False)
    assert fit.fit_mask[20] and not fit.fit_mask[10]
    assert not fit.fit_mask[70]
    fixed = fit.observation.mask.copy()
    fit.reset_fit_mask()
    np.testing.assert_array_equal(fit.observation.mask, fixed)
    assert not fit.fit_mask.any()


def test_extension_exact_runs_and_edges():
    flag = np.zeros(12, bool)
    flag[[0, 4, 5, 11]] = True
    expected = np.zeros(12, bool)
    expected[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11]] = True
    np.testing.assert_array_equal(_extend_mask(flag, 1.), expected)
    np.testing.assert_array_equal(_extend_mask(flag, 0.), flag)
    fit, baseline = clipping_case()
    flags = fit.mask_outliers(prediction=baseline, mask_v=2.)
    assert flags[9:12].all()
    with pytest.raises(ValueError, match="mask_v"):
        fit.mask_outliers(prediction=baseline)
    with pytest.raises(ValueError, match="positive"):
        fit.mask_outliers(prediction=baseline, mask_v=2., sigma_threshold=0.)


def test_masked_filter_poison_and_oversized_kernel():
    fit, baseline = clipping_case()
    flags = fit.mask_outliers(prediction=baseline, mask_v=2., extend_outlier_mask=False).copy()
    finite_flux = np.where(fit.observation.mask, 1e10, fit.observation.flux)
    other = SpecFit(replace(fit.observation, flux=finite_flux), fit.specmodel)
    np.testing.assert_array_equal(other.mask_outliers(prediction=baseline, mask_v=2.,
                                                     extend_outlier_mask=False), flags)
    fit.mask_outliers(prediction=baseline, mask_v=1e6)
    fit.set_fit_mask(np.zeros_like(flags))
    entirely_masked = SpecFit(replace(fit.observation, mask=np.ones_like(flags)), fit.specmodel)
    entirely_masked.mask_outliers(prediction=np.full(flags.shape, np.nan), mask_v=2.)
    assert not entirely_masked.fit_mask.any()


def test_residual_diagnostics_manual_and_no_bridging():
    fit, _, _ = case(single=True)
    wave = np.linspace(5000., 5001., 8)
    residual = np.array([0., 1., 2., np.nan, 4., 6., 10., 12.])
    mask = np.array([False, False, False, True, False, False, False, False])
    obs = Observation(wave, 1 + residual, np.ones(8), mask)
    fit = SpecFit(obs, fit.specmodel)
    extra = np.zeros(8, bool)
    extra[6] = True
    fit.set_fit_mask(extra)
    stats = fit.residual_diagnostics(np.ones(8), jitter=1.)[0]
    valid_res = np.array([0., 1., 2., 4., 6., 12.])
    assert stats["n_usable"] == 6
    np.testing.assert_allclose(stats["rms"], np.sqrt(np.mean(valid_res**2)))
    np.testing.assert_allclose(stats["mad_scale"], 1.4826 * np.median(np.abs(valid_res - np.median(valid_res))))
    np.testing.assert_allclose(stats["standardized_rms"], stats["rms"] / np.sqrt(2))
    assert stats["n_gt_3sigma"] == 2 and stats["n_gt_5sigma"] == 1
    assert stats["n_adjacent_pairs"] == 3
    np.testing.assert_allclose(stats["lag1_correlation"], np.corrcoef([0., 1., 4.], [1., 2., 6.])[0, 1])
    fit.set_fit_mask(np.ones(8, bool))
    empty = fit.residual_diagnostics(np.full(8, np.nan))[0]
    assert empty["n_usable"] == 0 and empty["n_gt_3sigma"] == 0
    assert np.isnan(empty["rms"]) and np.isnan(empty["lag1_correlation"])
    with pytest.raises(ValueError, match="jitter"):
        fit.residual_diagnostics(np.ones(8), jitter=-1.)


@pytest.mark.parametrize("single", [False, True])
def test_plot_smoke(single, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, config, point = case(single)
    pred = fit.reconstruct(point, model_kwargs=config)
    flags = np.zeros(fit.observation.shape, bool)
    flags[..., 48] = True
    fit.set_fit_mask(flags)
    fig, axes = fit.plot_models(pred, show_physical=True, save_path=tmp_path / "models.png")
    assert axes.shape == (fit.observation.n_regions, 2)
    assert (tmp_path / "models.png").stat().st_size > 0
    plt.close(fig)


def test_plot_limits_ignore_masked_extremes():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, config, point = case(single=True)
    flux = np.asarray(fit.observation.flux).copy()
    flux[3] = 1e8  # fixed mask
    flux[48] = -1e8  # fit mask
    fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
    flags = np.zeros(fit.observation.shape, bool)
    flags[48] = True
    fit.set_fit_mask(flags)
    pred = fit.reconstruct(point, model_kwargs=config)
    fig, axes = fit.plot_models(pred)
    assert axes[0, 0].get_ylim()[0] > 0 and axes[0, 0].get_ylim()[1] < 2
    assert axes[0, 1].get_ylim()[1] < 1
    plt.close(fig)
