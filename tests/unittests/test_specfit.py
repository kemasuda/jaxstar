"""SpecFit connects existing physics/inference, preserving mask and CCF behavior."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, init_to_value
import pytest

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (
    ConfiguredModel, reconstruct_single, Observation, SpecModel, SpecFit, model_single, single_star_params, chebyshev_basis,
    continuum_posterior, evaluate_continuum, gp_continuum_posterior, gp_conditional_mean,
)
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit.specfit import _extend_mask


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
                      exposure="synthetic")
    config = dict(atmosphere={"teff": dist.Uniform(5000., 6500.)},
                  broadening=broadening, rv=12., resolving_power=70000.,
                  sigma_constant=.1, sigma_continuum=.03, jitter=.004,
                  basis=chebyshev_basis(obs.wavelength))
    return SpecFit(obs, model), config, point


def configured(fit, settings):
    return ConfiguredModel(model_single, settings=settings, reconstruct_fn=reconstruct_single,
                           observation=fit.observation, specmodel=fit.specmodel)


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
    assert effective.exposure == "synthetic"
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


def coelho_case(single=False, logg=4.4):
    fit, _, _ = case(single)
    spectra = fit.specmodel.spectra
    flux = np.asarray(spectra.grid.field("flux").values)
    axes = {"teff": spectra.grid.axis("teff"), "logg": np.array([logg]),
            "feh": np.array([0.]), "alpha": np.array([0.])}
    grid = RectilinearGrid(axes=axes, fields={"flux": Field(
        flux[:, None, None, None, :, :], tuple(axes), payload_dims=("region", "pixel"))})
    model = SpecModel(replace(spectra, grid=grid, library="coelho"), vmax=30.)
    return SpecFit(fit.observation, model)


@pytest.mark.parametrize("single", [False, True])
def test_ccf_default_coelho_template_and_explicit_override(single):
    fit = coelho_case(single)
    template = {"teff": 5800., "logg": 4.4, "feh": 0., "alpha": 0.}
    controls = dict(v_limit=50., ccfvmax=40.)
    expected = fit.check_ccf(template, **controls)
    expected_ccf = fit.ccf["combined"].copy()
    for options in ({}, {"atmosphere": None}):
        np.testing.assert_array_equal(fit.check_ccf(**options, **controls), expected)
        np.testing.assert_array_equal(fit.ccf["combined"], expected_ccf)
    assert abs(expected[0] - 12.) < 2.
    fit.check_ccf({**template, "teff": 6000.}, **controls)
    assert not np.array_equal(fit.ccf["combined"], expected_ccf)
    np.testing.assert_array_equal(fit.check_ccf(**controls), expected)


def test_ccf_default_outside_coelho_grid_is_not_clipped():
    fit = coelho_case(logg=4.)
    with pytest.raises(ValueError, match="inside the prepared grid"):
        fit.check_ccf()
    rv, _ = fit.check_ccf({"teff": 5800., "logg": 4., "feh": 0., "alpha": 0.},
                          v_limit=50., ccfvmax=40.)
    assert abs(rv - 12.) < 2.


@pytest.mark.parametrize("library", ["bosz", "tlusty", "custom"])
def test_ccf_other_libraries_require_explicit_template(library):
    fit, _, _ = case()
    model = SpecModel(replace(fit.specmodel.spectra, library=library), vmax=30.)
    fit = SpecFit(fit.observation, model)
    with pytest.raises(ValueError, match="supply atmosphere explicitly"):
        fit.check_ccf()
    rv, _ = fit.check_ccf({"teff": 5800.}, v_limit=50., ccfvmax=40.)
    assert abs(rv - 12.) < 2.


@pytest.mark.parametrize("single", [False, True])
def test_ccf_plot_and_save_preserve_results(single, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit = coelho_case(single)
    controls = dict(v_limit=50., ccfvmax=40.)
    before = set(plt.get_fignums())
    expected = fit.check_ccf(**controls)
    stored = fit.ccf
    assert set(plt.get_fignums()) == before
    actual = fit.check_ccf(**controls, plot=True, save_path=tmp_path / "ccf.png")
    figures = [plt.figure(n) for n in sorted(set(plt.get_fignums()) - before)]
    try:
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(fit.ccf["combined"], stored["combined"])
        assert len(figures) == 1
        fig = figures[0]
        ax, = fig.axes
        for line, velocity, values, saved in zip(
                ax.lines, stored["velocities"], stored["ccfs"], fit.ccf["ccfs"]):
            np.testing.assert_array_equal(saved, values)
            np.testing.assert_array_equal(line.get_xdata(), velocity)
            np.testing.assert_allclose(line.get_ydata(), values / np.max(values))
        median = ax.lines[fit.observation.n_regions]
        np.testing.assert_array_equal(median.get_xdata(), stored["velocity"])
        np.testing.assert_allclose(median.get_ydata(),
                                   stored["combined"] / np.max(stored["combined"]))
        np.testing.assert_array_equal(ax.lines[-1].get_xdata(), [actual[0], actual[0]])
        np.testing.assert_allclose(ax.get_xlim(), [actual[0] - 40., actual[0] + 40.])
        title = ax.get_title(loc="left")
        assert f"Median RV = {actual[0]:.3f} km/s" in title
        assert f"CCF FWHM = {actual[1]:.3f} km/s" in title
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            *[f"Region {region}" for region in fit.regions], "Median CCF"]
        assert (tmp_path / "ccf.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert not list(tmp_path.glob("ccf_region*"))
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("filename", ["ccf.pdf", "ccf"])
def test_ccf_save_without_display(filename, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, _, _ = case()
    existing = plt.figure()
    before = set(plt.get_fignums())
    try:
        with plt.rc_context({"savefig.format": "pdf"}):
            rv, width = fit.check_ccf({"teff": 5800.}, v_limit=50., ccfvmax=40.,
                                      save_path=tmp_path / filename)
        assert (rv, width) == (fit.ccf["rv"], fit.ccf["width"])
        assert set(plt.get_fignums()) == before
        assert (tmp_path / "ccf.pdf").read_bytes().startswith(b"%PDF")
    finally:
        plt.close(existing)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("gp", [False, True])
@pytest.mark.parametrize("degree", [1, 0, 4])
@pytest.mark.parametrize("provide_basis", [False, True])
def test_reconstruction_matches_helpers(single, gp, degree, provide_basis, x64_context):
    with x64_context(True):
        fit, config, point = case(single)
        if degree != 4:
            config["degree"] = degree
        basis = chebyshev_basis(fit.observation.wavelength, degree=degree)
        if provide_basis:
            config["basis"] = basis
        else:
            config.pop("basis")
        extra = np.zeros(fit.observation.shape, bool)
        extra[..., 48] = True
        fit.set_fit_mask(extra)
        if gp:
            config.update(gp_amplitude=.02, gp_scale=.15, gp_solver="direct")
        result = fit.predict(point=point, model=configured(fit, config))
        obs = fit.fit_observation()
        physical = fit.specmodel(result["params"], obs.wavelength)
        options = dict(sigma_constant=.1, sigma_continuum=.03, jitter=.004,
                       degree=degree, basis=basis)
        if gp:
            post = gp_continuum_posterior(obs, physical, **options,
                                         gp_amplitude=.02, gp_scale=.15, gp_solver="direct")
        else:
            post = continuum_posterior(obs, physical, **options)
        continuum = evaluate_continuum(basis, post.mean)
        assert result["continuum_posterior"].mean.shape == obs.shape[:-1] + (degree + 1,)
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
            fit.predict(point={}, model=configured(fit, config))


def test_reconstruct_replays_empirical_model_and_ignores_derived_point_sites():
    fit, config, point = case()
    config["broadening"] = {key: value for key, value in config["broadening"].items() if key != "vmacro"}
    config["use_empirical_vmacro"] = True
    result = fit.predict(point=dict(point, vmacro=4., u1=999.), model=configured(fit, config))
    assert result["params"]["components"][0]["broadening"]["vmacro"] == 4.
    np.testing.assert_allclose(result["params"]["components"][0]["broadening"]["u1"], .36, atol=1e-6)
    assert np.all(np.isfinite(result["mean_flux"]))


@pytest.mark.parametrize("override", [None, 0])
def test_reconstruction_uses_model_signature_defaults(override, x64_context):
    with x64_context(True):
        fit, config, point = case(single=True)
        config.pop("basis")
        config.update(gp_amplitude=.02, gp_scale=.15)
        if override is not None:
            config["degree"] = override
        function = partial(model_single, degree=2, gp_solver="direct")
        model = ConfiguredModel(function, settings=config, reconstruct_fn=reconstruct_single,
                                observation=fit.observation, specmodel=fit.specmodel)
        degree = 2 if override is None else override
        assert model.settings["degree"] == degree
        assert model.settings["basis"] is None
        assert model.settings["gp_solver"] == "direct"
        result = fit.predict(point=point, model=model)
        expected = gp_continuum_posterior(
            fit.fit_observation(), result["physical_flux"], degree=degree,
            sigma_constant=.1, sigma_continuum=.03, jitter=.004,
            gp_amplitude=.02, gp_scale=.15, gp_solver="direct")
        assert result["continuum_posterior"].mean.shape == (degree + 1,)
        np.testing.assert_allclose(result["continuum_posterior"].mean, expected.mean, atol=1e-12)


def test_reconstruction_requires_resolved_model_settings():
    fit, config, point = case(single=True)
    config.pop("basis")

    # A wrapper hiding its continuum settings cannot use the standard callback
    # until it exposes them: reconstruction must not silently assume degree=4.
    def hidden_defaults(observation, specmodel, **options):
        return model_single(observation, specmodel, degree=2, **options)

    model = ConfiguredModel(hidden_defaults, settings=config, reconstruct_fn=reconstruct_single,
                            observation=fit.observation, specmodel=fit.specmodel)
    with pytest.raises(KeyError, match="degree"):
        fit.predict(point=point, model=model)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("max_draws", [None, 2])
def test_posterior_prediction_averages_model_curves(single, max_draws, x64_context):
    with x64_context(True):
        fit, config, _ = case(single)
        config.update(gp_amplitude=dist.LogUniform(.001, .1),
                      gp_scale=dist.LogUniform(.01, 1.), gp_solver="direct",
                      rv=dist.Uniform(8., 16.).expand((fit.observation.n_regions,)))
        model = configured(fit, config)
        samples = dict(teff=np.array([5200., 5600., 6200.]),
                       gp_amplitude=np.array([.004, .025, .07]),
                       gp_scale=np.array([.04, .12, .4]),
                       rv=np.repeat([[11.], [12.], [13.]], fit.observation.n_regions, axis=1),
                       u1=np.full(3, 999.))  # Derived sites in get_samples are ignored on replay.
        flags = np.zeros(fit.observation.shape, bool)
        flags[..., 48] = True
        fit.set_fit_mask(flags)
        result = fit.predict(samples=samples, model=model, max_draws=max_draws)
        indices = [0, 1, 2] if max_draws is None else [0, 2]
        np.testing.assert_array_equal(result["draw_indices"], indices)
        predictions = [fit.predict(point={name: value[i] for name, value in samples.items()}, model=model)
                       for i in indices]
        for name in ("physical_flux", "continuum", "continuum_flux", "mean_flux", "gp_mean"):
            assert result[name].shape == fit.observation.shape
            np.testing.assert_allclose(result[name], np.mean([p[name] for p in predictions], axis=0), atol=1e-12)
        np.testing.assert_allclose(result["mean_flux"] - result["continuum_flux"], result["gp_mean"], atol=1e-12)
        np.testing.assert_allclose(result["gp_mean_quantiles"],
                                   np.quantile([p["gp_mean"] for p in predictions], [.16, .84], axis=0),
                                   atol=1e-12)
        assert not {"params", "continuum_posterior", "jitter"}.intersection(result)
        mean_point = {name: np.mean(value[indices], axis=0) for name, value in samples.items()}
        at_mean = fit.predict(point=mean_point, model=model)
        assert np.max(np.abs(result["gp_mean"] - at_mean["gp_mean"])) > 1e-5


@pytest.mark.parametrize("options,message", [
    ({}, "exactly one"),
    ({"point": {}, "samples": {"teff": [5800.]}}, "exactly one"),
    ({"samples": {}}, "nonempty"),
    ({"samples": {"teff": 5800.}}, "leading draw axis"),
    ({"samples": {"teff": []}}, "same positive number"),
    ({"samples": {"teff": [5800.], "rv": [11., 12.]}}, "same positive number"),
    ({"samples": {"teff": [5800.]}, "max_draws": 0}, "max_draws"),
    ({"samples": {"teff": [5800.]}, "max_draws": -1}, "max_draws"),
    ({"samples": {"teff": [5800.]}, "max_draws": True}, "max_draws"),
    ({"samples": {"teff": [5800.]}, "max_draws": 1.5}, "max_draws"),
])
def test_predict_rejects_ambiguous_or_invalid_inputs(options, message):
    fit, config, _ = case(single=True)
    with pytest.raises(ValueError, match=message):
        fit.predict(model=configured(fit, config), **options)


def test_posterior_prediction_custom_callback_and_default_draw_limit():
    fit, _, _ = case(single=True)
    seen = []

    def custom(observation, specmodel):
        return numpyro.sample("x", dist.Normal(0., 1.))

    def callback(observation, specmodel, returned, trace, settings):
        seen.append(returned)
        return {"mean_flux": np.full(observation.shape, returned**2)}

    model = ConfiguredModel(custom, reconstruct_fn=callback)
    samples = {"x": np.arange(101.)}
    result = fit.predict(samples=samples, model=model)
    expected = np.linspace(0, 100, 50, dtype=int)
    np.testing.assert_array_equal(seen, expected)
    np.testing.assert_allclose(result["mean_flux"], np.mean(expected**2))
    assert set(result) == {"mean_flux", "draw_indices"}


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
        model = configured(fit, config)
        result = fit.predict(point={}, model=model)
        assert result["mean_flux"].dtype == np.dtype(dtype)
        diagnostics = fit.residual_diagnostics(point={}, model=model)
        assert len(diagnostics) == 2
        assert all(np.isfinite(row["standardized_rms"]) for row in diagnostics)
        # RMS(residual / sigma) and RMS(residual) / sigma round differently.
        rtol = max(1e-7, 2 * np.finfo(dtype).eps)
        for row, jitter in zip(diagnostics, config["jitter"]):
            np.testing.assert_allclose(row["standardized_rms"],
                                       row["rms"] / np.hypot(dtype(.01), jitter),
                                       rtol=rtol)


@pytest.mark.parametrize("gp", [False, True])
def test_single_model_svi_and_nuts_use_effective_observation(gp):
    fit, config, _ = case(single=True)
    if gp:
        config.update(gp_amplitude=.015, gp_scale=.15)
    extra = np.zeros(fit.observation.shape, bool)
    extra[48] = True
    # A large contaminated pixel must not pull inference away from the truth.
    flux = np.asarray(fit.observation.flux).copy()
    flux[48] = 1e3
    fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
    fit.set_fit_mask(extra)
    estimate = fit.run_svi(rng_key=jax.random.PRNGKey(3), model=configured(fit, config),
                          step_size=.04, num_steps=100, p_initial={"teff": 5600.}, progress_bar=False)
    assert np.isfinite(estimate["teff"])
    assert abs(float(estimate["teff"]) - 5800.) < 180.
    mcmc = fit.run_nuts(rng_key=jax.random.PRNGKey(4), model=configured(fit, config),
                       num_warmup=12, num_samples=12, progress_bar=False,
                       nuts_kwargs={"init_strategy": init_to_value(values=estimate), "max_tree_depth": 3})
    assert isinstance(mcmc, MCMC)
    assert mcmc.get_samples()["teff"].shape == (12,)
    assert np.all(np.isfinite(mcmc.get_samples()["teff"]))
    assert not {"model_flux", "gp_mean", "continuum_flux"}.intersection(mcmc.get_samples())
    with pytest.raises(ValueError, match="jit_model_args"):
        fit.run_nuts(rng_key=jax.random.PRNGKey(0), model=configured(fit, config),
                     mcmc_kwargs={"jit_model_args": True})


@pytest.mark.parametrize("method", ["run_svi", "run_nuts", "predict", "plot_models", "residual_diagnostics"])
def test_inference_and_reconstruction_require_explicit_model(method):
    fit, _, point = case(single=True)
    options = (dict(point=point) if method in ("predict", "plot_models", "residual_diagnostics")
               else dict(rng_key=jax.random.PRNGKey(0)))
    with pytest.raises(TypeError, match="model"):
        getattr(fit, method)(**options)


def test_custom_svi_nuts_callable():
    fit, _, _ = case(single=True)
    fit.set_fit_mask(np.arange(fit.observation.n_pixels) == 48)
    effective = fit.fit_observation()

    def custom(observation, specmodel, *, center):
        np.testing.assert_array_equal(observation.mask, effective.mask)
        assert specmodel is fit.specmodel
        x = numpyro.sample("custom_parameter", dist.Normal(center, 1.))
        numpyro.factor("custom_density", -.5 * (x - center)**2)

    result = fit.run_svi(model=partial(custom, center=2.), rng_key=jax.random.PRNGKey(1),
                         num_steps=30, step_size=.08, progress_bar=False,
                         p_initial={"custom_parameter": 1.})
    assert abs(float(result["custom_parameter"]) - 2.) < .5
    mcmc = fit.run_nuts(model=partial(custom, center=2.), rng_key=jax.random.PRNGKey(2),
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


def clipping_model(fit, *, include_params=True):
    """A custom model with a directly sampled baseline for clipping fixtures."""
    def function(observation, specmodel):
        return numpyro.sample("baseline", dist.Normal(1., .1).expand(observation.shape)
                              .to_event(len(observation.shape)))

    def reconstruct(observation, specmodel, returned, trace, settings):
        prediction = {"continuum_flux": returned, "mean_flux": observation.flux}
        if include_params:
            prediction["params"] = {"components": ({"broadening": {"vsini": 2.}},)}
        return prediction

    return ConfiguredModel(function, reconstruct_fn=reconstruct,
                           observation=fit.observation, specmodel=fit.specmodel)


def test_outliers_model_baseline_replacement_and_fixed_nan():
    fit, baseline = clipping_case()
    model = clipping_model(fit)
    point = {"baseline": baseline}
    # The model's mean_flux absorbs the spike; clipping still uses continuum_flux.
    flags = fit.mask_outliers(point=point, model=model, extend_outlier_mask=False).copy()
    assert flags[10] and not flags[70]
    # Improved baseline fits the first spike but introduces a second discrepancy.
    improved = np.array(fit.observation.flux, copy=True)
    improved[20] -= .2
    fit.set_fit_mask(flags)
    fit.mask_outliers(point={"baseline": improved}, model=model, extend_outlier_mask=False)
    assert fit.fit_mask[20] and not fit.fit_mask[10]
    assert not fit.fit_mask[70]
    fixed = fit.observation.mask.copy()
    fit.reset_fit_mask()
    np.testing.assert_array_equal(fit.observation.mask, fixed)
    assert not fit.fit_mask.any()


def test_outliers_predict_once_before_mask_update_and_plot_same_baseline():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, config, point = case(single=True)
    flux = np.array(fit.observation.flux)
    flux[20] += .2
    fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
    previous = np.zeros_like(fit.fit_mask)
    previous[48] = True
    fit.set_fit_mask(previous)
    predicted = []

    def reconstruct(observation, specmodel, returned, trace, settings):
        np.testing.assert_array_equal(fit.fit_mask, previous)
        np.testing.assert_array_equal(observation.mask, fit.observation.mask | previous)
        prediction = reconstruct_single(observation, specmodel, returned, trace, settings)
        predicted.append(prediction)
        return prediction

    model = ConfiguredModel(model_single, settings=config, reconstruct_fn=reconstruct,
                            observation=fit.observation, specmodel=fit.specmodel)
    flags = fit.mask_outliers(point=point, model=model, plot=True)
    assert len(predicted) == 1
    assert flags[20] and not flags[3]
    fig = plt.gcf()
    baseline = next(curve for curve in fig.axes[0].lines if curve.get_label() == "baseline")
    np.testing.assert_allclose(baseline.get_ydata(),
                               np.asarray(predicted[0]["continuum_flux"])[~fit.observation.mask])
    plt.close(fig)
    with pytest.raises(TypeError, match="prediction"):
        fit.mask_outliers(point=point, model=model, prediction=predicted[0])


def test_extension_exact_runs_and_edges():
    flag = np.zeros(12, bool)
    flag[[0, 4, 5, 11]] = True
    expected = np.zeros(12, bool)
    expected[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11]] = True
    np.testing.assert_array_equal(_extend_mask(flag, 1.), expected)
    np.testing.assert_array_equal(_extend_mask(flag, 0.), flag)
    fit, baseline = clipping_case()
    model = clipping_model(fit, include_params=False)
    point = {"baseline": baseline}
    flags = fit.mask_outliers(point=point, model=model, mask_v=2.)
    assert flags[9:12].all()
    with pytest.raises(ValueError, match="mask_v"):
        fit.mask_outliers(point=point, model=model)
    with pytest.raises(ValueError, match="positive"):
        fit.mask_outliers(point=point, model=model, mask_v=2., sigma_threshold=0.)


def test_masked_filter_poison_and_oversized_kernel():
    fit, baseline = clipping_case()
    point = {"baseline": baseline}
    model = clipping_model(fit)
    flags = fit.mask_outliers(point=point, model=model, extend_outlier_mask=False).copy()
    finite_flux = np.where(fit.observation.mask, 1e10, fit.observation.flux)
    other = SpecFit(replace(fit.observation, flux=finite_flux), fit.specmodel)
    np.testing.assert_array_equal(other.mask_outliers(point=point, model=clipping_model(other),
                                                     extend_outlier_mask=False), flags)
    fit.mask_outliers(point=point, model=model, mask_v=1e6)
    fit.set_fit_mask(np.zeros_like(flags))
    entirely_masked = SpecFit(replace(fit.observation, mask=np.ones_like(flags)), fit.specmodel)
    entirely_masked.mask_outliers(point={"baseline": np.full(flags.shape, np.nan)},
                                  model=clipping_model(entirely_masked))
    assert not entirely_masked.fit_mask.any()


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("extend", [False, True])
def test_outlier_quicklook_matches_clipping_quantities(single, extend, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from scipy.stats import median_abs_deviation, norm

    fit, baseline = clipping_case()
    # Nonuniform errors distinguish the before-clipping median from the median
    # after outlier removal. Fixed masked errors must not enter either scale.
    errors = np.linspace(.005, .1, baseline.size)
    errors[fit.observation.mask] = np.inf
    fit = SpecFit(replace(fit.observation, uncertainty=errors), fit.specmodel)
    if not single:
        multi, _, _ = case()
        obs = fit.observation
        fit = SpecFit(Observation(
            np.stack([obs.wavelength, obs.wavelength + 1000.]),
            np.stack([obs.flux, obs.flux]), np.stack([obs.uncertainty, obs.uncertainty]),
            np.stack([obs.mask, obs.mask]), region=(8, 9)), multi.specmodel)
        baseline = np.stack([baseline, baseline])
    options = dict(point={"baseline": baseline}, model=clipping_model(fit), sigma_threshold=3.,
                   extend_outlier_mask=extend)
    before = set(plt.get_fignums())
    expected = fit.mask_outliers(**options).copy()
    assert set(plt.get_fignums()) == before  # plotting is opt-in
    actual = fit.mask_outliers(**options, plot=True, save_path=tmp_path / "outliers.png")
    np.testing.assert_array_equal(actual, expected)
    assert actual is fit.fit_mask and not actual.flags.writeable
    figures = [plt.figure(n) for n in sorted(set(plt.get_fignums()) - before)]
    assert len(figures) == fit.observation.n_regions
    assert not (tmp_path / "outliers.png").exists()

    def line(ax, label):
        return next(curve for curve in ax.lines if curve.get_label() == label)

    for i, fig in enumerate(figures):
        saved = tmp_path / f"outliers_region{fit.regions[i]}.png"
        assert saved.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        upper, middle, lower, histogram = fig.axes
        wave = np.atleast_2d(fit.observation.wavelength)[i]
        flux = np.atleast_2d(fit.observation.flux)[i]
        fixed = np.atleast_2d(fit.observation.mask)[i]
        flagged = np.atleast_2d(actual)[i]
        model_flux = np.atleast_2d(baseline)[i]
        assert upper.get_title() == f"Region {fit.regions[i]}: outlier check"
        assert [text.get_text() for text in middle.get_legend().get_texts()] == ["median-filter trend"]
        assert [text.get_text() for text in lower.get_legend().get_texts()] == ["clipping threshold"]
        np.testing.assert_allclose(line(upper, "baseline").get_ydata(), model_flux[~fixed])
        residual = flux[~fixed] - model_flux[~fixed]
        np.testing.assert_allclose(line(middle, "data").get_ydata(), residual)
        trend = line(middle, "median-filter trend").get_ydata()[~fixed]
        highpass = residual - trend
        scatter = 1.4826 * median_abs_deviation(highpass)
        standardized = line(lower, "data").get_ydata()
        np.testing.assert_allclose(standardized, highpass / scatter)
        crossed = np.abs(standardized) > 3.
        if extend:
            assert flagged[~fixed][crossed].all()
            assert np.any(flagged[~fixed] & ~crossed)
        else:
            np.testing.assert_array_equal(crossed, flagged[~fixed])
        for ax in (upper, middle, lower):
            np.testing.assert_allclose(line(ax, "fit mask").get_xdata(), wave[flagged])
            np.testing.assert_allclose(line(ax, "data").get_xdata(), wave[~fixed])
        thresholds = [curve.get_ydata() for curve in lower.lines if curve.get_linestyle() == "--"]
        np.testing.assert_allclose(thresholds, [[3., 3.], [-3., -3.]])
        hist_thresholds = [curve.get_xdata() for curve in histogram.lines
                           if curve.get_linestyle() == "--"]
        np.testing.assert_allclose(hist_thresholds, [[3., 3.], [-3., -3.]])
        assert histogram.get_yscale() == "log"
        for patch in histogram.patches:
            selected = flagged[~fixed] if patch.get_label() == "fit mask" else ~flagged[~fixed]
            values, edges, _ = patch.get_data()
            np.testing.assert_array_equal(values, np.histogram(standardized[selected], bins=edges)[0])
            assert values.sum() == selected.sum()
        normal = line(histogram, "N(0, 1)")
        np.testing.assert_allclose(normal.get_ydata(),
                                   norm.pdf(normal.get_xdata()) * (~fixed).sum() * (edges[1] - edges[0]))
        median_error = np.median(np.atleast_2d(fit.observation.uncertainty)[i, ~fixed])
        annotation = histogram.texts[0].get_text()
        assert f"1.4826 × MAD = {scatter:.4g}" in annotation
        assert f"median(error_obs) = {median_error:.4g}" in annotation
        assert f"ratio = {scatter / median_error:.3f}" in annotation
        plt.close(fig)


@pytest.mark.parametrize("all_masked", [False, True])
def test_outlier_quicklook_zero_scale_and_empty_region(all_masked, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, baseline = clipping_case()
    flux = baseline.copy()
    flux[10] += .2  # MAD remains zero even with a spike.
    mask = np.ones_like(flux, dtype=bool) if all_masked else fit.observation.mask
    fit = SpecFit(replace(fit.observation, flux=flux, mask=mask), fit.specmodel)
    flags = fit.mask_outliers(point={"baseline": baseline}, model=clipping_model(fit),
                             plot=True, save_path=tmp_path / "outliers.pdf")
    assert (tmp_path / "outliers_region8.pdf").read_bytes().startswith(b"%PDF")
    fig = plt.gcf()
    lower = fig.axes[2]
    histogram = fig.axes[3]
    assert not any(curve.get_label() == "N(0, 1)" for curve in histogram.lines)
    if all_masked:
        assert not flags.any()
        assert any(text.get_text() == "No usable pixels" for text in lower.texts)
        assert any(text.get_text() == "No usable pixels" for text in histogram.texts)
    else:
        assert flags[10]
        assert lower.get_ylabel() == "Detrended residual"
        assert histogram.get_xlabel() == "Detrended residual"
        assert any("MAD scale = 0" in text.get_text() for text in lower.texts)
        assert "ratio = 0.000" in histogram.texts[0].get_text()
        for curve in lower.lines:
            assert np.all(np.isfinite(curve.get_ydata()))
    plt.close(fig)


@pytest.mark.parametrize("filename", ["outliers.pdf", "outliers"])
def test_outlier_save_without_display_and_escaped_region(filename, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, baseline = clipping_case()
    region = ("blue/UV",)
    spectra = replace(fit.specmodel.spectra, regions=region)
    fit = SpecFit(replace(fit.observation, region=region), SpecModel(spectra, vmax=30.))
    options = dict(point={"baseline": baseline}, model=clipping_model(fit))
    expected = fit.mask_outliers(**options).copy()
    existing = plt.figure()
    before = set(plt.get_fignums())
    try:
        with plt.rc_context({"savefig.format": "pdf"}):
            actual = fit.mask_outliers(**options, save_path=tmp_path / filename)
        assert actual is fit.fit_mask
        np.testing.assert_array_equal(actual, expected)
        assert set(plt.get_fignums()) == before
        assert (tmp_path / "outliers_regionblue%2FUV.pdf").read_bytes().startswith(b"%PDF")
        assert not (tmp_path / filename).exists()
    finally:
        plt.close(existing)


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

    seen = []

    def predict(point, observation, specmodel):
        np.testing.assert_array_equal(observation.mask, fit.effective_mask)
        assert specmodel is fit.specmodel
        seen.append(point)
        return point

    model = SimpleNamespace(predict=predict)
    point = {"mean_flux": np.ones(8), "continuum_flux": np.zeros(8), "jitter": 1.}
    stats = fit.residual_diagnostics(point=point, model=model)[0]
    assert len(seen) == 1 and seen[0] is point
    valid_res = np.array([0., 1., 2., 4., 6., 12.])
    assert stats["region"] == 8
    assert stats["n_usable"] == 6
    np.testing.assert_allclose(stats["rms"], np.sqrt(np.mean(valid_res**2)))
    np.testing.assert_allclose(stats["mad_scale"], 1.4826 * np.median(np.abs(valid_res - np.median(valid_res))))
    np.testing.assert_allclose(stats["standardized_rms"], stats["rms"] / np.sqrt(2))
    assert stats["n_gt_3sigma"] == 2 and stats["n_gt_5sigma"] == 1
    assert stats["n_adjacent_pairs"] == 3
    np.testing.assert_allclose(stats["lag1_correlation"], np.corrcoef([0., 1., 4.], [1., 2., 6.])[0, 1])
    continuum = fit.residual_diagnostics(point=point, model=model, baseline="continuum_flux")[0]
    np.testing.assert_allclose(continuum["rms"], np.sqrt(np.mean((valid_res + 1)**2)))
    no_jitter = fit.residual_diagnostics(point={"mean_flux": np.ones(8)}, model=model)[0]
    np.testing.assert_allclose(no_jitter["standardized_rms"], stats["rms"])
    with pytest.raises(ValueError, match="baseline"):
        fit.residual_diagnostics(point=point, model=model, baseline="physical_flux")
    with pytest.raises(ValueError, match="containing continuum_flux"):
        fit.residual_diagnostics(point={"mean_flux": np.ones(8)}, model=model,
                                 baseline="continuum_flux")
    with pytest.raises(ValueError, match="prediction must have shape"):
        fit.residual_diagnostics(point={"mean_flux": np.ones(7)}, model=model)
    fit.set_fit_mask(np.ones(8, bool))
    empty = fit.residual_diagnostics(point={"mean_flux": np.full(8, np.nan)}, model=model)[0]
    assert empty["n_usable"] == 0 and empty["n_gt_3sigma"] == 0
    assert np.isnan(empty["rms"]) and np.isnan(empty["lag1_correlation"])
    with pytest.raises(ValueError, match="jitter"):
        fit.residual_diagnostics(point={**point, "jitter": -1.}, model=model)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("label_observation", [False, True])
def test_plot_smoke(single, label_observation, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, config, point = case(single)
    if not label_observation:
        fit = SpecFit(replace(fit.observation, region=None), fit.specmodel)
    assert fit.regions == fit.specmodel.spectra.regions
    flags = np.zeros(fit.observation.shape, bool)
    flags[..., 48] = True
    fit.set_fit_mask(flags)
    model = configured(fit, config)
    plots = fit.plot_models(point=point, model=model, save_path=tmp_path / "models.png")
    assert isinstance(plots, list) and len(plots) == fit.observation.n_regions
    assert len({id(fig) for fig, _ in plots}) == len(plots)
    for region, (fig, axes) in zip(fit.regions, plots):
        assert axes.shape == (2,)
        upper, lower = axes
        assert upper.get_title() == f"Region {region}"
        assert upper.get_shared_x_axes().joined(upper, lower)
        assert not upper.get_xticklabels()
        assert lower.get_xlabel() == "Wavelength [Å]"
        assert 0 <= upper.get_position().y0 - lower.get_position().y1 < .02
        assert upper.get_legend() is None and lower.get_legend() is None
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "data", "fixed mask", "fit mask", "model\n(incl. continuum)"]
        fig.canvas.draw()
        assert fig.legends[0].get_window_extent().x0 > upper.get_window_extent().x1
        assert (tmp_path / f"models_region{region}.png").stat().st_size > 0
        plt.close(fig)
    assert not (tmp_path / "models.png").exists()
    if len(plots) > 1:
        assert not plots[0][1][0].get_shared_x_axes().joined(plots[0][1][0], plots[1][1][0])
    stats = fit.residual_diagnostics(point=point, model=model)
    assert tuple(row["region"] for row in stats) == fit.regions


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
    [(fig, axes)] = fit.plot_models(point=point, model=configured(fit, config))
    assert axes[0].get_ylim()[0] > 0 and axes[0].get_ylim()[1] < 2
    assert axes[1].get_ylim()[1] < 1
    plt.close(fig)


@pytest.mark.parametrize("masked_pixel", [3, 48])
def test_plot_limits_include_model_on_masked_pixels(masked_pixel):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, _, _ = case(single=True)
    flux = np.asarray(fit.observation.flux).copy()
    flux[3], flux[48] = 1e8, -1e8
    fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
    flags = np.zeros(fit.observation.shape, bool)
    flags[48] = True
    fit.set_fit_mask(flags)
    baseline = np.ones(fit.observation.shape)
    baseline[masked_pixel] = .1  # A line core inside either kind of mask.
    baseline[48 if masked_pixel == 3 else 3] = np.nan
    model = SimpleNamespace(predict=lambda point, observation, specmodel: point)
    [(fig, axes)] = fit.plot_models(point={"continuum_flux": baseline}, model=model)
    try:
        low, high = axes[0].get_ylim()
        assert 0 < low < .1
        assert 1 < high < 2
        assert axes[1].get_ylim()[1] < 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("show_gp", [False, True])
@pytest.mark.parametrize("posterior", [False, True])
def test_plot_uses_model_prediction_and_baseline_residuals(show_gp, posterior):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, config, point = case(single=True)
    config.update(gp_amplitude=.02, gp_scale=.15)
    model = configured(fit, config)
    options = dict(samples={"teff": np.array([5500., 6000.])}) if posterior else dict(point=point)
    prediction = fit.predict(model=model, **options)
    [(fig, axes)] = fit.plot_models(model=model, show_gp=show_gp, **options)
    upper, lower = axes
    assert len(upper.lines) == 3  # Data, fixed mask and physical model; no empty fit-mask series.
    expected_labels = (["data", "fixed mask", "model\n(incl. continuum)"]
                       + (["GP mean"] if show_gp else []))
    assert [text.get_text() for text in fig.legends[0].get_texts()] == expected_labels
    assert not upper.collections
    assert not lower.collections
    assert upper.get_legend() is None and lower.get_legend() is None
    baseline = next(line for line in upper.lines if line.get_label() == "model\n(incl. continuum)")
    assert lower.lines[-1].get_color() == baseline.get_color() == "C0"
    np.testing.assert_array_equal(lower.lines[-1].get_ydata(), [0, 0])
    np.testing.assert_allclose(baseline.get_ydata(), prediction["continuum_flux"])
    valid = ~fit.effective_mask
    np.testing.assert_allclose(lower.lines[0].get_ydata(),
                               (np.asarray(fit.observation.flux) - np.asarray(prediction["continuum_flux"]))[valid])
    gp_lines = [line for line in lower.lines if line.get_label() == "GP mean"]
    assert len(gp_lines) == int(show_gp)
    if show_gp:
        np.testing.assert_allclose(gp_lines[0].get_ydata(),
                                   prediction["mean_flux"] - prediction["continuum_flux"])
    plt.close(fig)
    with pytest.raises(TypeError, match="prediction"):
        fit.plot_models(prediction=prediction, model=model)


@pytest.mark.parametrize("show_gp", [False, True])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("max_draws,mean,bounds", [
    (1, 0., (0., 0.)),
    (2, 1., (.32, 1.68)),
    (None, 34., (.64, 68.64)),
])
def test_posterior_band_uses_selected_draws(single, max_draws, mean, bounds, show_gp):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, _, _ = case(single)
    scale = np.ones(fit.observation.shape)
    if not single:
        scale[1] *= 2
    scale[..., 3] *= 10  # The limits must also include the band at masked pixels.

    def predict(point, observation, specmodel):
        curve = 1 + point["level"] * scale
        gp = .5 - 2 * point["level"] * scale
        # GP and continuum vary together; subtract within each draw. A custom
        # callback need not return a separate gp_mean for plotting its band.
        return {"continuum_flux": curve, "mean_flux": curve + gp}

    model = SimpleNamespace(predict=predict)
    # With max_draws=2, the large middle draw is excluded from both summaries.
    options = dict(samples={"level": np.array([0., 100., 2.])}, model=model,
                   max_draws=max_draws)
    prediction = fit.predict(**options)
    expected = np.stack([1 + value * scale for value in bounds])
    expected_gp = np.stack([.5 - 2 * value * scale for value in bounds[::-1]])
    assert prediction["continuum_flux_quantiles"].shape == (2, *fit.observation.shape)
    np.testing.assert_allclose(prediction["continuum_flux_quantiles"], expected)
    np.testing.assert_allclose(prediction["gp_mean_quantiles"], expected_gp)
    np.testing.assert_allclose(prediction["continuum_flux"], 1 + mean * scale)
    plots = fit.plot_models(show_gp=show_gp, show_interval=True, **options)
    try:
        for i, (fig, (upper, lower)) in enumerate(plots):
            bands = [(upper, expected)] + ([(lower, expected_gp)] if show_gp else [])
            for axis, quantiles in bands:
                band = quantiles if single else quantiles[:, i]
                vertices = axis.collections[0].get_paths()[0].vertices
                np.testing.assert_allclose([vertices[:, 1].min(), vertices[:, 1].max()],
                                           [band.min(), band.max()])
                low, high = axis.get_ylim()
                assert low < band.min() and high > band.max()
            assert len(lower.collections) == int(show_gp)
    finally:
        for fig, _ in plots:
            plt.close(fig)


def test_plot_without_masks_and_filename_extension(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, config, point = case(single=True)
    spectra = replace(fit.specmodel.spectra, regions=("blue/UV",))
    specmodel = SpecModel(spectra, vmax=30.)
    obs = replace(fit.observation, flux=np.nan_to_num(fit.observation.flux, nan=1.),
                  uncertainty=np.full(fit.observation.shape, .01),
                  mask=np.zeros(fit.observation.shape, bool), region=("blue/UV",))
    fit = SpecFit(obs, specmodel)
    with plt.rc_context({"savefig.format": "pdf"}):
        [(fig, axes)] = fit.plot_models(point=point, model=configured(fit, config),
                                        save_path=tmp_path / "models")
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["data", "model\n(incl. continuum)"]
    assert (tmp_path / "models_regionblue%2FUV.pdf").stat().st_size > 0
    plt.close(fig)
