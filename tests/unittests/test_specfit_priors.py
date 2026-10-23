"""Grid-aware default priors used by the real single-star model."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.util import log_density
import pytest

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (Observation, SpecModel, SpecFit, model_single,
                             single_star_params, physical_logg_max, make_model_single, ConfiguredModel)
from jaxstar.specfit._data import _SpectralLibrary


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def make_fit(library="coelho", single=False, *, axis_overrides=None):
    axes = dict(teff=[5000., 6500.], logg=[3.5, 5.])
    if library == "coelho":
        axes.update(feh=[-.5, .5], alpha=[0.])
    elif library == "bosz":
        axes.update(mh=[-.5, .5], alpha=[0.], carbon=[0.], vmic=[.5, 2.5])
    elif library == "tlusty":
        axes.update(teff=[15000., 30000.], logZ=[-1., 0.])
    else:
        axes = dict(custom_temperature=[.2, .9], abundance=[0.])
    axes.update(axis_overrides or {})
    axes = {name: np.asarray(values) for name, values in axes.items()}
    count = 1 if single else 2
    centers = np.array([5000., 6000.])[:count, None]
    velocities = np.linspace(-250., 250., 501)
    wave = centers * np.exp(velocities / 299792.458)
    profile = np.exp(-.5 * (velocities / 4.)**2)
    shape = tuple(len(a) for a in axes.values())
    flux = np.broadcast_to(np.stack([1 - d * profile for d in np.linspace(.2, .5, shape[0])]).reshape(
        (shape[0],) + (1,) * (len(axes) - 1) + (1, 501)), shape + (count, 501))
    grid = RectilinearGrid(axes=axes, fields={"flux": Field(
        flux, tuple(axes), payload_dims=("region", "pixel"))})
    model = SpecModel(_SpectralLibrary(grid, wave, (8, 9)[:count], library), vmax=30.)
    observed = centers * np.exp(np.linspace(-120.37, 120.19, 64) / 299792.458)
    atmosphere = {name: float(np.mean(a)) for name, a in axes.items()}
    physical = single_star_params(atmosphere, vsini=6., vmacro=3., q1=.36, q2=.3,
                                  rv=12., resolving_power=70000.)
    flux = np.asarray(model(physical, observed))
    if single:
        observed, flux = observed[0], flux[0]
    obs = Observation(observed, flux, np.full_like(flux, .01), region=(8, 9)[:count])
    fit = SpecFit(obs, model)
    fit.check_ccf(atmosphere, v_limit=50., ccfvmax=40.)
    return fit


def trace(fit, model, seed=0, prior_only=False):
    if prior_only:
        return model.prior_trace(rng_key=jax.random.PRNGKey(seed))
    return handlers.trace(handlers.seed(model, seed)).get_trace(fit.fit_observation(), fit.specmodel)


@pytest.mark.parametrize("library", ["coelho", "bosz", "tlusty", "custom"])
def test_default_axes_singletons_and_native_model_sites(library):
    fit = make_fit(library)
    model = make_model_single(fit, resolving_power=70000., use_gp=False)
    result = trace(fit, model)
    axes = fit.specmodel.spectra.grid
    for name in axes.axis_names:
        if name == "vmic" and library == "bosz":
            assert result[name]["type"] == "deterministic"
            continue
        values = np.asarray(axes.axis(name))
        site = result[name]
        if len(values) == 1:
            assert site["type"] == "deterministic" and site["value"] == values[0]
        else:
            assert isinstance(site["fn"], dist.Uniform)
            np.testing.assert_array_equal([site["fn"].low, site["fn"].high], values[[0, -1]])
    assert result["rv"]["fn"].event_shape == (fit.observation.n_regions,)
    assert np.isfinite(result["spectrum"]["fn"].log_factor)
    assert not {"gp_amplitude", "gp_scale"}.intersection(result)


@pytest.mark.parametrize("single", [False, True])
def test_gp_defaults_reconstruct_and_finite_density_gradient(single):
    fit = make_fit(single=single)
    model = make_model_single(fit, resolving_power=70000., degree=2, gp_solver="quasisep")
    result = trace(fit, model)
    for name, limits in (("jitter", (-10., -3.)), ("gp_amplitude", (-5., -.5)),
                         ("gp_scale", (-5., 2.))):
        prior = result[name]["fn"]
        np.testing.assert_allclose(np.log([prior.low, prior.high]), limits)
    point = {name: site["value"] for name, site in result.items()
             if site["type"] == "sample" and not site["is_observed"]}
    prediction = fit.predict(point=point, model=model)
    assert prediction["mean_flux"].shape == fit.observation.shape
    assert np.isfinite(prediction["mean_flux"]).all()
    assert prediction["continuum_posterior"].mean.shape[-1] == 3
    value, gradient = jax.value_and_grad(lambda p: log_density(
        model, (fit.fit_observation(), fit.specmodel), {}, p)[0])(point)
    assert np.isfinite(value)
    assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(gradient))


def test_vecchia_configured_model_likelihood_and_reconstruction_boundary():
    from jaxstar.specfit import gp_marginalized_continuum_log_likelihood
    fit = make_fit(single=True)
    model = make_model_single(fit, resolving_power=70000., degree=2, gp_solver="vecchia1")
    result = trace(fit, model)
    params = handlers.seed(model, 0)(fit.fit_observation(), fit.specmodel)
    flux = fit.specmodel(params, fit.observation.wavelength)
    options = {name: result[name]["value"] for name in
               ("gp_amplitude", "gp_scale", "jitter", "sigma_constant", "sigma_continuum")}
    expected = gp_marginalized_continuum_log_likelihood(
        fit.fit_observation(), flux, degree=2, gp_solver="vecchia1", **options)
    np.testing.assert_allclose(result["spectrum"]["fn"].log_factor, expected, atol=1e-10)
    assert model.settings["gp_solver"] == "vecchia1"
    point = {name: site["value"] for name, site in result.items()
             if site["type"] == "sample" and not site["is_observed"]}
    with pytest.raises(ValueError, match="does not define a latent GP conditional mean"):
        fit.predict(point=point, model=model)


@pytest.mark.parametrize("width,half_width", [(12., 5.), (20., 10.), (80., 40.)])
def test_ccf_rv_without_clipping_and_prior_inspection_skips_spectrum(width, half_width):
    fit = make_fit()
    center = float(np.max(fit.rv_bounds[:, 1])) + 100.
    fit.ccf.update(rv=center, width=width)
    model = make_model_single(fit, resolving_power=np.array([70000., 80000.]))
    report = model.check_prior()
    rv_prior = model.prior_trace()["rv"]["fn"].base_dist
    np.testing.assert_allclose(rv_prior.low, [center - half_width] * 2)
    np.testing.assert_allclose(rv_prior.high, [center + half_width] * 2)
    assert list(report.columns) == ["prior"]
    assert report.loc["rv", "prior"].startswith("Uniform(")
    assert "shape=(2,)" in report.loc["rv", "prior"]
    support = np.min(np.asarray(fit.specmodel.velocity_grid[:, -1]))
    np.testing.assert_allclose(model.settings["vsini_max"], min(width, .9 * support))
    assert "spectrum" not in model.prior_trace()
    assert model.settings["degree"] == 4
    with pytest.raises(Exception, match="coverage"):
        trace(fit, model)


@pytest.mark.parametrize("use_vmic", [None, True, False])
def test_empirical_switches_and_conditional_display(use_vmic):
    fit = make_fit("bosz")
    model = make_model_single(fit, resolving_power=70000., use_gp=False,
        use_empirical_vmic=use_vmic, use_empirical_vmacro=True,
        vmacro_empirical_sigma=.7, use_physical_logg_max=True)
    result = trace(fit, model)
    assert result["vmic"]["type"] == ("sample" if use_vmic is False else "deterministic")
    np.testing.assert_allclose(result["logg"]["fn"].high,
                               physical_logg_max(result["teff"]["value"]))
    report = model.check_prior()
    assert list(report.columns) == ["prior"]
    assert report.loc["logg | teff", "prior"] == "Uniform(logg_min, physical_logg_max(teff))"
    assert report.loc["vmacro | teff", "prior"] == "TruncatedNormal(μ(teff), σ=0.7, low=0)"
    assert report.equals(model.check_prior(rng_key=jax.random.PRNGKey(9)))
    conditional = model.check_prior(condition={"teff": 5800.})
    assert conditional.loc["logg | teff", "prior"] == report.loc["logg | teff", "prior"]
    upper = float(physical_logg_max(5800.))
    assert conditional.loc["logg | teff", "conditional"] == f"teff=5800: Uniform(3.5, {upper:.6g})"
    assert "TruncatedNormal(loc=" in conditional.loc["vmacro | teff", "conditional"]
    assert "scale=0.7" in conditional.loc["vmacro | teff", "conditional"]
    if use_vmic is not False:
        label = "vmic | teff, logg, mh"
        assert report.loc[label, "prior"] == "empirical_vmic(teff, logg, mh)"
        assert conditional.loc[label, "conditional"] == ""
        conditional = model.check_prior(condition={"teff": 5800., "logg": 4., "mh": 0.})
        assert conditional.loc[label, "conditional"].startswith("teff=5800, logg=4, mh=0: ")
    assert np.isfinite(result["spectrum"]["fn"].log_factor)
    with pytest.raises(ValueError, match="outside its prior support"):
        model.check_prior(condition={"teff": 10000.})
    with pytest.raises(ValueError, match="sampled parameter"):
        model.check_prior(condition={"typo": 1.})


def test_overrides_fixed_values_and_settings_snapshots():
    fit = make_fit()
    resolution = np.array([70000., 80000.])
    bounds = {"teff": (5400., 6100.)}
    fixed = {"logg": 4.4, "vmacro": 2., "jitter": .001}
    model = make_model_single(fit, resolving_power=resolution, atmosphere_bounds=bounds,
                              rv_bounds=(-3., 3.), vsini_max=10., fixed=fixed, use_gp=False)
    fixed["logg"], bounds["teff"], resolution[0] = 4., (5000., 5500.), 90000.
    report = model.check_prior()
    assert report.loc["logg", "prior"] == "4.4 (fixed)"
    assert report.loc["teff", "prior"] == "Uniform(5400, 6100)"
    np.testing.assert_array_equal(model.settings["resolving_power"], [70000., 80000.])
    np.testing.assert_array_equal(model.settings["rv_bounds"][0], [-3., -3.])
    with pytest.raises(TypeError):
        model.settings["degree"] = 5
    with pytest.raises(TypeError):
        model.settings["fixed"]["logg"] = 3.
    with pytest.raises(ValueError):
        model.settings["resolving_power"][0] = 90000.
    other = SpecFit(fit.observation, SpecModel(fit.specmodel.spectra))
    with pytest.raises(ValueError, match="different SpecModel"):
        model(other.observation, other.specmodel)


def test_explicit_defaults_need_no_ccf_and_resolution_can_be_sampled():
    fit = make_fit()
    fit.ccf = None
    with pytest.raises(ValueError, match="run check_ccf"):
        make_model_single(fit, resolving_power=70000.)
    model = make_model_single(fit, resolving_power=70000., rv_bounds=(5., 15.), vsini_max=10.)
    assert model.check_prior().loc["vsini", "prior"] == "Uniform(0, 10)"
    for prior in (dist.Uniform(65000., 75000.),
                  dist.Uniform(jnp.array([65000., 75000.]), jnp.array([75000., 85000.])).to_event(1)):
        model = make_model_single(fit, resolving_power=prior, fixed={"rv": 12., "vsini": 6.})
        assert trace(fit, model)["resolving_power"]["type"] == "sample"


@pytest.mark.parametrize("options,message", [
    ({"atmosphere_bounds": {"unknown": (0., 1.)}}, "unknown atmosphere"),
    ({"atmosphere_bounds": {"teff": (4000., 6000.)}}, "outside the loaded grid"),
    ({"rv_bounds": (10., 0.)}, "ordered bounds"),
    ({"fixed": {"teff": 4000.}}, "outside its atmosphere bounds"),
    ({"fixed": {"logg": [4., 4.]}}, "shape"),
    ({"fixed": {"q1": 2.}}, "in \\[0, 1\\]"),
    ({"fixed": {"typo": 1.}}, "unknown fixed"),
    ({"vsini_max": 100.}, "broadening support"),
    ({"vmacro_bounds": (-1., 10.)}, "nonnegative"),
    ({"jitter_bounds": (0., 1.)}, "positive"),
    ({"gp_solver": "unknown"}, "gp_solver"),
    ({"use_empirical_vmic": True}, "empirical vmic requires"),
    ({"use_empirical_vmacro": True, "fixed": {"vmacro": 3.}}, "omit fixed vmacro"),
    ({"use_empirical_vmacro": True, "vmacro_empirical_sigma": 0.}, "sigma"),
    ({"use_gp": False, "fixed": {"gp_scale": 1.}}, "GP parameters conflict"),
])
def test_invalid_settings(options, message):
    fit = make_fit()
    with pytest.raises((ValueError, TypeError), match=message):
        make_model_single(fit, resolving_power=70000., **options)


@pytest.mark.parametrize("teff,expected", [
    ([3500., 8000.], [4500., 7000.]), ([5000., 8000.], [5000., 7000.]),
    ([3500., 6500.], [4500., 6500.]), ([5000., 6500.], [5000., 6500.]),
])
def test_physical_gravity_restricts_temperature(teff, expected):
    fit = make_fit(axis_overrides=dict(teff=teff))
    model = make_model_single(fit, resolving_power=70000., use_gp=False, use_physical_logg_max=True)
    np.testing.assert_array_equal(model.settings["atmosphere_bounds"]["teff"], expected)
    assert np.isfinite(trace(fit, model)["spectrum"]["fn"].log_factor)
    unrestricted = make_model_single(fit, resolving_power=70000., use_gp=False)
    np.testing.assert_array_equal(unrestricted.settings["atmosphere_bounds"]["teff"], teff)


@pytest.mark.parametrize("axes,error", [
    (dict(teff=[7500., 8000.]), "temperature range"),
    (dict(teff=[7000., 8000.]), "temperature range"),
    (dict(teff=[4000.]), "temperature range"),
    (dict(logg=[3.5, 4.5]), "outside the loaded logg grid"),
    (dict(logg=[4.8, 5.]), "outside the loaded logg grid"),
    (dict(logg=[4.8]), "fixed logg exceeds"),
])
def test_physical_gravity_rejects_incompatible_support(axes, error):
    fit = make_fit(axis_overrides=axes)
    with pytest.raises(ValueError, match=error):
        make_model_single(fit, resolving_power=70000., use_physical_logg_max=True)


@pytest.mark.parametrize("axes", [dict(teff=[4500.]), dict(teff=[7000.]),
                                  dict(logg=[4.]), dict(teff=[5800.], logg=[4.])])
def test_physical_gravity_preserves_compatible_fixed_coordinates(axes):
    fit = make_fit(axis_overrides=axes)
    model = make_model_single(fit, resolving_power=70000., use_gp=False, use_physical_logg_max=True)
    result = trace(fit, model)
    for name, values in axes.items():
        assert result[name]["type"] == "deterministic"
        assert result[name]["value"] == values[0]
    assert np.isfinite(result["spectrum"]["fn"].log_factor)


def test_custom_broadening_and_reconstruction_callback_use_current_mask():
    import numpyro
    fit = make_fit()
    original_model = fit.specmodel

    def identity(wavelength, flux, **kwargs):
        return wavelength, flux

    fit = SpecFit(fit.observation, SpecModel(original_model.spectra,
                                            broadening_operator=identity, vmax=30.))
    fit.ccf = dict(rv=0., width=80.)
    model = make_model_single(fit, resolving_power=np.inf, use_gp=False)
    assert model.settings["vsini_max"] == 80.
    assert np.isfinite(trace(fit, model)["spectrum"]["fn"].log_factor)

    # A sampled coefficient and custom reconstruction need no marginalized-
    # continuum sites. The callback receives the actual return value and trace.
    def custom(observation, specmodel, *, degree, prior_only=False):
        c = numpyro.sample("coefficient", dist.Normal(1., .1))
        return {"coefficient": c}

    def reconstruct(obs, specmodel, params, sites, settings):
        assert specmodel is fit.specmodel
        assert settings["degree"] == 0
        assert params["coefficient"] == sites["coefficient"]["value"]
        return {"mean_flux": np.full(obs.shape, float(params["coefficient"])), "mask": obs.mask}

    custom_model = ConfiguredModel(custom, settings={"degree": 0}, reconstruct_fn=reconstruct,
                                    observation=fit.observation, specmodel=fit.specmodel)
    before = fit.predict(point={"coefficient": 1.2}, model=custom_model)
    mask = np.zeros(fit.observation.shape, bool)
    mask[:, 5] = True
    fit.set_fit_mask(mask)
    after = fit.predict(point={"coefficient": 1.2}, model=custom_model)
    np.testing.assert_array_equal(before["mean_flux"], after["mean_flux"])
    assert np.all(np.asarray(after["mask"])[:, 5]) and not np.any(np.asarray(before["mask"])[:, 5])
    with pytest.raises(ValueError, match="latent site 'coefficient'"):
        fit.predict(point={}, model=custom_model)
    assert custom_model.check_prior().loc["coefficient", "prior"] == "Normal(1, 0.1)"
