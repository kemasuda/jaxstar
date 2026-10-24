"""Mixture prefit initialization, conditional priors and public workflow."""

from dataclasses import replace

import jax
import numpy as np
import numpyro.distributions as dist
from numpyro.infer.util import log_density
import pytest

from jaxstar.specfit import make_model_single, physical_logg_max
from test_specfit import case
from test_specfit_priors import make_fit


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def options(**overrides):
    return dict(resolving_power=70000., degree=2,
                fixed=dict(rv=12., vsini=6., vmacro=3., q1=.36, q2=.3),
                **overrides)


def capture_svi(monkeypatch, fit):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return kwargs["p_initial"]

    monkeypatch.setattr(fit, "run_svi", run)
    return calls


@pytest.mark.parametrize("layout", ["1d", "one_row", "two_rows"])
def test_defaults_shapes_reproducibility_and_unchanged_masks(layout, monkeypatch):
    fit, _, _ = case(single=layout != "two_rows")
    if layout == "one_row":
        fit.observation = replace(fit.observation, **{
            name: getattr(fit.observation, name)[None, :]
            for name in ("wavelength", "flux", "uncertainty", "mask")})
    mask = np.zeros(fit.observation.shape, bool)
    mask[..., 10] = True
    fit.set_fit_mask(mask)
    before = fit.effective_mask.copy()
    settings = options()
    calls = capture_svi(monkeypatch, fit)

    model, point = fit.run_mixture_prefit(
        rng_key=jax.random.PRNGKey(4), model_kwargs=settings,
        step_size=.02, num_steps=37, progress_bar=False)
    assert point is calls[0]["p_initial"]
    assert model is calls[0]["model"]
    assert calls[0]["step_size"] == .02 and calls[0]["num_steps"] == 37
    assert calls[0]["progress_bar"] is False
    assert set(point) == {"teff", "jitter", "mismatch_fraction", "mismatch_scale",
                          "continuum_coefficients"}
    assert 5000. < point["teff"] < 6500.
    np.testing.assert_allclose(point["jitter"], .002)
    np.testing.assert_allclose(point["mismatch_fraction"], .01)
    np.testing.assert_allclose(point["mismatch_scale"], .1)
    expected = np.zeros(fit.observation.shape[:-1] + (3,))
    expected[..., 0] = 1.
    np.testing.assert_array_equal(point["continuum_coefficients"], expected)
    assert model.settings["likelihood"] == "gaussian_mixture"
    assert model.settings["use_gp"] is False
    prior = model.prior_trace()
    np.testing.assert_allclose(prior["mismatch_fraction"]["fn"].mean, .1)
    np.testing.assert_allclose(prior["mismatch_scale"]["fn"].scale, .1)
    assert "likelihood" not in settings and "mismatch_fraction" not in settings

    _, again = fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(4), model_kwargs=settings)
    _, different = fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(5), model_kwargs=settings)
    for name in point:
        np.testing.assert_array_equal(again[name], point[name])
    assert different["teff"] != point["teff"]
    prediction = fit.predict(model=model, point=point)
    assert np.isfinite(prediction["mismatch_probability"][mask]).all()
    assert np.isnan(prediction["mismatch_probability"][fit.observation.mask]).all()
    np.testing.assert_array_equal(fit.fit_mask, mask)
    np.testing.assert_array_equal(fit.effective_mask, before)


def test_user_values_override_defaults_and_conditional_prior_draws(monkeypatch):
    fit = make_fit()
    capture_svi(monkeypatch, fit)
    initial = dict(teff=6400., jitter=.004, mismatch_fraction=.03,
                   mismatch_scale=.15, continuum_coefficients=np.tile([1.02, -.01], (2, 1)))
    settings = dict(resolving_power=70000., degree=1, use_physical_logg_max=True,
                    use_empirical_vmacro=True, mismatch_scale=dist.HalfNormal(.2))
    key = jax.random.PRNGKey(9)
    model, point = fit.run_mixture_prefit(rng_key=key, model_kwargs=settings, p_initial=initial)
    for name, value in initial.items():
        np.testing.assert_array_equal(point[name], value)
    assert 3.5 < point["logg"] < physical_logg_max(initial["teff"])
    expected = model.prior_trace(rng_key=jax.random.split(key)[0], condition=initial)
    for name in ("logg", "vmacro", "feh", "rv"):
        np.testing.assert_array_equal(point[name], expected[name]["value"])
    assert set(point) == {name for name, site in expected.items()
                          if site["type"] == "sample" and not site["is_observed"]}
    assert np.isfinite(log_density(model, (fit.fit_observation(), fit.specmodel), {}, point)[0])
    np.testing.assert_array_equal(initial["continuum_coefficients"], np.tile([1.02, -.01], (2, 1)))


def test_fixed_noise_and_mixture_parameters_stay_fixed(monkeypatch):
    fit, _, _ = case(single=True)
    calls = capture_svi(monkeypatch, fit)
    settings = options(mismatch_fraction=0., mismatch_scale=0.)
    settings["fixed"].update(jitter=0., teff=5800.)
    model, point = fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(2), model_kwargs=settings,
                                         p_initial={"continuum_coefficients": [1., 0., 0.]})
    assert set(calls[0]["p_initial"]) == {"continuum_coefficients"}
    prediction = fit.predict(model=model, point=point)
    for name in ("jitter", "mismatch_fraction", "mismatch_scale"):
        assert prediction[name] == 0.
    np.testing.assert_array_equal(prediction["mismatch_probability"][fit.observation.valid], 0.)


def test_custom_priors_excluding_defaults_require_explicit_initial_values(monkeypatch):
    fit, _, _ = case(single=True)
    calls = capture_svi(monkeypatch, fit)
    settings = options(jitter_bounds=(.005, .02), mismatch_fraction=dist.Uniform(.1, .2),
                       mismatch_scale=dist.Uniform(.2, .3))
    with pytest.raises(ValueError, match="outside its prior support"):
        fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(0), model_kwargs=settings)
    assert not calls
    values = dict(jitter=.006, mismatch_fraction=.15, mismatch_scale=.25)
    _, point = fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(0),
                                      model_kwargs=settings, p_initial=values)
    for name, value in values.items():
        np.testing.assert_allclose(point[name], value)


@pytest.mark.parametrize("initial,message", [
    ({"typo": 1.}, "sampled parameter"),
    ({"rv": 12.}, "sampled parameter"),
    ({"teff": [5800.]}, "must be scalar"),
    ({"teff": 7000.}, "outside its prior support"),
    ({"continuum_coefficients": [1., 0.]}, "wrong shape"),
    ({"continuum_coefficients": [1., np.inf, 0.]}, "finite"),
    ({"mismatch_fraction": np.nan}, "mismatch_fraction"),
    ({"mismatch_fraction": 0.}, "unconstrained coordinates"),
    ({"mismatch_fraction": 1.}, "unconstrained coordinates"),
    ({"mismatch_scale": 0.}, "outside its prior support"),
    ({"jitter": .001}, "unconstrained coordinates"),
    ({"jitter": .05}, "unconstrained coordinates"),
    ({"teff": 5000.}, "unconstrained coordinates"),
])
def test_invalid_initial_values_fail_before_svi(initial, message, monkeypatch):
    fit, _, _ = case(single=True)
    calls = capture_svi(monkeypatch, fit)
    mask = fit.effective_mask.copy()
    with pytest.raises(ValueError, match=message):
        fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(0), model_kwargs=options(), p_initial=initial)
    assert not calls
    np.testing.assert_array_equal(fit.effective_mask, mask)


@pytest.mark.parametrize("changes,message", [
    ({"use_gp": True}, "requires use_gp=False"),
    ({"likelihood": "gaussian"}, "requires likelihood='gaussian_mixture'"),
    ({"fixed": {"rv": 12., "vsini": 6., "gp_amplitude": .01}}, "requires use_gp=False"),
])
def test_conflicting_model_settings_are_rejected(changes, message, monkeypatch):
    fit, _, _ = case()
    calls = capture_svi(monkeypatch, fit)
    with pytest.raises(ValueError, match=message):
        fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(0), model_kwargs={**options(), **changes})
    assert not calls


@pytest.mark.parametrize("argument", ["model_kwargs", "p_initial"])
def test_options_must_be_mappings(argument):
    fit, _, _ = case()
    with pytest.raises(TypeError, match=argument):
        fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(0), **{argument: []})


def test_real_prefit_improves_fit_classifies_mismatch_and_respects_fit_mask(monkeypatch):
    fit, _, _ = case(single=True)
    flux = fit.observation.flux.copy()
    flux[[18, 20, 78]] += [.25, .5, -.3]
    flux[40] = 10.
    fit.observation = replace(fit.observation, flux=flux)
    mask = np.zeros(fit.observation.shape, bool)
    mask[40] = True
    fit.set_fit_mask(mask)
    before = fit.effective_mask.copy()
    start = {}
    run_svi = fit.run_svi

    def record_start(**kwargs):
        start.update(kwargs["p_initial"])
        return run_svi(**kwargs)

    monkeypatch.setattr(fit, "run_svi", record_start)
    settings = options()
    settings["degree"] = 1
    model, point = fit.run_mixture_prefit(
        rng_key=jax.random.PRNGKey(2), model_kwargs=settings, p_initial={"teff": 5600.},
        step_size=.02, num_steps=250, progress_bar=False)
    assert set(start).issubset(point)
    assert all(np.isfinite(value).all() for value in point.values())
    density = lambda p: log_density(model, (fit.fit_observation(), fit.specmodel), {}, p)[0]
    assert density(point) > density(start)
    np.testing.assert_allclose(point["continuum_coefficients"], [1.01, .01], atol=.005)
    np.testing.assert_allclose(point["teff"], 5800., atol=100.)
    prediction = fit.predict(model=model, point=point)
    assert np.all(np.asarray(prediction["mismatch_probability"])[[18, 20, 40, 78]] > .99)
    assert np.nanmedian(prediction["mismatch_probability"]) < .05
    np.testing.assert_array_equal(fit.effective_mask, before)
    selected = fit.mask_outliers(model=model, point=point, method="gaussian_mixture", mask_velocity=0.)
    assert selected[[18, 20, 40, 78]].all()

    # The notebook passes the full prefit point directly: shared sites are
    # reused, mixture-only entries ignored, and missing GP sites initialized.
    final_model = make_model_single(fit, **settings, use_gp=True, gp_solver="vecchia1")
    final_point = run_svi(model=final_model, rng_key=jax.random.PRNGKey(1),
                          p_initial=point, num_steps=5, progress_bar=False)
    assert {"teff", "jitter", "gp_amplitude", "gp_scale"}.issubset(final_point)
    assert not {"continuum_coefficients", "mismatch_fraction", "mismatch_scale"}.intersection(final_point)
    assert np.isfinite(log_density(
        final_model, (fit.fit_observation(), fit.specmodel), {}, final_point)[0])
    final_prediction = fit.predict(model=final_model, point=final_point)
    assert "mismatch_probability" not in final_prediction
    assert "continuum_posterior" in final_prediction
    assert np.isfinite(final_prediction["mean_flux"]).all()
    np.testing.assert_array_equal(fit.fit_mask, selected)
