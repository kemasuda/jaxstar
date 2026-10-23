"""Post-fit mixture classification leaves the Gaussian/GP inference unchanged."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from jaxstar.specfit import SpecFit
from jaxstar.specfit.specfit import _extend_mask
from jaxstar.specfit.sampling import _C_KMS
from jaxstar.specfit._outliers import _fit_gaussian_mixture
from test_specfit import case, configured, clipping_case, clipping_model


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("solver", [None, "direct", "quasisep"])
def test_fitted_covariance_and_physical_residual(single, solver, x64_context):
    with x64_context():
        fit, settings, point = case(single=single)
        jitter = .004 if single else np.array([.004, .008])
        settings["jitter"] = jitter
        if solver:
            settings.update(gp_amplitude=.02, gp_scale=.1, gp_solver=solver)
        flux = fit.observation.flux.copy()
        flux[..., 50] -= .4
        fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
        model = configured(fit, settings)
        original_prediction = fit.predict(point=point, model=model)
        if solver:
            assert np.all(np.abs(original_prediction["gp_mean"][..., 50]) > .1)
        fixed = fit.observation.mask.copy()
        flags = fit.mask_outliers(point=point, model=model,
                                  extend_outlier_mask=False)
        d = fit.outlier_diagnostics
        variance = fit.observation.uncertainty**2 + (
            jitter if single else jitter[:, None])**2 + (0.02**2 if solver else 0.)
        np.testing.assert_allclose(d["normal_residual_variance"][~fixed], variance[~fixed], rtol=1e-12)
        residual = flux[~fixed] - original_prediction["continuum_flux"][~fixed]
        fraction, scale = d["mismatch_fraction"], d["mismatch_scale"]
        terms = np.stack((np.log1p(-fraction) + norm.logpdf(residual, scale=np.sqrt(variance[~fixed])),
                          np.log(fraction) + norm.logpdf(residual, scale=np.sqrt(variance[~fixed] + scale**2))))
        expected = np.exp(terms[1] - logsumexp(terms, axis=0))
        np.testing.assert_allclose(d["mismatch_probability"][~fixed], expected, rtol=1e-10)
        assert flags[..., 50].all() and not flags[fixed].any()
        assert np.isnan(d["mismatch_probability"][fixed]).all()
        np.testing.assert_array_equal(flags, d["mismatch_probability"] > .5)
        np.testing.assert_array_equal(fit.observation.mask, fixed)
        # The model and its prediction contract still contain no mixture sites.
        assert not {"mismatch_fraction", "mismatch_scale"}.intersection(model.prior_trace())
        fit.reset_fit_mask()
        np.testing.assert_array_equal(fit.predict(point=point, model=model)["mean_flux"],
                                      original_prediction["mean_flux"])


def custom_case():
    fit, baseline = clipping_case()
    calls = []

    def predict(point, observation, specmodel):
        calls.append(observation.mask.copy())
        return {"continuum_flux": point["baseline"], "mean_flux": observation.flux,
                "normal_residual_variance": np.full(observation.shape, .01**2)}

    return fit, {"baseline": baseline}, SimpleNamespace(predict=predict), calls


def test_threshold_extension_reconsideration_and_single_prediction():
    fit, point, model, calls = custom_case()
    old = fit.fit_mask.copy()
    old[20] = True
    fit.set_fit_mask(old)
    fixed = fit.observation.mask.copy()
    flags = fit.mask_outliers(point=point, model=model, method="gaussian_mixture",
                              p_outlier=.9, extension_factor=.5, mask_velocity=None).copy()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], fixed | old)
    q = fit.outlier_diagnostics["mismatch_probability"]
    np.testing.assert_array_equal(flags, _extend_mask(q > .9, .5) & ~fixed)
    assert flags[10] and not flags[20]
    for threshold in (0., 1.):
        result = fit.mask_outliers(point=point, model=model, method="gaussian_mixture",
                                   p_outlier=threshold, extend_outlier_mask=False)
        np.testing.assert_array_equal(result, fit.outlier_diagnostics["mismatch_probability"] > threshold)
    assert not fit.fit_mask.any()  # Strict comparison at threshold 1.


@pytest.mark.parametrize("value", [-.1, 1.1, np.nan, np.inf, [.9]])
def test_invalid_threshold_preserves_mask(value):
    fit, point, model, calls = custom_case()
    before = fit.fit_mask.copy()
    with pytest.raises(ValueError, match="p_outlier"):
        fit.mask_outliers(point=point, model=model, method="gaussian_mixture", p_outlier=value)
    np.testing.assert_array_equal(fit.fit_mask, before)
    assert not calls


def test_invalid_variance_or_custom_model_preserves_mask():
    fit, point, model, _ = custom_case()
    previous = fit.fit_mask.copy()
    previous[20] = True
    fit.set_fit_mask(previous)
    for variance in (-np.ones(fit.observation.shape), np.full(fit.observation.shape, np.nan)):
        bad = SimpleNamespace(predict=lambda *args: {
            "continuum_flux": point["baseline"], "normal_residual_variance": variance})
        with pytest.raises(ValueError):
            fit.mask_outliers(point=point, model=bad, method="gaussian_mixture")
        np.testing.assert_array_equal(fit.fit_mask, previous)
    with pytest.raises(ValueError, match="normal_residual_variance"):
        fit.mask_outliers(point=point, model=clipping_model(fit), method="gaussian_mixture")
    with pytest.raises(ValueError, match="method"):
        fit.mask_outliers(point=point, model=model, method="unknown")


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("velocity, extend", [(0., True), (10., True), (10., False)])
def test_velocity_extension_on_nonuniform_wavelengths(single, velocity, extend, monkeypatch):
    fit, _, _ = case(single=single)
    wave = np.atleast_2d(fit.observation.wavelength)
    positions = np.linspace(0., 1., wave.shape[-1]) ** 1.4
    wave = wave[:, :1] * np.exp(np.log(wave[:, -1:] / wave[:, :1]) * positions)
    fit = SpecFit(replace(fit.observation, wavelength=wave[0] if single else wave), fit.specmodel)
    fixed = fit.observation.mask.copy()
    q = np.full(fit.observation.shape, .1)
    q[..., [0, 4, 50, 53, -1]] = .9  # Edges, overlapping windows and fixed-mask neighbor.
    q[..., 25] = .5  # Strict default threshold: not a seed.
    if not single:
        q[1] = .1
        q[1, 30] = .9  # Each region has independent seeds.
    monkeypatch.setattr("jaxstar.specfit._outliers._fit_gaussian_mixture",
                        lambda *args: ({}, q[~fixed]))
    model = SimpleNamespace(predict=lambda *args: {
        "continuum_flux": np.ones(q.shape), "normal_residual_variance": np.ones(q.shape)})
    options = {} if velocity == 10. else {"mask_velocity": velocity}
    flags = fit.mask_outliers(point={}, model=model, method="gaussian_mixture",
                              extension_factor=100., extend_outlier_mask=extend, **options)
    expected = []
    for wavelengths, probabilities in zip(wave, np.atleast_2d(q)):
        distance = _C_KMS * np.abs(np.log(wavelengths[:, None] / wavelengths[probabilities > .5]))
        expected.append(np.any(distance <= (velocity if extend else 0.), axis=1))
    expected = np.asarray(expected).reshape(q.shape) & ~fixed
    np.testing.assert_array_equal(flags, expected)
    np.testing.assert_array_equal(fit.observation.mask, fixed)
    np.testing.assert_array_equal(fit.outlier_diagnostics["mismatch_probability"],
                                  np.where(fixed, np.nan, q))
    assert fit.outlier_diagnostics["p_outlier"] == .5


@pytest.mark.parametrize("value", [-1., np.nan, np.inf, [10.]])
def test_invalid_mask_velocity_preserves_mask(value):
    fit, point, model, calls = custom_case()
    before = fit.fit_mask.copy()
    with pytest.raises(ValueError, match="mask_velocity"):
        fit.mask_outliers(point=point, model=model, method="gaussian_mixture", mask_velocity=value)
    np.testing.assert_array_equal(fit.fit_mask, before)
    assert not calls


def test_mixture_fit_null_case_and_flux_rescaling():
    residual = np.random.default_rng(321).normal(0., .01, 1000)
    residual[::50] += .25
    variance = np.full(residual.shape, .01**2)
    fitted, q = _fit_gaussian_mixture(residual, variance)
    scaled, scaled_q = _fit_gaussian_mixture(residual * 37., variance * 37.**2)
    assert 0 < fitted["mismatch_fraction"] < 1 and fitted["mismatch_scale"] > 0
    assert np.all(q[::50] > .99) and np.median(q) < .01
    np.testing.assert_allclose(scaled_q, q, atol=1e-7)
    np.testing.assert_allclose(scaled["mismatch_scale"], fitted["mismatch_scale"] * 37., rtol=1e-5)
    null, null_q = _fit_gaussian_mixture(np.zeros(100), np.ones(100))
    assert null["collapsed"] and np.all(null_q < .01)


def test_default_is_gaussian_mixture():
    fit, point, model, _ = custom_case()
    expected = fit.mask_outliers(point=point, model=model, method="gaussian_mixture").copy()
    fit.reset_fit_mask()
    actual = fit.mask_outliers(point=point, model=model)
    np.testing.assert_array_equal(actual, expected)
    assert fit.outlier_diagnostics["method"] == "gaussian_mixture"


def test_probabilities_plot_and_fully_masked(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, point, model, _ = custom_case()
    before = set(plt.get_fignums())
    fit.mask_outliers(point=point, model=model, method="gaussian_mixture", plot=True)
    figure = plt.gcf()
    assert len(figure.axes) == 4
    upper, middle, lower, histogram = figure.axes
    valid = ~fit.observation.mask
    baseline = next(line for line in upper.lines if line.get_label() == "physical spectrum × continuum")
    # The physical prediction is displayed even where observations are masked.
    np.testing.assert_allclose(baseline.get_xdata(), fit.observation.wavelength)
    np.testing.assert_allclose(baseline.get_ydata(), point["baseline"])
    # The fixture's mean_flux absorbs the spike; the plot must still show it.
    np.testing.assert_allclose(middle.lines[0].get_ydata()[valid],
                               fit.observation.flux[valid] - point["baseline"][valid])
    np.testing.assert_allclose(lower.lines[0].get_ydata(),
                               fit.outlier_diagnostics["mismatch_probability"], equal_nan=True)
    assert histogram.get_yscale() == "log"
    assert middle.get_shared_x_axes().joined(upper, lower)
    plt.close(figure)
    fit.mask_outliers(point=point, model=model, method="gaussian_mixture",
                      save_path=tmp_path / "mask.png")
    assert (tmp_path / "mask_region8.png").is_file()
    assert set(plt.get_fignums()) == before
    fit.observation = replace(fit.observation, mask=np.ones(fit.observation.shape, bool))
    fit.mask_outliers(point=point, model=model, method="gaussian_mixture",
                      save_path=tmp_path / "all_masked.png")
    assert not fit.fit_mask.any()
    assert np.isnan(fit.outlier_diagnostics["mismatch_probability"]).all()
