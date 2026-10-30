"""Model probabilities or residual-mixture fitting feed the same masking rule."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from jaxstar.specfit import SpecFit, chebyshev_basis
from jaxstar.specfit.specfit import _extend_mask
from jaxstar.specfit.sampling import _C_KMS
from jaxstar.specfit._outliers import _fit_gaussian_mixture, _mixture_residual_densities
from test_specfit import case, configured, clipping_case, clipping_model


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("solver", [None, "direct", "quasisep", "vecchia1"])
def test_fitted_covariance_and_physical_residual(single, solver, x64_context, monkeypatch):
    with x64_context():
        fit, settings, point = case(single=single)
        jitter = .004 if single else np.array([.004, .008])
        settings["jitter"] = jitter
        if solver:
            settings.update(gp_amplitude=.02, gp_scale=.1, gp_solver=solver)
        flux = fit.observation.flux.copy()
        flux[..., 50] -= .4
        fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
        previous = fit.fit_mask.copy()
        if solver == "vecchia1":
            # Both normal and contaminated fit-masked pixels must be reconsidered.
            previous[..., [20, 50]] = True
            fit.set_fit_mask(previous)

            def unavailable(*args, **kwargs):
                raise AssertionError("Vecchia masking must not construct a tinygp GP or predict its mean")

            monkeypatch.setattr("jaxstar.specfit.likelihood._region_gp", unavailable)
            monkeypatch.setattr("jaxstar.specfit.reconstruction.gp_conditional_mean", unavailable)
        model = configured(fit, settings)
        original_prediction = fit.predict(point=point, model=model)
        if solver == "vecchia1":
            assert "gp_mean" not in original_prediction
            assert np.isfinite(original_prediction["model_flux"]).all()
        elif solver:
            assert np.all(np.abs(original_prediction["gp_mean"][..., 50]) > .1)
        fixed = fit.observation.mask.copy()
        flags = fit.mask_outliers(point=point, model=model,
                                  extend_outlier_mask=False)
        d = fit.outlier_diagnostics
        assert d["probability_source"] == "residual_fit"
        variance = fit.observation.uncertainty**2 + (
            jitter if single else jitter[:, None])**2 + (0.02**2 if solver else 0.)
        np.testing.assert_allclose(d["normal_residual_variance"][~fixed], variance[~fixed], rtol=1e-12)
        residual = flux[~fixed] - original_prediction["model_flux"][~fixed]
        fraction, scale = d["mismatch_fraction"], d["mismatch_scale"]
        terms = np.stack((np.log1p(-fraction) + norm.logpdf(residual, scale=np.sqrt(variance[~fixed])),
                          np.log(fraction) + norm.logpdf(residual, scale=np.sqrt(variance[~fixed] + scale**2))))
        expected = np.exp(terms[1] - logsumexp(terms, axis=0))
        np.testing.assert_allclose(d["mismatch_probability"][~fixed], expected, rtol=1e-10)
        assert flags[..., 50].all() and not flags[fixed].any()
        if solver == "vecchia1":
            assert not flags[..., 20].any()
        assert np.isnan(d["mismatch_probability"][fixed]).all()
        np.testing.assert_array_equal(flags, d["mismatch_probability"] > .5)
        np.testing.assert_array_equal(fit.observation.mask, fixed)
        # The model and its prediction contract still contain no mixture sites.
        assert not {"mismatch_fraction", "mismatch_scale"}.intersection(model.prior_trace())
        fit.set_fit_mask(previous)
        np.testing.assert_array_equal(fit.predict(point=point, model=model)["mean_flux"],
                                      original_prediction["mean_flux"])


def custom_case():
    fit, baseline = clipping_case()
    calls = []

    def predict(point, observation, specmodel):
        calls.append(observation.mask.copy())
        return {"model_flux": point["baseline"], "mean_flux": observation.flux,
                "normal_residual_variance": np.full(observation.shape, .01**2)}

    return fit, {"baseline": baseline}, SimpleNamespace(predict=predict), calls


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("extend", [False, True])
def test_mixture_prediction_reused_without_refitting_and_old_flags_reconsidered(single, extend, x64_context, monkeypatch):
    with x64_context():
        fit, settings, point = case(single)
        flux = fit.observation.flux.copy()
        flux[..., 50] -= .4
        fit = SpecFit(replace(fit.observation, flux=flux), fit.specmodel)
        settings.update(likelihood="gaussian_mixture", mismatch_fraction=dist.Beta(1., 9.),
                        mismatch_scale=dist.HalfNormal(.1), degree=1,
                        basis=chebyshev_basis(fit.observation.wavelength, 1))
        point = dict(point, mismatch_fraction=.08, mismatch_scale=.1,
                     continuum_coefficients=np.broadcast_to([1.01, .01],
                         (2,) if single else (2, 2)).copy())
        model = configured(fit, settings)
        old = np.zeros(fit.observation.shape, bool)
        old[..., [20, 50]] = True  # A good and a bad pixel, both previously excluded.
        fit.set_fit_mask(old)
        fixed = fit.observation.mask.copy()
        prediction = fit.predict(point=point, model=model)
        predicted = np.asarray(prediction["mismatch_probability"])
        velocity = .5 * float(np.max(fit.specmodel.broadening_fwhm(prediction["params"])))
        assert np.all(predicted[..., 50] > .99) and np.all(predicted[..., 20] < .5)

        def no_refit(*args, **kwargs):
            raise AssertionError("model probabilities must bypass residual-mixture optimization")

        monkeypatch.setattr("jaxstar.specfit._outliers._fit_gaussian_mixture", no_refit)
        flags = fit.mask_outliers(point=point, model=model, extend_outlier_mask=extend)
        expected = np.zeros(fit.observation.shape, bool)
        for wavelengths, q, row in zip(np.atleast_2d(fit.observation.wavelength),
                                       np.atleast_2d(predicted), np.atleast_2d(expected)):
            distances = _C_KMS * np.abs(np.log(wavelengths[:, None] / wavelengths[q > .5]))
            row[:] = np.any(distances <= (velocity if extend else 0.), axis=1)
        np.testing.assert_array_equal(flags, expected & ~fixed)
        assert flags[..., 50].all() and not flags[..., 20].any()
        diagnostics = fit.outlier_diagnostics
        assert diagnostics["probability_source"] == "model"
        assert diagnostics["mismatch_fraction"] == .08
        assert diagnostics["mismatch_scale"] == .1
        assert "log_likelihood" not in diagnostics and "collapsed" not in diagnostics
        np.testing.assert_array_equal(diagnostics["mismatch_probability"], predicted)
        variance = fit.observation.uncertainty**2 + settings["jitter"]**2
        np.testing.assert_allclose(diagnostics["normal_residual_variance"][~fixed], variance[~fixed])
        np.testing.assert_array_equal(fit.observation.mask, fixed)
        np.testing.assert_array_equal(fit.predict(point=point, model=model)["mismatch_probability"], predicted)
        # Repeated calls with the same point cannot expand the mask recursively.
        np.testing.assert_array_equal(fit.mask_outliers(point=point, model=model,
                                                       extend_outlier_mask=extend), flags)


@pytest.mark.parametrize("metadata", [(), ("mismatch_fraction",), ("mismatch_scale",),
                                      ("mismatch_fraction", "mismatch_scale")])
def test_custom_probabilities_snapshot_and_plot(metadata, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    fit, point, model, _ = custom_case()
    prediction = model.predict(point, fit.observation, fit.specmodel)
    q = np.full(fit.observation.shape, .1)
    q[20] = .9  # Intentionally different from the residual-based outlier at pixel 10.
    q[fit.observation.mask] = 1.  # Fixed masks must not seed dilation.
    prediction["mismatch_probability"] = q
    parameters = dict(mismatch_fraction=.2, mismatch_scale=.15)
    prediction.update({name: parameters[name] for name in metadata})
    model = SimpleNamespace(predict=lambda *args: prediction)

    def no_refit(*args, **kwargs):
        raise AssertionError("supplied probabilities must be used verbatim")

    monkeypatch.setattr("jaxstar.specfit._outliers._fit_gaussian_mixture", no_refit)
    try:
        flags = fit.mask_outliers(point=point, model=model, extend_outlier_mask=False,
                                  plot=True, save_path=tmp_path / "prediction.png")
        assert flags[20] and not flags[10] and not flags[fit.observation.mask].any()
        diagnostics = fit.outlier_diagnostics
        assert diagnostics["probability_source"] == "model"
        assert diagnostics["mismatch_fraction"] == (.2 if "mismatch_fraction" in metadata else None)
        assert diagnostics["mismatch_scale"] == (.15 if "mismatch_scale" in metadata else None)
        expected = np.where(fit.observation.mask, np.nan, q)
        np.testing.assert_array_equal(diagnostics["mismatch_probability"], expected)
        figure = plt.gcf()
        np.testing.assert_array_equal(figure.axes[2].lines[0].get_ydata(), expected)
        histogram = figure.axes[3]
        summary = histogram.texts[0].get_text()
        assert ("f_mismatch = 0.2" if "mismatch_fraction" in metadata
                else "f_mismatch = not supplied") in summary
        retained, masked = [patch.get_data() for patch in histogram.patches if patch.get_fill()]
        np.testing.assert_array_equal(masked.baseline, retained.values)
        assert masked.values.sum() == np.count_nonzero(~fit.observation.mask)
        assert retained.values.sum() == np.count_nonzero(~fit.observation.mask & ~flags)
        overlays = {line.get_label(): line for line in histogram.lines}
        if len(metadata) == 2:
            assert set(overlays) == {"normal component", "outlier component", "total mixture"}
            assert all(line.get_linewidth() == .8 for line in overlays.values())
            x = overlays["total mixture"].get_xdata()
            normal_range = np.abs(x) <= 6.
            np.testing.assert_array_equal(overlays["normal component"].get_xdata(), x[normal_range])
            np.testing.assert_allclose(overlays["total mixture"].get_ydata()[normal_range],
                                       overlays["normal component"].get_ydata()
                                       + overlays["outlier component"].get_ydata()[normal_range])
        else:
            assert not overlays
        assert (tmp_path / "prediction_region8.png").is_file()
        # Diagnostics must not retain mutable references owned by a custom model.
        q[:] = .7
        prediction["normal_residual_variance"][:] = 4.
        np.testing.assert_array_equal(diagnostics["mismatch_probability"], expected)
        np.testing.assert_allclose(diagnostics["normal_residual_variance"], .01**2)
    finally:
        plt.close("all")


@pytest.mark.parametrize("bad", [-.1, 1.1, np.nan, np.inf, [0., 1.]])
def test_invalid_model_probabilities_preserve_mask_without_fallback(bad, monkeypatch):
    fit, point, model, _ = custom_case()
    old = fit.fit_mask.copy()
    old[20] = True
    fit.set_fit_mask(old)
    prediction = model.predict(point, fit.fit_observation(), fit.specmodel)
    q = np.full(fit.observation.shape, .1)
    if np.ndim(bad):
        q = np.asarray(bad)
    else:
        q[20] = bad  # Validation must cover old fit-masked pixels too.
    prediction["mismatch_probability"] = q
    model = SimpleNamespace(predict=lambda *args: prediction)
    monkeypatch.setattr("jaxstar.specfit._outliers._fit_gaussian_mixture",
                        lambda *args: pytest.fail("invalid model probabilities must not trigger a refit"))
    with pytest.raises(ValueError):
        fit.mask_outliers(point=point, model=model)
    np.testing.assert_array_equal(fit.fit_mask, old)


def test_sigma_clip_ignores_supplied_mismatch_probabilities():
    fit, point, model, _ = custom_case()
    options = dict(point=point, method="sigma_clip", mask_v=10., extend_outlier_mask=False)
    expected = fit.mask_outliers(model=model, **options).copy()
    fit.reset_fit_mask()
    prediction = model.predict(point, fit.observation, fit.specmodel)
    prediction["mismatch_probability"] = np.full(fit.observation.shape, np.nan)
    model = SimpleNamespace(predict=lambda *args: prediction)
    np.testing.assert_array_equal(fit.mask_outliers(model=model, **options), expected)


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
            "model_flux": point["baseline"], "normal_residual_variance": variance})
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
        "model_flux": np.ones(q.shape), "normal_residual_variance": np.ones(q.shape)})
    flags = fit.mask_outliers(point={}, model=model, method="gaussian_mixture",
                              extension_factor=100., extend_outlier_mask=extend, mask_velocity=velocity)
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


@pytest.mark.parametrize("value", [-1., np.nan, np.inf, [10.], "automatic", "10"])
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
    expected = fit.mask_outliers(point=point, model=model, method="gaussian_mixture", mask_velocity=10.).copy()
    fit.reset_fit_mask()
    actual = fit.mask_outliers(point=point, model=model, mask_velocity=10.)
    np.testing.assert_array_equal(actual, expected)
    assert fit.outlier_diagnostics["method"] == "gaussian_mixture"


@pytest.mark.parametrize("fraction", [0., .8, 1.])
def test_mixture_residual_component_densities(fraction):
    from scipy.integrate import simpson

    # The two variance groups give outlier widths 2 and 4 after normalization.
    # Check peak heights and integrated component weights, including a
    # majority of outliers; averaging V_i would give the wrong density.
    x = np.linspace(-40., 40., 4001)
    variance = np.array([1., .2, 1., .2])
    normal, outlier = _mixture_residual_densities(x, variance, fraction, np.sqrt(3.))
    np.testing.assert_allclose(normal[x == 0.], (1 - fraction) * .3989422804014327)
    np.testing.assert_allclose(outlier[x == 0.], fraction * .14960335515053727)
    np.testing.assert_allclose(outlier[x == 4.], fraction * .04374408219318993)
    np.testing.assert_allclose((simpson(normal, x=x), simpson(outlier, x=x)), (1 - fraction, fraction))
    # Changing flux units must not change this standardized diagnostic.
    rescaled = _mixture_residual_densities(x, variance * 37.**2, fraction, np.sqrt(3.) * 37.)
    np.testing.assert_allclose(rescaled, (normal, outlier))


def test_probabilities_plot_and_fully_masked(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fit, point, model, _ = custom_case()
    before = set(plt.get_fignums())
    fit.mask_outliers(point=point, model=model, method="gaussian_mixture", mask_velocity=10., plot=True)
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
                      mask_velocity=10., save_path=tmp_path / "mask.png")
    assert (tmp_path / "mask_region8.png").is_file()
    assert set(plt.get_fignums()) == before
    fit.observation = replace(fit.observation, mask=np.ones(fit.observation.shape, bool))
    fit.mask_outliers(point=point, model=model, method="gaussian_mixture",
                      mask_velocity=10., save_path=tmp_path / "all_masked.png")
    assert not fit.fit_mask.any()
    assert np.isnan(fit.outlier_diagnostics["mismatch_probability"]).all()
