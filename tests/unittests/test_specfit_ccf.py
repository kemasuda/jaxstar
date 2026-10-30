"""Post-fit CCF comparison detects mismatches without modifying the fit."""

from dataclasses import replace
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxstar.specfit import SpecFit, SpecModel, single_star_params
from jaxstar.specfit._ccf import _region_ccfs
from test_specfit import case, configured


@pytest.fixture(autouse=True)
def precision_and_figures(x64_context):
    with x64_context():
        yield
    plt.close("all")


def exact_case(single=False, fixed_ip=False):
    fit, _, _ = case(single)
    if fixed_ip:
        specmodel = SpecModel(fit.specmodel.spectra, vmax=30., fixed_ip=dict(
            mu1_kms=-1., mu2_kms=4., sigma1_kms=1., sigma2_kms=2., fraction2=.25))
        fit = SpecFit(fit.observation, specmodel)
    continuum = np.broadcast_to(np.linspace(.85, 1.15, 96), fit.observation.shape)

    def predict(point, observation, specmodel):
        params = single_star_params(
            {"teff": point.get("teff", 5800.)}, vsini=point.get("vsini", 6.),
            vmacro=3., q1=.36, q2=.3,
            rv=point.get("rv", 12. if single else np.array([-8., 12.])),
            resolving_power=None if fixed_ip else 70000.)
        flux = np.asarray(specmodel(params, observation.wavelength)) * continuum
        # Deliberately unrelated GP correction: it must not enter the CCF.
        return dict(params=params, continuum=continuum, model_flux=flux,
                    mean_flux=flux + .2 * np.sin(np.arange(96)))

    prediction = predict({}, fit.observation, fit.specmodel)
    measured = np.array(prediction["model_flux"])
    measured[np.asarray(fit.observation.mask)] = np.nan
    fit = SpecFit(replace(fit.observation, flux=measured), fit.specmodel)
    mask = np.zeros(fit.observation.shape, bool)
    mask[..., 44:47] = True  # Interior gap near a line.
    fit.set_fit_mask(mask)
    return fit, SimpleNamespace(predict=predict)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("weighting", ["uniform", "inverse_variance"])
@pytest.mark.parametrize("fixed_ip", [False, True])
def test_exact_match_preserves_masks_and_legacy_ccf(single, weighting, fixed_ip):
    fit, model = exact_case(single, fixed_ip)
    fit.check_ccf({"teff": 5800.}, v_limit=50., ccfvmax=40.)
    old_ccf = fit.ccf
    old_mask = fit.effective_mask.copy()
    result = fit.ccf_diagnostics(point={}, model=model, weighting=weighting)
    assert [r["region"] for r in result] == list(fit.regions)
    for r in result:
        np.testing.assert_allclose(r["data_ccf"], r["model_ccf"], atol=1e-14)
        np.testing.assert_allclose(r["residual"], 0., atol=1e-14)
        assert r["model_ccf"].max() == pytest.approx(1.)
        # A masked line core can shift both peaks; they must still agree.
        assert np.argmax(r["data_ccf"]) == np.argmax(r["model_ccf"])
        assert r["draw_indices"] is None
        assert r["n_pixels"] == 92
    assert fit.ccf is old_ccf
    np.testing.assert_array_equal(fit.effective_mask, old_mask)


def test_velocity_offsets_use_each_regions_fitted_rv():
    fit, model = exact_case()
    fit.reset_fit_mask()
    for r in fit.ccf_diagnostics(point={}, model=model):
        assert r["velocity"][np.argmax(r["model_ccf"])] == pytest.approx(0., abs=1.)


def test_depth_width_and_rv_discrepancies_remain_visible():
    fit, model = exact_case(single=True)
    # Scaling line depths must survive common rather than separate peak normalization.
    continuum = model.predict({}, fit.observation, fit.specmodel)["continuum"]
    observed = continuum + .7 * (fit.observation.flux - continuum)
    scaled = SpecFit(replace(fit.observation, flux=observed), fit.specmodel)
    scaled.set_fit_mask(fit.fit_mask)
    [r] = scaled.ccf_diagnostics(point={}, model=model)
    np.testing.assert_allclose(r["data_ccf"], .7 * r["model_ccf"], atol=1e-13)
    [wrong_width] = fit.ccf_diagnostics(point={"vsini": 18.}, model=model)
    assert np.max(np.abs(wrong_width["residual"])) > .1
    [wrong_rv] = fit.ccf_diagnostics(point={"rv": 7.}, model=model, velocity_step=.5)
    v = wrong_rv["velocity"]
    assert v[np.argmax(wrong_rv["data_ccf"])] == pytest.approx(5., abs=1.)
    assert v[np.argmax(wrong_rv["model_ccf"])] == pytest.approx(0., abs=1.)


def test_masked_nan_inf_and_outliers_do_not_enter_correlations():
    fit, model = exact_case()
    expected = fit.ccf_diagnostics(point={}, model=model)
    flux, error = np.array(fit.observation.flux), np.array(fit.observation.uncertainty)
    flux[fit.effective_mask] = 1e100
    error[fit.effective_mask] = np.inf
    other = SpecFit(replace(fit.observation, flux=flux, uncertainty=np.where(
        fit.observation.mask, error, .01)), fit.specmodel)
    other.set_fit_mask(fit.fit_mask)
    for a, b in zip(expected, other.ccf_diagnostics(point={}, model=model)):
        np.testing.assert_array_equal(a["data_ccf"], b["data_ccf"])


def test_irregular_pixels_blends_and_weights_match_direct_cross_product():
    c = 299792.458
    source = 5000 * np.exp(np.linspace(-180., 180., 1001) / c)
    v = np.linspace(-180., 180., 1001)
    template = 1 - .3*np.exp(-.5*((v+20)/3)**2) - .1*np.exp(-.5*((v-4)/9)**2)
    wave = 5000 * np.exp(np.array([-65., -22., -20., -19., -5., 3., 8., 40.]) / c)
    data = np.array([1., .8, .72, .73, .9, .88, .92, 1.])
    baseline = data + np.array([0., 0., .03, .02, -.02, 0., 0., 0.])
    error = np.linspace(.01, .04, len(wave))
    offsets = np.array([-4., 0., 4.])
    result = _region_ccfs(wave, data, baseline, error, source, template,
                          3., offsets, "inverse_variance")
    expected = []
    weights = 1 / error**2
    for offset in offsets:
        beta = (3. + offset) / c
        doppler = np.sqrt((1+beta)/(1-beta))
        t = np.interp(np.log(wave / doppler), np.log(source), template)
        expected.append([np.average((f-np.average(f, weights=weights)) *
                                    (t-np.average(t, weights=weights)), weights=weights)
                         for f in (data, baseline)])
    np.testing.assert_allclose(result, np.array(expected).T, rtol=1e-9, atol=1e-13)


def test_samples_average_paired_raw_ccfs_before_normalizing():
    fit, model = exact_case(single=True)
    samples = {"teff": np.array([5200., 5500., 6200.]), "rv": np.array([10., 11., 15.])}
    points = [{name: values[i] for name, values in samples.items()} for i in (0, 2)]
    individual = [fit.ccf_diagnostics(point=p, model=model)[0] for p in points]
    [actual] = fit.ccf_diagnostics(samples=samples, model=model, max_draws=2)
    raw_model = np.mean([r["model_ccf"] * r["normalization"] for r in individual], axis=0)
    raw_data = np.mean([r["data_ccf"] * r["normalization"] for r in individual], axis=0)
    np.testing.assert_allclose(actual["model_ccf"], raw_model / raw_model.max())
    np.testing.assert_allclose(actual["data_ccf"], raw_data / raw_model.max())
    np.testing.assert_array_equal(actual["draw_indices"], [0, 2])


def test_builtin_reconstruction_point_and_samples():
    fit, config, point = case(single=True)
    model = configured(fit, config)
    for options in (dict(point=point), dict(samples={"teff": np.array([5700., 5900.])})):
        [result] = fit.ccf_diagnostics(model=model, **options)
        assert np.isfinite(result["residual"]).all()


@pytest.mark.parametrize("options, message", [
    ({}, "exactly one"),
    ({"point": {}, "samples": {"teff": [5800.]}}, "exactly one"),
    ({"point": {}, "ccfvmax": 0}, "ccfvmax"),
    ({"point": {}, "velocity_step": float("nan")}, "velocity_step"),
    ({"point": {}, "weighting": "bad"}, "weighting"),
    ({"samples": {}}, "nonempty"),
    ({"samples": {"teff": [5800.]}, "max_draws": True}, "max_draws"),
    ({"point": {}, "ccfvmax": 500.}, "coverage"),
])
def test_invalid_inputs(options, message):
    fit, model = exact_case(single=True)
    with pytest.raises(ValueError, match=message):
        fit.ccf_diagnostics(model=model, **options)


@pytest.mark.parametrize("problem, message", [
    ("missing", "prediction fields"), ("multiple", "one stellar"),
    ("rv", "finite scalar"), ("continuum", "positive"),
    ("flat", "positive peak"),
])
def test_invalid_predictions(problem, message):
    fit, model = exact_case(single=True)
    p = model.predict({}, fit.observation, fit.specmodel)
    if problem == "missing":
        del p["params"]
    elif problem == "multiple":
        p["params"]["components"] *= 2
    elif problem == "rv":
        p["params"]["components"][0]["rv"] = np.nan
    elif problem == "continuum":
        p["continuum"] = -p["continuum"]
    else:
        p["model_flux"] = p["continuum"]
    with pytest.raises(ValueError, match=message):
        fit.ccf_diagnostics(point={}, model=SimpleNamespace(predict=lambda *args: p))


def test_no_usable_pixels():
    fit, model = exact_case(single=True)
    fit.set_fit_mask(np.ones(fit.observation.shape, bool))
    with pytest.raises(ValueError, match="usable pixels"):
        fit.ccf_diagnostics(point={}, model=model)


@pytest.mark.parametrize("filename", ["ccf.png", "ccf"])
def test_plot_curves_residuals_and_region_files(tmp_path, filename):
    fit, model = exact_case()
    result = fit.ccf_diagnostics(point={}, model=model)
    with plt.rc_context({"savefig.format": "png"}):
        plots = fit.plot_ccf(point={}, model=model, save_path=tmp_path / filename)
    for (fig, (upper, lower)), r in zip(plots, result):
        np.testing.assert_allclose(upper.lines[0].get_ydata(), r["data_ccf"])
        np.testing.assert_allclose(upper.lines[1].get_ydata(), r["model_ccf"])
        np.testing.assert_allclose(lower.lines[0].get_ydata(), r["residual"])
        np.testing.assert_array_equal(lower.lines[0].get_xdata(), r["velocity"])
        assert upper.get_shared_x_axes().joined(upper, lower)
        assert (tmp_path / f"ccf_region{r['region']}.png").read_bytes().startswith(b"\x89PNG")
