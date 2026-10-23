"""Saved spectral fits retain observations, masks, native sites and provenance."""

from dataclasses import replace
import json
from pathlib import Path

import arviz as az
import jax
import numpy as np
import numpyro.distributions as dist
import pytest
import xarray as xr

from jaxstar.specfit import (
    ConfiguredModel, FitResult, SpecFit, load_fit_result, make_model_single,
    save_fit_result,
)
from test_specfit import case, configured
from test_specfit_priors import make_fit


@pytest.fixture(scope="module")
def sampled_fit(x64_context):
    with x64_context():
        fit, config, point = case()
        config["rv"] = dist.Uniform(10., 14.).expand((2,)).to_event(1)
        point = {**point, "rv": np.array([12., 12.])}
        mask = np.zeros(fit.observation.shape, bool)
        mask[0, 10] = True
        fit.set_fit_mask(mask)
        model = configured(fit, config)
        mcmc = fit.run_nuts(
            model=model, rng_key=jax.random.PRNGKey(81), num_warmup=5,
            num_samples=6, num_chains=2, progress_bar=False,
            nuts_kwargs={"max_tree_depth": 2},
            mcmc_kwargs={"chain_method": "sequential"})
        yield fit, config, model, point, mcmc


def test_mcmc_roundtrip_and_reconstruction(sampled_fit, tmp_path, monkeypatch):
    fit, config, model, point, mcmc = sampled_fit
    expected = fit.predict(point=point, model=model)
    original_call = ConfiguredModel.__call__

    def no_evaluation(*args, **kwargs):
        raise AssertionError("saving/loading must not evaluate the model")

    monkeypatch.setattr(ConfiguredModel, "__call__", no_evaluation)
    output = save_fit_result(tmp_path / "target", sf=fit, model=model, point=point,
                             mcmc=mcmc, metadata={"target": "star", "seed": 81})
    result = load_fit_result(output)
    monkeypatch.setattr(ConfiguredModel, "__call__", original_call)
    assert isinstance(result, FitResult)
    assert output == (tmp_path / "target").resolve()
    for name in ("wavelength", "flux", "uncertainty", "mask"):
        actual = getattr(result.observation, name)
        np.testing.assert_array_equal(actual, getattr(fit.observation, name))
        assert actual.dtype == getattr(fit.observation, name).dtype
    assert result.observation.region == (8, 9)
    assert result.observation.exposure == "synthetic"
    np.testing.assert_array_equal(result.fit_mask, fit.fit_mask)
    assert not result.fit_mask.flags.writeable
    for name, value in point.items():
        np.testing.assert_array_equal(result.point[name], value)
    assert result.idata.posterior.rv.dims == ("chain", "draw", "region")
    np.testing.assert_array_equal(result.idata.posterior.region, [8, 9])
    assert result.idata.posterior.sizes["chain"] == 2
    assert result.idata.posterior.sizes["draw"] == 6
    assert not hasattr(result.idata, "log_likelihood")
    np.testing.assert_array_equal(result.idata.sample_stats.diverging,
                                   mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    for name, value in mcmc.get_samples().items():
        np.testing.assert_array_equal(result.samples[name], value)
    assert result.samples["rv"].shape == (12, 2)
    assert result.metadata["sampler"]["fit_state_verified"]
    assert set(result.metadata["sampler"]["latent_names"]) == {"teff", "rv"}
    assert result.metadata["user"] == {"target": "star", "seed": 81}
    assert result.metadata["model"]["settings"]["atmosphere"]["teff"]["parameters"] == {
        "low": 5000., "high": 6500.}
    assert len(result.metadata["grid"]["grid_sha256"]) == 64
    assert result.metadata["grid"]["flux_shape"] == list(fit.specmodel.spectra.grid.field("flux").values.shape)
    restored_fit = SpecFit(result.observation, fit.specmodel)
    restored_fit.set_fit_mask(result.fit_mask)
    restored_model = configured(restored_fit, config)
    prediction = restored_fit.predict(point=result.point, model=restored_model)
    for name in ("physical_flux", "model_flux", "mean_flux"):
        np.testing.assert_allclose(prediction[name], expected[name], equal_nan=True)
    original_mean = fit.predict(samples=mcmc.get_samples(), model=model, max_draws=2)
    restored_mean = restored_fit.predict(samples=result.samples, model=restored_model, max_draws=2)
    np.testing.assert_allclose(restored_mean["mean_flux"], original_mean["mean_flux"], equal_nan=True)


@pytest.mark.parametrize("single", [False, True])
def test_point_only_and_resolved_factory_settings(single, tmp_path, x64_context):
    with x64_context():
        fit = make_fit(single=single)
        model = make_model_single(fit, resolving_power=np.inf, use_gp=False)
        point = {name: site["value"] for name, site in model.prior_trace().items()
                 if site["type"] == "sample" and not site["is_observed"]}
        output = save_fit_result(tmp_path / "point", sf=fit, model=model, point=point)
        result = load_fit_result(output)
        assert result.idata is None
        with pytest.raises(ValueError, match="no posterior"):
            _ = result.samples
        assert result.observation.shape == fit.observation.shape
        assert not (output / "posterior.nc").exists()
        settings = result.metadata["model"]["settings"]
        for actual, expected in zip(settings["rv_bounds"], model.settings["rv_bounds"]):
            np.testing.assert_array_equal(actual["values"], expected)
        assert settings["vsini_max"] == model.settings["vsini_max"]
        assert settings["resolving_power"]["values"] == {"nonfinite": "inf"}
        assert result.metadata["ccf"]["rv"] == fit.ccf["rv"]
        np.testing.assert_array_equal(result.metadata["grid"]["wavelength"]["values"],
                                      fit.specmodel.spectra.wavelength)
        for name in point:
            np.testing.assert_array_equal(result.point[name], point[name])


def test_supplied_idata_preserves_groups_and_coordinate_order(tmp_path, arviz_from_dict):
    fit, config, _ = case()
    idata = arviz_from_dict(
        posterior={"teff": np.arange(6.).reshape(2, 3),
                   "rv": np.arange(12.).reshape(2, 3, 2)},
        sample_stats={"diverging": np.array([[False, True, False], [False, False, False]])},
        posterior_predictive={"flux": np.ones((2, 3, 4))},
        coords={"region": ["blue", "red"]}, dims={"rv": ["region"]})
    # A caller may have transposed xarray data or added plot-only quantities.
    idata.posterior["rv"] = idata.posterior.rv.transpose("region", "draw", "chain")
    idata.posterior["display_teff"] = idata.posterior.teff + 1000.
    original = idata.copy()
    output = save_fit_result(tmp_path / "idata", sf=fit, model=configured(fit, config), idata=idata)
    result = load_fit_result(output)
    assert result.point is None
    if int(az.__version__.split(".", 1)[0]) >= 1:
        assert result.idata.groups == idata.groups
    else:
        assert result.idata.groups() == idata.groups()
    for name in ("posterior", "sample_stats", "posterior_predictive"):
        xr.testing.assert_identical(getattr(result.idata, name), getattr(original, name))
        xr.testing.assert_identical(getattr(idata, name), getattr(original, name))
    np.testing.assert_array_equal(result.samples["rv"], np.arange(12.).reshape(6, 2))
    # Loaded arrays remain usable after the backing file goes away.
    (output / "posterior.nc").unlink()
    np.testing.assert_array_equal(result.samples["display_teff"], np.arange(6.) + 1000.)


def test_overwrite_keeps_unrelated_files_and_removes_stale_posterior(sampled_fit, tmp_path):
    fit, _, model, point, mcmc = sampled_fit
    output = save_fit_result(tmp_path / "target", sf=fit, model=model, mcmc=mcmc)
    (output / "notes.txt").write_text("keep this")
    with pytest.raises(FileExistsError, match="overwrite"):
        save_fit_result(output, sf=fit, model=model, point=point)
    save_fit_result(output, sf=fit, model=model, point=point, overwrite=True)
    assert load_fit_result(output).idata is None
    assert not (output / "posterior.nc").exists()
    assert (output / "notes.txt").read_text() == "keep this"


def test_failed_save_does_not_replace_existing_result(sampled_fit, tmp_path, monkeypatch, arviz_data_type):
    fit, _, model, point, mcmc = sampled_fit
    output = save_fit_result(tmp_path / "target", sf=fit, model=model, point=point)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail(*args, **kwargs):
        raise OSError("simulated NetCDF write failure")

    monkeypatch.setattr(arviz_data_type, "to_netcdf", fail)
    with pytest.raises(OSError, match="simulated"):
        save_fit_result(output, sf=fit, model=model, mcmc=mcmc, overwrite=True)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(tmp_path.glob(".specfit-*"))
    np.testing.assert_array_equal(load_fit_result(output).point["teff"], point["teff"])


@pytest.mark.parametrize("filename", ["arrays.npz", "posterior.nc"])
def test_corruption_is_rejected(sampled_fit, tmp_path, filename):
    fit, _, model, _, mcmc = sampled_fit
    output = save_fit_result(tmp_path / "target", sf=fit, model=model, mcmc=mcmc)
    with (output / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        load_fit_result(output)


def test_wrong_format_and_unsafe_manifest_are_rejected(tmp_path):
    fit, config, point = case(single=True)
    output = save_fit_result(tmp_path / "target", sf=fit, model=configured(fit, config), point=point)
    path = output / "result.json"
    manifest = json.loads(path.read_text())
    manifest["version"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported"):
        load_fit_result(output)
    manifest["version"] = 1
    manifest["files"]["../outside"] = "anything"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        load_fit_result(output)


def test_mcmc_mask_and_model_mismatch_rejected(sampled_fit, tmp_path):
    fit, config, model, point, mcmc = sampled_fit
    original = fit.fit_mask.copy()
    fit.set_fit_mask(~original)
    try:
        with pytest.raises(ValueError, match="mask changed"):
            save_fit_result(tmp_path / "mask", sf=fit, model=model, mcmc=mcmc)
    finally:
        fit.set_fit_mask(original)
    with pytest.raises(ValueError, match="do not match"):
        save_fit_result(tmp_path / "model", sf=fit, model=configured(fit, config), mcmc=mcmc)
    assert not (tmp_path / "mask").exists()
    assert not (tmp_path / "model").exists()


def test_invalid_inputs_and_custom_callable(tmp_path, arviz_data_type):
    fit, config, point = case(single=True)
    model = configured(fit, config)
    output = tmp_path / "target"
    with pytest.raises(ValueError, match="supply point"):
        save_fit_result(output, sf=fit, model=model)
    with pytest.raises(ValueError, match="only one"):
        save_fit_result(output, sf=fit, model=model, mcmc=object(), idata=object())
    with pytest.raises(TypeError, match="numeric"):
        save_fit_result(output, sf=fit, model=model, point={"teff": "bad"})
    with pytest.raises(ValueError, match="posterior group"):
        save_fit_result(output, sf=fit, model=model, idata=arviz_data_type())
    assert not output.exists()
    # No hidden model evaluation or executable deserialization for a custom fit.
    def custom_model(observation, specmodel):
        raise AssertionError("must not be called")
    output = save_fit_result(output, sf=fit, model=custom_model, point={},
                             metadata={"input": Path("spectrum.csv")})
    result = load_fit_result(output)
    assert result.point == {}
    assert result.metadata["model"]["function"]["callable"].endswith("custom_model")
    assert result.metadata["user"]["input"] == "spectrum.csv"
