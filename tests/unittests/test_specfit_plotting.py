"""Posterior diagnostics preserve sampled sites, coordinates and saved draws."""

import arviz as az
import jax
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
import xarray as xr

from jaxstar.specfit import (
    load_fit_result, plot_mcmc_diagnostics, save_fit_result,
)
from test_specfit import case


@pytest.fixture(scope="module")
def sampled():
    fit, _, _ = case()
    guard = {"allow_evaluation": True}

    def model(observation, specmodel):
        if not guard["allow_evaluation"]:
            raise AssertionError("diagnostics must not evaluate the model")
        theta = numpyro.sample("theta", dist.Normal(0., 1.))
        numpyro.sample("rv", dist.Normal(0., 1.).expand((2,)).to_event(1))
        numpyro.sample("jitter", dist.LogNormal(-3., .3))
        numpyro.deterministic("derived", 2 * theta)
        numpyro.deterministic("fixed", 7.)

    mcmc = fit.run_nuts(
        model=model, rng_key=jax.random.PRNGKey(142), num_warmup=40,
        num_samples=80, num_chains=2, progress_bar=False,
        mcmc_kwargs={"chain_method": "sequential"})
    guard["allow_evaluation"] = False
    return fit, model, mcmc


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_default_trace_has_all_sampled_components_and_chains(sampled):
    fit, _, mcmc = sampled
    # Even a caller's small ArviZ limit must not silently drop sampled sites.
    with az.rc_context({"plot.max_subplots": 2}):
        figures = fit.plot_mcmc_diagnostics(mcmc)
        assert az.rcParams["plot.max_subplots"] == 2
    assert set(figures) == {"trace", "corner"}
    assert len(figures["trace"].axes) == 8  # theta, jitter, rv[8], rv[9]
    assert len(figures["corner"].axes) == 16
    axes = np.array(figures["trace"].axes).reshape(-1, 2)
    titles = [marginal.get_title() or marginal.get_xlabel() for marginal, _ in axes]
    assert set(titles) == {"theta", "jitter", "rv\n8", "rv\n9"}
    samples = mcmc.get_samples(group_by_chain=True)
    for title, (_, trace) in zip(titles, axes):
        if title.startswith("rv"):
            expected = np.asarray(samples["rv"])[..., [8, 9].index(int(title[-1]))]
        else:
            expected = np.asarray(samples[title])
        chains = [line.get_ydata() for line in trace.lines if len(line.get_ydata()) == 80]
        np.testing.assert_allclose(chains, expected)
    # Both APIs should show marginal densities, not the 1.x ECDF default.
    marginal = axes[titles.index("theta"), 0]
    assert any(np.any(np.diff(line.get_ydata()) > 0)
               and np.any(np.diff(line.get_ydata()) < 0)
               for line in marginal.lines)


def test_saved_results_and_netcdf_keep_latent_defaults(sampled, tmp_path):
    fit, model, mcmc = sampled
    output = save_fit_result(tmp_path / "run", sf=fit, model=model, mcmc=mcmc)
    result = load_fit_result(output)
    for source in (result, result.idata):
        figures = plot_mcmc_diagnostics(source, corner_vars="theta")
        assert len(figures["trace"].axes) == 8
        assert len(figures["corner"].axes) == 1
        for fig in figures.values():
            plt.close(fig)
    # Earlier bundles keep latent names in the manifest only.
    result.idata.posterior.attrs.clear()
    figures = plot_mcmc_diagnostics(result, corner_vars="theta")
    assert len(figures["trace"].axes) == 8


def test_log_copy_keeps_divergences_coordinates_and_native_selection(monkeypatch, arviz_from_dict):
    import corner

    rng = np.random.default_rng(51)
    idata = arviz_from_dict(
        posterior={"jitter": np.exp(rng.normal(-3., .3, (2, 80))),
                   "rv": rng.normal(size=(2, 80, 2)),
                   "fixed": np.ones((2, 80))},
        sample_stats={"diverging": rng.random((2, 80)) < .03},
        coords={"region": ["blue", "red"]}, dims={"rv": ["region"]})
    # Both plotting paths must align chain/draw, even for transposed input.
    idata.posterior["rv"] = idata.posterior.rv.transpose("region", "draw", "chain")
    original = idata.copy()
    corner_call = corner.corner
    seen = {}

    def capture(data, **kwargs):
        seen["data"], seen["kwargs"] = data, kwargs
        return corner_call(data, **kwargs)

    monkeypatch.setattr(corner, "corner", capture)
    figures = plot_mcmc_diagnostics(
        idata, trace_vars=["rv", "jitter"], corner_vars=["jitter", "fixed"],
        log_vars="jitter", labels={"jitter": "lnsigma"})
    assert len(figures["trace"].axes) == 6
    assert len(figures["corner"].axes) == 4
    assert "lnsigma" in [ax.get_title() or ax.get_xlabel() for ax in figures["trace"].axes]
    np.testing.assert_allclose(seen["data"].posterior.jitter, np.log(original.posterior.jitter))
    xr.testing.assert_identical(seen["data"].sample_stats, original.sample_stats)
    assert seen["kwargs"]["divergences"] is True
    assert seen["kwargs"]["var_names"] == ["jitter", "fixed"]
    assert seen["data"].posterior.region.values.tolist() == ["blue", "red"]
    xr.testing.assert_identical(idata.posterior, original.posterior)
    xr.testing.assert_identical(idata.sample_stats, original.sample_stats)


def test_external_idata_defaults_to_all_outputs_and_kwargs_select_coordinates(arviz_from_dict):
    rng = np.random.default_rng(9)
    idata = arviz_from_dict(posterior={"rv": rng.normal(size=(1, 60, 2)),
                                    "fixed": np.ones((1, 60))},
                         coords={"region": [8, 9]}, dims={"rv": ["region"]})
    figures = plot_mcmc_diagnostics(
        idata, corner_kwargs={"coords": {"region": [9]}, "color": "C2"})
    assert len(figures["trace"].axes) == 6
    assert len(figures["corner"].axes) == 4
    labels = [ax.get_xlabel() for ax in figures["corner"].axes]
    assert "rv\n9" in labels and "rv\n8" not in labels


@pytest.mark.parametrize("options, error, message", [
    ({"trace_vars": ["missing"]}, KeyError, "saved posterior"),
    ({"trace_vars": []}, ValueError, "variable names"),
    ({"corner_vars": ["x", "x"]}, ValueError, "distinct"),
    ({"log_vars": ["x"]}, ValueError, "positive"),
    ({"trace_kwargs": {"var_names": ["x"]}}, ValueError, "trace_vars"),
])
def test_invalid_selections_and_log_values_fail_before_plotting(options, error, message, arviz_from_dict):
    idata = arviz_from_dict(posterior={"x": np.linspace(-1, 1, 50)[None]})
    with pytest.raises(error, match=message):
        plot_mcmc_diagnostics(idata, **options)
    assert not plt.get_fignums()


def test_corner_failure_closes_only_new_figures(monkeypatch, arviz_from_dict):
    import corner

    existing = plt.figure()
    idata = arviz_from_dict(posterior={"x": np.linspace(-1, 1, 50)[None]})

    def fail(*args, **kwargs):
        plt.figure()
        raise RuntimeError("failed plot")

    monkeypatch.setattr(corner, "corner", fail)
    with pytest.raises(RuntimeError, match="failed plot"):
        plot_mcmc_diagnostics(idata)
    assert plt.get_fignums() == [existing.number]
