"""NumPyro convenience mapping, normalized factor and cheap inference checks."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.util import log_density
import pytest

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (Observation, SpecModel, model_single, single_star_params,
    marginalized_continuum_log_likelihood, continuum_posterior, chebyshev_basis)
from jaxstar.specfit._data import _SpectralLibrary


@pytest.fixture
def case():
    wave = np.array([5000., 6000.])[:, None] * np.exp(np.linspace(-250., 250., 601) / 299792.458)
    profile = np.exp(-.5 * (np.linspace(-250., 250., 601) / 4.)**2)
    flux = np.stack([np.broadcast_to(1-depth*profile, wave.shape) for depth in (.2, .5)])
    grid = RectilinearGrid(axes={"teff": np.array([5000., 6500.])},
        fields={"flux": Field(flux, ("teff",), payload_dims=("region", "pixel"))})
    lib = _SpectralLibrary(grid, wave, (8, 9), "test", "vacuum", "normalized")
    model = SpecModel(lib, vmax=30.)
    wave_obs = np.array([5000., 6000.])[:, None] * np.exp(np.linspace(-80.37, 80.19, 48) / 299792.458)
    point = {"teff": 5800., "vsini": 6., "vmacro": 3., "q1": .36, "q2": .3,
             "rv": 2., "resolving_power": 70000.}
    params = single_star_params({"teff": point["teff"]}, **{k: v for k,v in point.items() if k != "teff"})
    physical = model(params, wave_obs)
    mask = np.zeros(wave_obs.shape, dtype=bool)
    mask[:, 3] = True
    measured, error = np.asarray(physical).copy(), np.full(wave_obs.shape, .01)
    measured[mask], error[mask] = np.nan, np.inf
    obs = Observation(wave_obs, measured, error, mask, region=(8, 9))
    kwargs = dict(atmosphere={"teff": dist.Uniform(5000., 6500.)},
        broadening={name: point[name] for name in ("vsini", "vmacro", "q1", "q2")},
        rv=point["rv"], resolving_power=point["resolving_power"],
        sigma_constant=.1, sigma_continuum=.03, jitter=.004,
        basis=chebyshev_basis(obs.wavelength))
    return obs, model, kwargs, point, physical


def trace_at(obs, model, kwargs, values):
    return handlers.trace(handlers.substitute(model_single, data=values)).get_trace(obs, model, **kwargs)


def test_trace_mapping_and_no_continuum_latents(case):
    obs, model, kwargs, point, physical = case
    trace = trace_at(obs, model, kwargs, point)
    latent = {name for name, site in trace.items() if site["type"] == "sample" and not site["is_observed"]}
    assert latent == {"teff"}
    assert {"u1", "u2", "jitter", "sigma_continuum", "spectrum"} <= set(trace)
    assert "model_flux" not in trace
    np.testing.assert_allclose(trace["u1"]["value"], .36, atol=1e-7)
    np.testing.assert_allclose(trace["u2"]["value"], .24, atol=1e-7)
    actual = trace["spectrum"]["fn"].log_prob(trace["spectrum"]["value"])
    expected = marginalized_continuum_log_likelihood(obs, physical, sigma_constant=.1,
        sigma_continuum=.03, jitter=.004, basis=kwargs["basis"])
    np.testing.assert_allclose(actual, expected, atol=3e-4)
    saved = trace_at(obs, model, dict(kwargs, save_model_flux=True), point)
    np.testing.assert_allclose(saved["model_flux"]["value"], physical, atol=2e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log_density_jit_grad_masks_and_truth_preference(case, dtype, x64_context):
    with x64_context(dtype == "float64"):
        obs, model, kwargs, point, _ = case
        def density(values, obs, model):
            return log_density(model_single, (obs, model), kwargs, values)[0]
        theta = {"teff": jnp.asarray(point["teff"])}
        eager = density(theta, obs, model)
        value, grad = jax.jit(jax.value_and_grad(density))(theta, obs, model)
        assert np.isfinite(value) and np.isfinite(grad["teff"])
        np.testing.assert_allclose(value, eager, atol=1e-3 if dtype == "float32" else 1e-9)
        assert value > density({"teff": 6300.}, obs, model) + 3
        clean = Observation(obs.wavelength, np.where(obs.mask, 1., obs.flux),
                            np.where(obs.mask, .1, obs.uncertainty), obs.mask, region=obs.region)
        np.testing.assert_allclose(density(theta, clean, model), eager, atol=1e-5)


def test_sampled_hyperparameters_and_region_vectors(case):
    obs, model, kwargs, point, _ = case
    config = dict(kwargs, jitter=dist.HalfNormal(jnp.array([.01, .02])),
                  sigma_continuum=dist.LogNormal(jnp.log(.03), .5),
                  rv=dist.Uniform(jnp.array([0., 0.]), jnp.array([4., 4.])),
                  resolving_power=dist.Uniform(jnp.array([65000., 65000.]), jnp.array([75000., 75000.])))
    values = {"teff": 5800., "jitter": jnp.array([.004, .008]), "sigma_continuum": .025,
              "rv": jnp.array([2., 2.]), "resolving_power": jnp.array([70000., 70000.])}
    trace = trace_at(obs, model, config, values)
    for name in ("jitter", "rv", "resolving_power"):
        assert trace[name]["fn"].event_shape == (2,)
    params = single_star_params({"teff": 5800.}, **kwargs["broadening"],
                                rv=values["rv"], resolving_power=values["resolving_power"])
    physical = model(params, obs.wavelength)
    expected = marginalized_continuum_log_likelihood(obs, physical, sigma_constant=.1,
        sigma_continuum=.025, jitter=values["jitter"], basis=kwargs["basis"])
    np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, expected, atol=2e-4)
    fn = jax.jit(jax.value_and_grad(lambda v: log_density(model_single, (obs, model), config, v)[0]))
    value, gradient = fn(values)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(gradient))
    # The same fixed-jitter covariance is used for conditional reconstruction.
    post = continuum_posterior(obs, physical, sigma_constant=.1,
                              sigma_continuum=.025, jitter=values["jitter"])
    assert np.all(np.isfinite(post.mean))


@pytest.mark.parametrize("change,message", [
    ({"atmosphere": {"teff": np.ones(2)}}, "scalar"),
    ({"atmosphere": {}}, "named axes"),
    ({"jitter": np.ones((2, 48))}, "scalar or shape"),
    ({"broadening": {"vsini": 6.}}, "exactly"),
    ({"broadening": {"vsini": 6., "vmacro": 3., "q1": 1.1, "q2": .3}}, "q1 and q2"),
])
def test_invalid_config(case, change, message):
    obs, model, kwargs, point, _ = case
    with pytest.raises(ValueError, match=message):
        trace_at(obs, model, dict(kwargs, **change), {"teff": point["teff"]} if "atmosphere" not in change else {})


def test_region_alignment_and_distinct_pixel_grid(case):
    obs, model, kwargs, point, _ = case
    source = np.asarray(model.spectra.wavelength)
    assert obs.n_pixels != source.shape[-1]
    assert np.all(np.min(np.abs(obs.wavelength[..., None] - source[:, None, :]), axis=-1) > 0)
    swapped = Observation(obs.wavelength, obs.flux, obs.uncertainty, obs.mask, region=(9, 8))
    with pytest.raises(ValueError, match="row order"):
        trace_at(swapped, model, kwargs, point)


def test_small_svi_and_nuts_same_model(case):
    obs, model, kwargs, point, _ = case
    guide = AutoLaplaceApproximation(model_single, init_loc_fn=init_to_value(values={"teff": 5600.}))
    svi = SVI(model_single, guide, numpyro.optim.Adam(.04), Trace_ELBO(), **kwargs)
    result = svi.run(jax.random.PRNGKey(0), 100, obs, model, progress_bar=False)
    assert np.all(np.isfinite(result.losses))
    assert result.losses[-1] < result.losses[0]
    estimate = guide.median(result.params)
    assert abs(float(estimate["teff"]) - point["teff"]) < 80.
    mcmc = MCMC(NUTS(model_single, init_strategy=init_to_value(values=estimate)),
                num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1), obs, model, **kwargs)
    samples = mcmc.get_samples()
    assert np.all(np.isfinite(samples["teff"]))
    assert abs(float(jnp.mean(samples["teff"])) - point["teff"]) < 150.
    assert not {"norm", "slope", "coefficients"}.intersection(samples)


def test_inferutils_optional_smoke(case):
    inferutils = pytest.importorskip("numpyro_inferutils")
    obs, model, kwargs, point, _ = case
    estimate = inferutils.find_map_svi(model_single, .04, 100,
        rng_key=jax.random.PRNGKey(0), p_initial={"teff": 5600.}, progress_bar=False,
        observation=obs, specmodel=model, **kwargs)
    assert abs(float(estimate["teff"]) - point["teff"]) < 80.


def test_custom_model_uses_same_low_level_joint_density(case):
    obs, model, kwargs, point, _ = case
    config = dict(kwargs, jitter=dist.HalfNormal(.01),
                  sigma_continuum=dist.LogNormal(jnp.log(.03), .5))
    def custom(obs, model):
        teff = numpyro.sample("teff", config["atmosphere"]["teff"])
        jitter = numpyro.sample("jitter", config["jitter"])
        sc = numpyro.sample("sigma_continuum", config["sigma_continuum"])
        params = single_star_params({"teff": teff}, **config["broadening"],
                                    rv=config["rv"], resolving_power=config["resolving_power"])
        numpyro.factor("spectrum", marginalized_continuum_log_likelihood(
            obs, model(params, obs.wavelength), basis=config["basis"],
            sigma_constant=.1, sigma_continuum=sc, jitter=jitter))
    theta = {"teff": point["teff"], "jitter": .004, "sigma_continuum": .025}
    standard = log_density(model_single, (obs, model), config, theta)[0]
    value, gradient = jax.jit(jax.value_and_grad(
        lambda t, o, m: log_density(custom, (o, m), {}, t)[0]))(theta, obs, model)
    np.testing.assert_allclose(value, standard, atol=3e-4)
    assert all(np.all(np.isfinite(g)) for g in gradient.values())


def test_single_region_1d_observation(case):
    from dataclasses import replace
    obs, model, kwargs, point, _ = case
    library = model.spectra
    field = library.grid.field("flux")
    grid = RectilinearGrid(axes={"teff": library.grid.axis("teff")},
        fields={"flux": Field(field.values[..., :1, :], field.dims, payload_dims=field.payload_dims)})
    single_model = SpecModel(replace(library, grid=grid, wavelength=np.asarray(library.wavelength)[:1], regions=(8,)), vmax=30.)
    single_obs = Observation(obs.wavelength[0], obs.flux[0], obs.uncertainty[0], obs.mask[0], region=(8,))
    config = dict(kwargs, basis=chebyshev_basis(single_obs.wavelength), save_model_flux=True)
    trace = trace_at(single_obs, single_model, config, {"teff": point["teff"]})
    assert trace["model_flux"]["value"].shape == single_obs.shape
    assert np.isfinite(log_density(model_single, (single_obs, single_model), config, {"teff": point["teff"]})[0])
