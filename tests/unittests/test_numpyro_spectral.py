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
    marginalized_continuum_log_likelihood, continuum_posterior, chebyshev_basis,
    prepare_gp_observation, gp_marginalized_continuum_log_likelihood)
from jaxstar.specfit._data import _SpectralLibrary


def _case():
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


@pytest.fixture
def case():
    return _case()


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


@pytest.fixture
def gp_case(x64_context):
    pytest.importorskip("tinygp")
    with x64_context():
        obs, model, kwargs, point, physical = _case()
        mask = obs.mask.copy()
        mask[:, 7:10] = True
        y, error = obs.flux.copy(), obs.uncertainty.copy()
        y[mask], error[mask] = np.nan, np.inf
        obs = Observation(obs.wavelength, y, error, mask, region=obs.region)
        config = dict(kwargs, gp_amplitude=.015, gp_scale=.3)
        yield obs, model, config, point, physical


@pytest.mark.parametrize("solver", ["auto", "quasisep", "direct"])
def test_gp_fixed_factor_and_compact_trace(gp_case, solver):
    import inspect
    obs, model, config, point, physical = gp_case
    assert "gp_data" not in inspect.signature(model_single).parameters
    config = dict(config, gp_solver=solver, jitter=np.array([.004, .008]))
    trace = trace_at(obs, model, config, point)
    expected = gp_marginalized_continuum_log_likelihood(
        prepare_gp_observation(obs), physical, gp_amplitude=.015, gp_scale=.3,
        sigma_constant=.1, sigma_continuum=.03, jitter=config["jitter"],
        basis=config["basis"], gp_solver=solver)
    np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, expected, atol=1e-11)
    assert trace["gp_amplitude"]["type"] == trace["gp_scale"]["type"] == "deterministic"
    assert {name for name, site in trace.items()
            if site["type"] == "sample" and not site["is_observed"]} == {"teff"}
    assert set(trace) == {"teff", "vsini", "vmacro", "q1", "q2", "rv",
        "resolving_power", "sigma_constant", "sigma_continuum", "jitter",
        "gp_amplitude", "gp_scale", "u1", "u2", "spectrum"}
    saved = trace_at(obs, model, dict(config, save_model_flux=True), point)
    np.testing.assert_allclose(saved["model_flux"]["value"], physical, atol=1e-12)
    params = handlers.substitute(model_single, data=point)(obs, model, **config)
    expected_params = single_star_params({"teff": point["teff"]},
        **{k: v for k, v in point.items() if k != "teff"})
    for actual, expected in zip(jax.tree.leaves(params), jax.tree.leaves(expected_params)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gp_sampled_sites_concrete_jit_and_joint_density(solver, dtype, x64_context):
    pytest.importorskip("tinygp")
    with x64_context(dtype == "float64"):
        obs, model, kwargs, point, physical = _case()
        data = prepare_gp_observation(obs)
        config = dict(kwargs, gp_solver=solver,
            gp_amplitude=dist.LogNormal(jnp.log(.015), .4),
            gp_scale=dist.LogNormal(jnp.log(.3), .4),
            jitter=dist.HalfNormal(jnp.array([.01, .02])),
            sigma_continuum=dist.LogNormal(jnp.log(.03), .5))
        theta = dict(teff=5800., gp_amplitude=.015, gp_scale=.3,
                     jitter=jnp.array([.004, .008]), sigma_continuum=.025)
        trace = trace_at(obs, model, config, theta)
        latent = {n for n, s in trace.items() if s["type"] == "sample" and not s["is_observed"]}
        assert latent == set(theta)
        assert trace["gp_amplitude"]["fn"].event_shape == trace["gp_scale"]["fn"].event_shape == ()
        assert not {"lna", "lnc", "save_pred", "model_flux"}.intersection(trace)
        expected = gp_marginalized_continuum_log_likelihood(
            data, physical, gp_amplitude=.015, gp_scale=.3, sigma_constant=.1,
            sigma_continuum=.025, jitter=theta["jitter"], basis=config["basis"], gp_solver=solver)
        for name in theta:
            prior = config["atmosphere"][name] if name == "teff" else config[name]
            expected += jnp.sum(prior.log_prob(theta[name]))
        def density(values, specmodel):
            # Only Observation is concrete; large spectral-library arrays can
            # still be passed as dynamic JIT data through SpecModel.
            return log_density(model_single, (obs, specmodel), config, values)[0]
        eager, eager_grad = jax.value_and_grad(density)(theta, model)
        actual, gradient = jax.jit(jax.value_and_grad(density))(theta, model)
        tol = 1e-3 if dtype == "float32" else 1e-10
        np.testing.assert_allclose(actual, expected, atol=tol)
        np.testing.assert_allclose(actual, eager, atol=tol)
        assert actual.dtype == jnp.dtype(dtype)
        assert all(np.all(np.isfinite(x)) for x in gradient.values())
        for name in theta:
            np.testing.assert_allclose(gradient[name], eager_grad[name], atol=tol)
        # Different masked poison must not change inference or its gradients.
        changed = Observation(obs.wavelength, np.where(obs.mask, np.inf, obs.flux),
                              np.where(obs.mask, np.nan, obs.uncertainty),
                              obs.mask, region=obs.region)
        def changed_density(values, specmodel):
            return log_density(model_single, (changed, specmodel), config, values)[0]
        other, other_grad = jax.jit(jax.value_and_grad(changed_density))(theta, model)
        np.testing.assert_allclose(other, actual, atol=tol)
        for name in theta:
            np.testing.assert_allclose(other_grad[name], gradient[name], atol=tol)


@pytest.mark.parametrize("solver", ["quasisep", "direct"])
def test_gp_svi_and_nuts_masked_sampled_hyperparameters(gp_case, solver):
    from numpyro.infer.autoguide import AutoNormal
    obs, model, config, point, _ = gp_case
    config = dict(config, gp_amplitude=dist.LogNormal(jnp.log(.015), .4),
                  gp_scale=dist.LogNormal(jnp.log(.3), .4), gp_solver=solver)
    initial = dict(teff=point["teff"], gp_amplitude=.015, gp_scale=.3)
    guide = AutoNormal(model_single, init_loc_fn=init_to_value(values=initial))
    svi = SVI(model_single, guide, numpyro.optim.Adam(.02), Trace_ELBO(), **config)
    result = svi.run(jax.random.PRNGKey(12), 20, obs, model, progress_bar=False)
    assert np.all(np.isfinite(result.losses))
    estimate = guide.median(result.params)
    assert set(initial).issubset(estimate)
    assert not {"model_flux", "gp_mean", "coefficients"}.intersection(estimate)
    assert float(estimate["gp_amplitude"]) > 0 and float(estimate["gp_scale"]) > 0
    sampler = MCMC(NUTS(model_single, init_strategy=init_to_value(values=initial)),
                   num_warmup=8, num_samples=8, progress_bar=False)
    sampler.run(jax.random.PRNGKey(13), obs, model, **config)
    samples = sampler.get_samples()
    assert set(initial).issubset(samples)
    assert not {"model_flux", "gp_mean", "coefficients"}.intersection(samples)
    assert all(np.all(np.isfinite(x)) for x in samples.values())
    assert np.all(samples["gp_amplitude"] > 0) and np.all(samples["gp_scale"] > 0)


@pytest.mark.parametrize("change,error,message", [
    ({"gp_scale": None}, ValueError, "both gp_amplitude and gp_scale"),
    ({"gp_amplitude": None}, ValueError, "both gp_amplitude and gp_scale"),
    ({"gp_amplitude": np.array([.01, .02])}, ValueError, "gp_amplitude must be scalar"),
    ({"gp_scale": dist.LogNormal(jnp.zeros(2), .4)}, ValueError, "gp_scale must be scalar"),
    ({"gp_amplitude": 0.}, ValueError, "gp_amplitude must be finite and positive"),
    ({"gp_scale": np.inf}, ValueError, "gp_scale must be finite and positive"),
    ({"gp_solver": "unknown"}, ValueError, "gp_solver must be"),
])
def test_gp_configuration_errors(gp_case, change, error, message):
    obs, model, config, point, _ = gp_case
    with pytest.raises(error, match=message):
        seeded = handlers.seed(handlers.substitute(model_single, data=point), 0)
        handlers.trace(seeded).get_trace(obs, model, **dict(config, **change))


def test_gp_internal_preparation_uses_current_mask_and_data(gp_case):
    obs, model, config, point, physical = gp_case
    changed_mask = obs.mask.copy()
    changed_mask[:, 14:17] = True
    changed = Observation(obs.wavelength, obs.flux, obs.uncertainty, changed_mask, region=obs.region)
    trace = trace_at(changed, model, config, point)
    expected = gp_marginalized_continuum_log_likelihood(prepare_gp_observation(changed), physical,
        gp_amplitude=.015, gp_scale=.3, sigma_constant=.1, sigma_continuum=.03,
        jitter=.004, basis=config["basis"])
    np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, expected, atol=1e-11)
    assert trace["spectrum"]["fn"].log_factor != trace_at(obs, model, config, point)["spectrum"]["fn"].log_factor
    # Changing data or masks requires no separately managed preparation cache.
    updated = Observation(obs.wavelength, obs.flux + .002, obs.uncertainty * 1.2,
                          obs.mask, region=obs.region)
    trace = trace_at(updated, model, config, point)
    expected = gp_marginalized_continuum_log_likelihood(prepare_gp_observation(updated), physical,
        gp_amplitude=.015, gp_scale=.3, sigma_constant=.1, sigma_continuum=.03,
        jitter=.004, basis=config["basis"])
    np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, expected, atol=1e-11)


def test_gp_dynamic_observation_is_intentionally_unsupported(gp_case):
    obs, model, config, point, _ = gp_case
    assert np.any(obs.mask)
    fn = jax.jit(jax.value_and_grad(lambda p, o, m: log_density(
        model_single, (o, m), config, p)[0]))
    with pytest.raises(ValueError, match="GP masks must be fixed.*Ordinary SVI/NUTS.*jit_model_args=True"):
        fn({"teff": point["teff"]}, obs, model)


def test_iid_nuts_preserves_dynamic_model_arguments(case):
    from functools import partial
    obs, model, config, point, _ = case
    sampler = MCMC(NUTS(partial(model_single, **config),
                        init_strategy=init_to_value(values={"teff": point["teff"]})),
                   num_warmup=8, num_samples=8, progress_bar=False, jit_model_args=True)
    sampler.run(jax.random.PRNGKey(14), obs, model)
    assert np.all(np.isfinite(sampler.get_samples()["teff"]))


@pytest.mark.parametrize("dtype,value,jitter_gradient,sc_gradient,teff_gradient", [
    ("float32", 309.6420593261719, [-1489.8358154296875, -2071.884765625],
     -326.5060119628906, 7.903543883003294e-05),
    ("float64", 309.6426378883581, [-1489.8364308461478, -2071.8847537307847],
     -326.50523535342893, 7.886845737471048e-05),
])
def test_iid_pre_gp_integration_reference(dtype, value, jitter_gradient, sc_gradient,
                                          teff_gradient, x64_context):
    with x64_context(dtype == "float64"):
        obs, model, kwargs, point, _ = _case()
        config = dict(kwargs, jitter=dist.HalfNormal(jnp.array([.01, .02])),
                      sigma_continuum=dist.LogNormal(jnp.log(.03), .5))
        theta = dict(teff=point["teff"], jitter=jnp.array([.004, .008]), sigma_continuum=.025)
        fn = jax.jit(jax.value_and_grad(lambda p, o, m: log_density(
            model_single, (o, m), config, p)[0]))
        actual, gradients = fn(theta, obs, model)
        # Captured from 85439c7 before adding any GP arguments/branches.
        tol = dict(rtol=3e-6, atol=2e-7) if dtype == "float32" else dict(rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual, value, **tol)
        np.testing.assert_allclose(gradients["jitter"], jitter_gradient, **tol)
        np.testing.assert_allclose(gradients["sigma_continuum"], sc_gradient, **tol)
        np.testing.assert_allclose(gradients["teff"], teff_gradient, **tol)
        trace = trace_at(obs, model, config, theta)
        assert list(trace) == ["teff", "vsini", "vmacro", "q1", "q2", "rv",
            "resolving_power", "sigma_constant", "sigma_continuum", "jitter", "u1", "u2", "spectrum"]


def test_gp_single_region_1d_observation(gp_case):
    from dataclasses import replace
    obs, model, config, point, _ = gp_case
    library = model.spectra
    field = library.grid.field("flux")
    grid = RectilinearGrid(axes={"teff": library.grid.axis("teff")},
        fields={"flux": Field(field.values[..., :1, :], field.dims, payload_dims=field.payload_dims)})
    single_model = SpecModel(replace(library, grid=grid,
        wavelength=np.asarray(library.wavelength)[:1], regions=(8,)), vmax=30.)
    single_obs = Observation(obs.wavelength[0], obs.flux[0], obs.uncertainty[0],
                             obs.mask[0], region=(8,))
    config = dict(config, basis=chebyshev_basis(single_obs.wavelength), save_model_flux=True)
    trace = trace_at(single_obs, single_model, config, point)
    flux = trace["model_flux"]["value"]
    assert flux.shape == single_obs.shape
    expected = gp_marginalized_continuum_log_likelihood(prepare_gp_observation(single_obs), flux,
        gp_amplitude=.015, gp_scale=.3, sigma_constant=.1, sigma_continuum=.03,
        jitter=.004, basis=config["basis"])
    np.testing.assert_allclose(trace["spectrum"]["fn"].log_factor, expected, atol=1e-11)
