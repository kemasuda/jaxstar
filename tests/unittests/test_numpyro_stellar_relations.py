"""Stellar-relation modes through the real NumPyro, SpecModel and likelihood APIs."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model, log_density
import pytest

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (
    Observation, SpecModel, model_single, single_star_params, chebyshev_basis,
    marginalized_continuum_log_likelihood, physical_logg_max, empirical_vmic,
    empirical_vmacro_valenti_fischer2005, load_spectral_grid, save_spectral_grid,
)
from jaxstar.specfit._data import _SpectralLibrary


@pytest.fixture
def stellar_case():
    # Deliberately shuffled axes: dependencies must not rely on storage order.
    axes = {"vmic": [.5, 2.5], "alpha": [0.], "logg": [3.5, 5.],
            "mh": [-.5, .5], "carbon": [0.], "teff": [4000., 7500.]}
    nodes = dict(zip(axes, np.meshgrid(*axes.values(), indexing="ij")))
    depth = (.2 + .04 * (nodes["teff"] - 4000.) / 3500.
             + .02 * (nodes["logg"] - 3.5) + .03 * nodes["mh"]
             + .08 * nodes["vmic"])
    velocity = np.linspace(-200., 200., 401)
    wave = np.array([5000., 6000.])[:, None] * np.exp(velocity / 299792.458)
    profile = np.exp(-.5 * (velocity / 5.)**2)
    flux = np.broadcast_to(1 - depth[..., None, None] * profile,
                           depth.shape + wave.shape).copy()
    grid = RectilinearGrid(axes=axes, fields={
        "flux": Field(flux, tuple(axes), payload_dims=("region", "pixel"))})
    library = _SpectralLibrary(grid, wave, (8, 9), "bosz", "vacuum", "normalized")
    model = SpecModel(library, vmax=30.)
    coordinates = {"teff": 5800., "logg": 4.2, "mh": -.1, "vmic": 1.2,
                   "alpha": 0., "carbon": 0.}
    broadening = {"vsini": 6., "vmacro": 3., "q1": .36, "q2": .3}
    params = single_star_params(coordinates, **broadening, rv=2., resolving_power=70000.)
    wave_obs = np.array([5000., 6000.])[:, None] * np.exp(
        np.linspace(-60.37, 60.19, 24) / 299792.458)
    obs = Observation(wave_obs, model(params, wave_obs), np.full(wave_obs.shape, .02),
                      region=(8, 9))
    config = dict(
        atmosphere={"vmic": dist.Uniform(.7, 1.7), "alpha": 0.,
                    "logg": dist.Uniform(3.8, 4.9), "mh": dist.Uniform(-.3, .3),
                    "carbon": 0., "teff": dist.Uniform(5000., 6500.)},
        broadening=broadening, rv=2., resolving_power=70000.,
        sigma_constant=.1, sigma_continuum=.03, jitter=.004,
        basis=chebyshev_basis(obs.wavelength), save_model_flux=True,
    )
    return obs, model, config, coordinates


def trace_at(obs, model, config, coordinates):
    return handlers.trace(handlers.substitute(model_single, data=coordinates)).get_trace(
        obs, model, **config)


def derived_config(config, **options):
    atmosphere = {k: v for k, v in config["atmosphere"].items() if k != "vmic"}
    return dict(config, atmosphere=atmosphere, **options)


def without_axis(model, name):
    library = model.spectra
    field = library.grid.field("flux")
    names = library.grid.axis_names
    values = np.take(np.asarray(field.values), 0, axis=names.index(name))
    remaining = tuple(k for k in names if k != name)
    grid = RectilinearGrid(axes={k: library.grid.axis(k) for k in remaining}, fields={
        "flux": Field(values, remaining, payload_dims=field.payload_dims)})
    return SpecModel(replace(library, grid=grid), vmax=30.)


def test_disabled_options_preserve_sites_spectrum_and_joint_density(stellar_case):
    obs, model, config, coordinates = stellar_case
    config = dict(config, use_physical_logg_max=False, use_empirical_vmic=False,
                  use_empirical_vmacro=False)
    trace = trace_at(obs, model, config, coordinates)
    expected_sites = set(model.spectra.grid.axis_names) | {
        "vsini", "vmacro", "q1", "q2", "rv", "resolving_power", "sigma_constant",
        "sigma_continuum", "jitter", "u1", "u2", "model_flux", "spectrum"}
    assert set(trace) == expected_sites
    assert {name for name, site in trace.items()
            if site["type"] == "sample" and not site["is_observed"]} == {
                "teff", "logg", "mh", "vmic"}
    params = single_star_params(coordinates, **config["broadening"],
                               rv=config["rv"], resolving_power=config["resolving_power"])
    flux = model(params, obs.wavelength)
    np.testing.assert_array_equal(trace["model_flux"]["value"], flux)
    expected = marginalized_continuum_log_likelihood(
        obs, flux, basis=config["basis"], sigma_constant=.1, sigma_continuum=.03, jitter=.004)
    latents = {k: coordinates[k] for k, v in config["atmosphere"].items()
               if isinstance(v, dist.Distribution)}
    expected += sum(config["atmosphere"][k].log_prob(v) for k, v in latents.items())
    np.testing.assert_allclose(log_density(model_single, (obs, model), config, latents)[0],
                               expected, atol=1e-5)


@pytest.mark.parametrize("temperature", [5200., 6300., 7200.])
def test_physical_logg_uses_sampled_teff_and_replaces_uniform_upper(stellar_case, temperature):
    obs, model, config, coordinates = stellar_case
    config = dict(config, use_empirical_vmic=False, use_physical_logg_max=True,
                  atmosphere=dict(config["atmosphere"], logg=dist.Uniform(3.8, 4.0),
                                  teff=dist.Uniform(temperature - 100., temperature + 100.)))
    trace = trace_at(obs, model, config, dict(coordinates, teff=temperature))
    assert trace["logg"]["type"] == "sample"
    np.testing.assert_allclose(trace["logg"]["fn"].low, 3.8, atol=2e-7)
    np.testing.assert_allclose(trace["logg"]["fn"].high, physical_logg_max(temperature))
    assert "logg_scaled" not in trace


@pytest.mark.parametrize("gravity,valid", [(4.2, True), (4.9, False), (np.nan, False)])
def test_physical_logg_fixed_values(stellar_case, gravity, valid):
    obs, model, config, coordinates = stellar_case
    config = dict(config, use_empirical_vmic=False, use_physical_logg_max=True,
                  atmosphere=dict(config["atmosphere"], logg=gravity))
    values = {k: v for k, v in coordinates.items() if k != "logg"}
    if valid:
        trace = trace_at(obs, model, config, values)
        assert trace["logg"]["type"] == "deterministic"
        np.testing.assert_allclose(trace["logg"]["value"], gravity)
    else:
        with pytest.raises(ValueError, match="fixed logg.*physical_logg_max"):
            trace_at(obs, model, config, values)


@pytest.mark.parametrize("prior,message", [
    (dist.Uniform(4.9, 5.), "lower bound.*below physical_logg_max"),
    (dist.Uniform(np.inf, np.inf), "lower bound.*below physical_logg_max"),
    (dist.Normal(4.2, .2), "supports only a scalar Uniform"),
    (dist.Uniform(jnp.array([3.8]), jnp.array([4.9])), "logg must be scalar"),
])
def test_physical_logg_invalid_priors(stellar_case, prior, message):
    obs, model, config, coordinates = stellar_case
    config = dict(config, use_empirical_vmic=False, use_physical_logg_max=True,
                  atmosphere=dict(config["atmosphere"], logg=prior))
    with pytest.raises(ValueError, match=message):
        trace_at(obs, model, config, coordinates)


@pytest.mark.parametrize("mode", [None, True])
def test_bosz_empirical_vmic_is_deterministic_and_reaches_specmodel(stellar_case, mode):
    obs, model, config, coordinates = stellar_case
    config = derived_config(config, use_empirical_vmic=mode)
    values = {k: v for k, v in coordinates.items() if k != "vmic"}
    trace = trace_at(obs, model, config, values)
    expected_vmic = empirical_vmic(values["teff"], values["logg"], values["mh"])
    assert trace["vmic"]["type"] == "deterministic"
    np.testing.assert_array_equal(trace["vmic"]["value"], expected_vmic)
    params = handlers.substitute(model_single, data=values)(obs, model, **config)
    np.testing.assert_array_equal(params["components"][0]["atmosphere"]["vmic"], expected_vmic)
    expected_flux = model(params, obs.wavelength)
    np.testing.assert_array_equal(trace["model_flux"]["value"], expected_flux)
    explicit = dict(config, use_empirical_vmic=False, atmosphere=dict(config["atmosphere"], vmic=2.))
    assert not np.allclose(trace["model_flux"]["value"],
                           trace_at(obs, model, explicit, values)["model_flux"]["value"])


def test_bosz_identity_survives_prepared_storage(stellar_case, tmp_path):
    obs, model, config, coordinates = stellar_case
    path = tmp_path / "prepared.npz"
    save_spectral_grid(path, model.spectra)
    restored = SpecModel(load_spectral_grid(path), vmax=30.)
    assert restored.spectra.library == "bosz"
    values = {k: v for k, v in coordinates.items() if k != "vmic"}
    trace = trace_at(obs, restored, derived_config(config), values)
    assert trace["vmic"]["type"] == "deterministic"
    np.testing.assert_array_equal(trace["vmic"]["value"],
                                  empirical_vmic(values["teff"], values["logg"], values["mh"]))


@pytest.mark.parametrize("vmic", [1.2, dist.Uniform(.7, 1.7)])
def test_explicit_vmic_false_or_non_bosz_auto(stellar_case, vmic):
    obs, model, config, coordinates = stellar_case
    config = dict(config, atmosphere=dict(config["atmosphere"], vmic=vmic))
    custom = SpecModel(replace(model.spectra, library="custom", sources=("bosz-like.npz",)), vmax=30.)
    for active_model, mode in ((model, False), (custom, None)):
        trace = trace_at(obs, active_model, dict(config, use_empirical_vmic=mode), coordinates)
        assert trace["vmic"]["type"] == ("sample" if isinstance(vmic, dist.Distribution) else "deterministic")
        np.testing.assert_allclose(trace["vmic"]["value"], coordinates["vmic"])
    with pytest.raises(ValueError, match="exactly the named axes"):
        trace_at(obs, custom, derived_config(config), coordinates)


def test_explicit_empirical_vmic_on_compatible_non_bosz(stellar_case):
    obs, model, config, coordinates = stellar_case
    custom = SpecModel(replace(model.spectra, library="custom"), vmax=30.)
    values = {k: v for k, v in coordinates.items() if k != "vmic"}
    trace = trace_at(obs, custom, derived_config(config, use_empirical_vmic=True), values)
    assert trace["vmic"]["type"] == "deterministic"
    np.testing.assert_array_equal(trace["vmic"]["value"],
                                  empirical_vmic(values["teff"], values["logg"], values["mh"]))


@pytest.mark.parametrize("mode", [None, True])
def test_active_empirical_vmic_rejects_explicit_value(stellar_case, mode):
    obs, model, config, coordinates = stellar_case
    with pytest.raises(ValueError, match="omit atmosphere.*vmic"):
        trace_at(obs, model, dict(config, use_empirical_vmic=mode), coordinates)


@pytest.mark.parametrize("missing", ["teff", "logg", "mh", "vmic"])
def test_empirical_vmic_requires_compatible_axes(stellar_case, missing):
    obs, model, config, coordinates = stellar_case
    with pytest.raises(ValueError, match="requires teff, logg, mh and vmic"):
        trace_at(obs, without_axis(model, missing),
                 derived_config(config, use_empirical_vmic=True), coordinates)


@pytest.mark.parametrize("sigma", [None, .4])
def test_empirical_vmacro_prior_is_one_shared_stellar_latent(stellar_case, sigma):
    obs, model, config, coordinates = stellar_case
    broadening = {k: v for k, v in config["broadening"].items() if k != "vmacro"}
    config = dict(config, use_empirical_vmic=False, use_empirical_vmacro=True,
                  broadening=broadening)
    if sigma is not None:
        config["vmacro_empirical_sigma"] = sigma
    values = dict(coordinates, vmacro=4.2)
    trace = trace_at(obs, model, config, values)
    site = trace["vmacro"]
    expected = empirical_vmacro_valenti_fischer2005(coordinates["teff"], sigma=1. if sigma is None else sigma)
    assert site["type"] == "sample" and not site["is_observed"]
    assert jnp.shape(site["value"]) == ()
    assert site["fn"].batch_shape == site["fn"].event_shape == ()
    np.testing.assert_array_equal(site["fn"].base_dist.loc, expected.base_dist.loc)
    np.testing.assert_array_equal(site["fn"].base_dist.scale, expected.base_dist.scale)
    np.testing.assert_array_equal(site["fn"].low, 0.)
    params = handlers.substitute(model_single, data=values)(obs, model, **config)
    assert jnp.shape(params["components"][0]["broadening"]["vmacro"]) == ()
    assert obs.n_regions == 2
    np.testing.assert_array_equal(trace["model_flux"]["value"], model(params, obs.wavelength))


@pytest.mark.parametrize("sigma", [0., -1., np.nan, np.inf, [1., 1.]])
def test_empirical_vmacro_rejects_invalid_or_per_region_sigma(stellar_case, sigma):
    obs, model, config, coordinates = stellar_case
    broadening = {k: v for k, v in config["broadening"].items() if k != "vmacro"}
    config = dict(config, use_empirical_vmic=False, use_empirical_vmacro=True,
                  broadening=broadening, vmacro_empirical_sigma=sigma)
    with pytest.raises(ValueError, match="vmacro_empirical_sigma must be a finite positive scalar"):
        trace_at(obs, model, config, coordinates)


def test_empirical_vmacro_rejects_explicit_value(stellar_case):
    obs, model, config, coordinates = stellar_case
    with pytest.raises(ValueError, match="omit broadening.*vmacro"):
        trace_at(obs, model, dict(config, use_empirical_vmic=False, use_empirical_vmacro=True), coordinates)


@pytest.mark.parametrize("kind", ["vmic", "vmacro", "other_atmosphere", "other_broadening"])
def test_ordinary_modes_keep_strict_keys(stellar_case, kind):
    obs, model, config, coordinates = stellar_case
    config = dict(config, use_empirical_vmic=False)
    if kind in ("vmic", "other_atmosphere"):
        atmosphere = dict(config["atmosphere"])
        if kind == "vmic":
            atmosphere.pop("vmic")
        else:
            atmosphere["unknown"] = 0.
        config["atmosphere"] = atmosphere
        message = "exactly the named axes"
    else:
        broadening = dict(config["broadening"])
        if kind == "vmacro":
            broadening.pop("vmacro")
        else:
            broadening["unknown"] = 0.
        config["broadening"] = broadening
        message = "broadening must supply exactly"
    with pytest.raises(ValueError, match=message):
        trace_at(obs, model, config, coordinates)


@pytest.mark.parametrize("kind", ["missing_atmosphere", "extra_atmosphere", "missing_broadening", "extra_broadening"])
def test_derived_modes_relax_only_derived_keys(stellar_case, kind):
    obs, model, config, coordinates = stellar_case
    config = derived_config(config, use_empirical_vmacro=True)
    config["broadening"] = {k: v for k, v in config["broadening"].items() if k != "vmacro"}
    target = dict(config["atmosphere"] if "atmosphere" in kind else config["broadening"])
    if kind.startswith("missing"):
        target.pop("mh" if "atmosphere" in kind else "q1")
    else:
        target["unknown"] = 0.
    config["atmosphere" if "atmosphere" in kind else "broadening"] = target
    with pytest.raises(ValueError, match="must supply exactly"):
        trace_at(obs, model, config, coordinates)


@pytest.mark.parametrize("option,missing,message", [
    ("use_physical_logg_max", "teff", "requires teff and logg"),
    ("use_physical_logg_max", "logg", "requires teff and logg"),
    ("use_empirical_vmacro", "teff", "requires a teff"),
])
def test_teff_dependent_modes_require_axes(stellar_case, option, missing, message):
    obs, model, config, coordinates = stellar_case
    with pytest.raises(ValueError, match=message):
        trace_at(obs, without_axis(model, missing),
                 dict(config, use_empirical_vmic=False, **{option: True}), coordinates)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_combined_relations_joint_density_jit_and_gradients(stellar_case, dtype, x64_context):
    with x64_context(dtype == "float64"):
        obs, model, config, coordinates = stellar_case
        config = derived_config(config, use_physical_logg_max=True, use_empirical_vmacro=True)
        config["broadening"] = {k: v for k, v in config["broadening"].items() if k != "vmacro"}
        values = {k: jnp.asarray(v, dtype=dtype) for k, v in coordinates.items()
                  if k in ("teff", "logg", "mh")}
        values["vmacro"] = jnp.asarray(4.2, dtype=dtype)
        density = lambda p: log_density(model_single, (obs, model), config, p)[0]
        eager = density(values)
        value, gradient = jax.jit(jax.value_and_grad(density))(values)
        np.testing.assert_allclose(value, eager, atol=1e-4 if dtype == "float32" else 1e-10)
        assert np.isfinite(value)
        assert all(np.isfinite(g) for g in gradient.values())
        trace = trace_at(obs, model, config, values)
        assert {k for k, site in trace.items()
                if site["type"] == "sample" and not site["is_observed"]} == set(values)


def test_hmc_initialization_and_conditional_support_transform(stellar_case):
    obs, model, config, coordinates = stellar_case
    config = derived_config(config, use_physical_logg_max=True, use_empirical_vmacro=True)
    config["broadening"] = {k: v for k, v in config["broadening"].items() if k != "vmacro"}
    values = {k: coordinates[k] for k in ("teff", "logg", "mh")}
    values["vmacro"] = 4.2
    info = initialize_model(
        jax.random.PRNGKey(3), model_single, init_strategy=init_to_value(values=values),
        model_args=(obs, model), model_kwargs=config)
    unconstrained = info.param_info.z
    assert set(unconstrained) == set(values)
    initial = info.postprocess_fn(unconstrained)
    for name, value in values.items():
        np.testing.assert_allclose(initial[name], value, rtol=2e-6)

    # Moving only unconstrained Teff must also update logg's interval transform.
    shifted = dict(unconstrained, teff=unconstrained["teff"] + .4)
    physical = info.postprocess_fn(shifted)
    fraction = jax.nn.sigmoid(unconstrained["logg"])
    expected_logg = 3.8 + fraction * (physical_logg_max(physical["teff"]) - 3.8)
    np.testing.assert_allclose(physical["logg"], expected_logg, rtol=2e-6)
    assert not np.isclose(physical["logg"], initial["logg"])
    np.testing.assert_allclose(physical["vmic"], empirical_vmic(
        physical["teff"], physical["logg"], physical["mh"]), rtol=2e-6)
    energy, gradient = jax.jit(jax.value_and_grad(info.potential_fn))(shifted)
    assert np.isfinite(energy)
    assert all(np.isfinite(g) for g in gradient.values())
