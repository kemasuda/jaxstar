"""Flat parameter construction agrees with the existing physical API."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from jaxstar.specfit import SpecModel, model_single, single_star_params
from test_numpyro_spectral import _case
from test_specmodel import forward_tolerance, make_library, observations, parameters
from test_specmodel_sbn import sbn_parameters


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def flat_component(component):
    return {**component["atmosphere"], **component["broadening"], "rv": component["rv"]}


def flat_point():
    nested = parameters()
    return {**flat_component(nested["components"][0]), **nested["instrument"]}


def assert_tree_equal(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_star_stages_width_and_jit(dtype, x64_context):
    with x64_context(dtype == np.float64):
        library = make_library(dtype=dtype)
        model, wave = SpecModel(library), observations(library)
        point = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), flat_point())
        expected = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), parameters())
        actual = model.make_params(point)
        assert_tree_equal(actual, expected)
        assert_tree_equal(jax.jit(lambda m, p: m.make_params(p))(model, point), expected)
        for stage in ("intrinsic", "broadened", "full"):
            reference = getattr(model, stage)(expected, wave)
            np.testing.assert_array_equal(getattr(model, stage)(actual, wave), reference)
            compiled = jax.jit(lambda m, p, w: getattr(m, stage)(m.make_params(p), w))
            np.testing.assert_allclose(compiled(model, point, wave), reference,
                                       **forward_tolerance(dtype, stage))
        np.testing.assert_array_equal(model.broadening_fwhm(actual), model.broadening_fwhm(expected))
        # This first step adds an explicit converter, not implicit flat inputs.
        with pytest.raises(ValueError, match="components"):
            model(point, wave)


@pytest.mark.parametrize("fixed_ip", [False, True])
def test_complete_builtin_point_matches_model_return_without_replay(fixed_ip, monkeypatch):
    obs, model, settings, _, _ = _case()
    if fixed_ip:
        model = SpecModel(model.spectra, vmax=30., fixed_ip=dict(
            mu1_kms=-1., mu2_kms=2., sigma1_kms=1.5, sigma2_kms=2., fraction2=.3))
        settings.pop("resolving_power")
    with handlers.seed(rng_seed=7), handlers.trace() as trace:
        expected = model_single(obs, model, **settings, prior_only=True,
                                likelihood="gaussian_mixture", use_gp=False,
                                mismatch_fraction=.04, mismatch_scale=.1)
    point = {name: site["value"] for name, site in trace.items()
             if site["type"] in ("sample", "deterministic") and not site.get("is_observed", False)}
    assert {"q1", "q2", "u1", "u2", "continuum_coefficients", "jitter"} <= point.keys()
    if not fixed_ip:
        assert "resolving_power" in point
    original = copy.deepcopy(point)

    def forbidden(*args, **kwargs):
        raise AssertionError("parameter construction must not evaluate a spectrum")

    monkeypatch.setattr(SpecModel, "_atmosphere", forbidden)
    monkeypatch.setattr(SpecModel, "_broaden", forbidden)
    actual = model.make_params(point)
    assert_tree_equal(actual, expected)
    assert_tree_equal(point, original)
    # Unused, explicitly allowed inference sites need no numerical processing.
    point.update(gp_amplitude=np.nan, gp_scale=None, model_flux=object())
    assert_tree_equal(model.make_params(point), expected)


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("fixed_ip", [False, True])
def test_sbn_region_parameters_and_light_accounting(count, fixed_ip):
    library = make_library()
    ip = dict(mu1_kms=[-1., -.5], mu2_kms=2., sigma1_kms=1.5,
              sigma2_kms=[2., 3.], fraction2=.3)
    model = SpecModel(library, fixed_ip=ip if fixed_ip else None)
    expected = sbn_parameters(count)
    expected["components"][0]["rv"] = jnp.array([-8., 9.])
    expected["components"][1]["broadening"]["vsini"] = jnp.array([5., 7.])
    expected["flux_weights"] = np.arange(1., 2 * count + 1).reshape(count, 2)
    expected["dilution"] = np.array([.1, .2])
    expected["instrument"] = {} if fixed_ip else {"resolving_power": np.array([60000., 80000.])}
    point = dict(components=tuple(flat_component(c) for c in expected["components"]),
                 flux_weights=expected["flux_weights"], dilution=expected["dilution"],
                 **expected["instrument"])
    actual = model.make_params(point)
    assert_tree_equal(actual, expected)
    wave = observations(library)
    parts = model.decompose(actual, wave)
    reference = model.decompose(expected, wave)
    assert_tree_equal(parts, reference)
    np.testing.assert_array_equal(model(actual, wave), reference.total)
    np.testing.assert_array_equal(model.broadening_fwhm(actual), model.broadening_fwhm(expected))
    compiled = jax.jit(lambda m, p, w: m.decompose(m.make_params(p), w))(model, point, wave)
    for a, b in zip(compiled, reference):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=2e-13)
    # Existing scalar/per-region mixed weight convention also survives conversion.
    point["flux_weights"] = (1.,) + tuple(expected["flux_weights"][i] for i in range(1, count))
    expected["flux_weights"] = point["flux_weights"]
    np.testing.assert_array_equal(model(model.make_params(point), wave), model(expected, wave))


def test_partial_inputs_delegate_required_values_and_ranges_to_physical_stage():
    model = SpecModel(make_library())
    wave = observations(model.spectra)
    atmosphere = model.make_params({"custom_temperature": .4})
    np.testing.assert_array_equal(model.intrinsic(atmosphere, wave), model.intrinsic(parameters(), wave))
    with pytest.raises(ValueError, match="broadening"):
        model(atmosphere, wave)
    broadening = flat_point()
    broadening.pop("custom_temperature")
    broadening.pop("rv")
    partial = model.make_params(broadening)
    np.testing.assert_array_equal(model.broadening_fwhm(partial), model.broadening_fwhm(parameters()))
    with pytest.raises(ValueError, match="atmosphere"):
        model(partial, wave)
    # No guessed RV, atmosphere, instrument or light weights.
    assert set(partial["components"][0]) == {"broadening"}
    no_r = model.make_params({k: v for k, v in broadening.items() if k != "resolving_power"})
    with pytest.raises(ValueError, match="instrument.resolving_power"):
        model.broadening_fwhm(no_r)
    star = {k: v for k, v in flat_point().items() if k != "resolving_power"}
    stars = model.make_params({"components": (star, star), "resolving_power": 70000.})
    assert model.broadening_fwhm(stars).shape == (2, 2)
    with pytest.raises(ValueError, match="explicit flux_weights"):
        model(stars, wave)
    # Parameters unused by the requested stage are not physically validated.
    invalid = model.make_params({**flat_point(), "custom_temperature": np.nan, "rv": np.nan,
                                 "dilution": -1.})
    np.testing.assert_array_equal(model.broadening_fwhm(invalid), model.broadening_fwhm(parameters()))


def test_limb_darkening_pairs_and_gradients():
    model = SpecModel(make_library())
    wave = observations(model.spectra)
    point = {k: v for k, v in flat_point().items() if k not in ("u1", "u2")}
    point.update(q1=.36, q2=np.array([.3, .4]))

    def existing(p):
        return single_star_params({"custom_temperature": p["custom_temperature"]},
                                  **{k: v for k, v in p.items() if k != "custom_temperature"})

    assert_tree_equal(model.make_params(point), existing(point))
    complete = {**point, **existing(point)["components"][0]["broadening"]}
    assert_tree_equal(model.make_params(complete), existing(point))
    for constructor in (model.make_params, existing):
        value, gradient = jax.jit(jax.value_and_grad(lambda p: model(constructor(p), wave).sum()))(point)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(v)) for v in jax.tree.leaves(gradient))
        if constructor == model.make_params:
            reference_value, reference_gradient = value, gradient
        else:
            np.testing.assert_allclose(value, reference_value, rtol=1e-12)
            for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(reference_gradient)):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    bad = {**complete, "u1": np.array([.5, .5])}
    with pytest.raises(ValueError, match="inconsistent"):
        model.make_params(bad)
    with pytest.raises(Exception, match="inconsistent"):
        jax.jit(lambda p: model.make_params(p)["components"][0]["broadening"]["u1"])(bad).block_until_ready()


@pytest.mark.parametrize("point,match", [
    ({"vsnii": 3.}, "unexpected keys"),
    ({"components": []}, "nonempty"),
    ({"components": {}}, "nonempty"),
    ({"components": (None,)}, "mapping"),
    ({"components": ({},), "rv": 3.}, "outer parameters"),
    ({"components": ({"resolving_power": 70000.},)}, "flat component"),
    ({"components": ({"flux_weights": [1.]},)}, "flat component"),
    ({"components": ({"atmosphere": {}},)}, "flat component"),
    ({"instrument": {}}, "unexpected keys"),
    ({"q1": .3}, "supply both q1 and q2"),
    ({"u2": .3}, "supply both u1 and u2"),
    ({"q1": -.1, "q2": .3}, r"finite in \[0, 1\]"),
    ({"q1": .3, "q2": np.nan}, r"finite in \[0, 1\]"),
    ({"custom_temperature": [.4, .5]}, "custom_temperature must have shape"),
    ({"rv": np.ones((2, 1))}, "rv must have shape"),
    ({"rv": [1., 2., 3.]}, "rv must have shape"),
    ({"rv": "3"}, "real numeric"),
    ({"rv": 1j}, "real numeric"),
    ({"rv": True}, "real numeric"),
    ({"resolving_power": None}, "real numeric"),
    ({"flux_weights": (1., 2.)}, "one entry per component"),
    ({"flux_weights": np.ones((2, 2))}, "flux_weights must have shape"),
    ({"flux_weights": (np.ones(3),)}, "flux_weights entry must have shape"),
    ({"dilution": np.ones((2, 1))}, "dilution must have shape"),
])
def test_invalid_structure_names_and_shapes(point, match):
    with pytest.raises((TypeError, ValueError), match=match):
        SpecModel(make_library()).make_params(point)


def test_fixed_ip_conflict_and_nonmapping_input():
    model = SpecModel(make_library(), fixed_ip=dict(
        mu1_kms=0., mu2_kms=1., sigma1_kms=2., sigma2_kms=3., fraction2=.3))
    with pytest.raises(ValueError, match="omit resolving_power"):
        model.make_params(flat_point())
    with pytest.raises(TypeError, match="mapping"):
        model.make_params([flat_point()])
