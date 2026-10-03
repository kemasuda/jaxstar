"""Generic deterministic composition, independent component physics and AD."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.specfit import SpecModel, SpectralDecomposition
from test_specmodel import forward_tolerance, make_library, observations, parameters


@pytest.fixture(autouse=True)
def precision(x64_context, request):
    dtype = getattr(request.node, "callspec", None)
    dtype = np.float64 if dtype is None else dtype.params.get("dtype", np.float64)
    # Match the existing benchmark's precision configurations. The unchanged
    # Fourier kernel can promote float32 inputs when x64 is enabled globally.
    with x64_context(dtype == np.float64):
        yield


def sbn_parameters(count=3, dtype=np.float64):
    params = parameters()
    components = []
    for i in range(count):
        component = copy.deepcopy(params["components"][0])
        component["atmosphere"]["custom_temperature"] = .2 + .2 * i
        component["broadening"] = {"vsini": 5.3 + i, "vmacro": 2.7 + .3 * i,
                                   "u1": .5 - .03 * i, "u2": .2 + .02 * i}
        component["rv"] = -9.3 + 8.1 * i
        components.append(component)
    params.update(components=tuple(components), flux_weights=np.array([1 / (i + 1) for i in range(count)]), dilution=.13)
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), params)


@pytest.mark.parametrize("stage", ["intrinsic", "broadened", "full"])
@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stages_decomposition_and_jit(count, stage, dtype):
    library, params = make_library(dtype=dtype), sbn_parameters(count, dtype)
    model, wave = SpecModel(library), observations(library)
    parts = model.decompose(params, wave, stage=stage)
    assert isinstance(parts, SpectralDecomposition)
    assert parts.total.shape == wave.shape
    assert parts.components.shape == parts.weighted_components.shape == (count, *wave.shape)
    assert parts.flux_weights.shape == parts.stellar_fractions.shape == parts.light_fractions.shape == (count, 2)
    assert parts.dilution.shape == (2,)
    assert parts.total.dtype == parts.components.dtype == dtype
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    for i, component in enumerate(params["components"]):
        single = {"components": (component,), "instrument": params["instrument"]}
        expected = getattr(model, stage)(single, wave)
        np.testing.assert_array_equal(parts.components[i], expected)
    np.testing.assert_allclose(parts.stellar_fractions.sum(axis=0), 1, atol=tolerance)
    np.testing.assert_allclose(parts.light_fractions.sum(axis=0) + parts.dilution, 1, atol=tolerance)
    np.testing.assert_array_equal(parts.total, parts.dilution[:, None] + parts.weighted_components.sum(axis=0))
    evaluate = lambda m, p, w: getattr(m, stage)(p, w)
    np.testing.assert_array_equal(evaluate(model, params, wave), parts.total)
    physical_tolerance = forward_tolerance(dtype, stage)
    np.testing.assert_allclose(jax.jit(evaluate)(model, params, wave), parts.total, **physical_tolerance)
    compiled = jax.jit(lambda m, p, w: m.decompose(p, w, stage=stage))(model, params, wave)
    for name in parts._fields:
        budget = physical_tolerance if name in ("total", "components", "weighted_components") else {"rtol": tolerance, "atol": tolerance}
        np.testing.assert_allclose(getattr(compiled, name), getattr(parts, name), **budget, err_msg=name)
    if stage == "full":
        np.testing.assert_array_equal(model(params, wave), model.full(params, wave))


@pytest.mark.parametrize("dilution", [0., .2])
def test_sb2_hand_calculation(dilution):
    library, params = make_library(), sbn_parameters(2)
    params.update(flux_weights=(1., .25), dilution=dilution)
    parts = SpecModel(library).decompose(params, observations(library))
    f1, f2 = parts.components
    expected = dilution + (1 - dilution) * (f1 + .25 * f2) / 1.25
    np.testing.assert_allclose(parts.total, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dilution", [0., .27])
def test_single_star_defaults_and_weight_scale(dilution):
    library, params = make_library(), parameters()
    model, wave = SpecModel(library), observations(library)
    original = model(params, wave)
    np.testing.assert_array_equal(model.decompose(params, wave).components[0], original)
    params["dilution"] = dilution
    for weight in (None, 1., 7.3):
        if weight is not None:
            params["flux_weights"] = (weight,)
        np.testing.assert_allclose(model(params, wave), dilution + (1 - dilution) * original, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(model.decompose(params, wave).stellar_fractions, np.ones((1, 2)))


def test_scale_invariance():
    library, params = make_library(), sbn_parameters()
    model, wave = SpecModel(library), observations(library)
    params["flux_weights"] = np.array([1., .3, .1])
    original = model.decompose(params, wave)
    params["flux_weights"] = np.array([10., 3., 1.])
    scaled = model.decompose(params, wave)
    for name in ("stellar_fractions", "light_fractions", "weighted_components", "total"):
        np.testing.assert_allclose(getattr(scaled, name), getattr(original, name), rtol=1e-14, atol=1e-14)


def test_per_region_weights_dilution_and_mixed_entries():
    library, params = make_library(), sbn_parameters(2)
    model, wave = SpecModel(library), observations(library)
    params.update(flux_weights=(1., np.array([.25, .75])), dilution=np.array([.1, .3]))
    mixed = model.decompose(params, wave)
    params["flux_weights"] = np.array([[1., 1.], [.25, .75]])
    np.testing.assert_array_equal(model(params, wave), mixed.total)
    np.testing.assert_allclose(mixed.total, params["dilution"][:, None] + (1 - params["dilution"][:, None]) * (
        mixed.components[0] + params["flux_weights"][1, :, None] * mixed.components[1]) /
        params["flux_weights"].sum(axis=0)[:, None])
    params["flux_weights"] = np.array([[1., 1.], [.25, .25]])
    shared = model(params, wave)
    np.testing.assert_array_equal(shared[0], mixed.total[0])
    assert not np.array_equal(shared[1], mixed.total[1])
    params["dilution"] = np.array([.1, .4])
    changed = model(params, wave)
    np.testing.assert_array_equal(changed[0], shared[0])
    assert not np.array_equal(changed[1], shared[1])


def test_scalar_equals_constant_region_values_and_zero_weights():
    library, params = make_library(), sbn_parameters()
    model, wave = SpecModel(library), observations(library)
    original = model(params, wave)
    params["flux_weights"] = jnp.broadcast_to(params["flux_weights"][:, None], (3, 2))
    params["dilution"] = jnp.broadcast_to(params["dilution"], (2,))
    np.testing.assert_array_equal(model(params, wave), original)
    params.update(flux_weights=np.array([[0., 1.], [1., 0.], [0., 0.]]), dilution=0.)
    parts = model.decompose(params, wave)
    np.testing.assert_array_equal(parts.total[0], parts.components[1, 0])
    np.testing.assert_array_equal(parts.total[1], parts.components[0, 1])


@pytest.mark.parametrize("bad", [[-1., .3], [np.nan, .3], [1., np.inf], [0., 0.], [[1., 0.], [.3, 0.]]])
def test_invalid_weight_domain(bad):
    library, params = make_library(), sbn_parameters(2)
    params["flux_weights"] = np.asarray(bad)
    with pytest.raises(ValueError, match="flux_weights must be finite nonnegative"):
        SpecModel(library)(params, observations(library))


@pytest.mark.parametrize("bad", [-.1, 1., 1.1, np.nan, np.inf, np.array([0., -.1])])
def test_invalid_dilution_domain(bad):
    library, params = make_library(), sbn_parameters(2)
    params["dilution"] = bad
    with pytest.raises(ValueError, match="0 <= dilution < 1"):
        SpecModel(library)(params, observations(library))


@pytest.mark.parametrize("key,bad", [
    ("flux_weights", .5), ("flux_weights", np.ones(3)), ("flux_weights", np.ones((2, 1))),
    ("flux_weights", np.ones((2, 2, 173))), ("flux_weights", (1.,)),
    ("flux_weights", (1., np.ones((2, 1)))), ("dilution", np.ones(3)), ("dilution", np.ones((2, 1)))])
def test_invalid_composition_shapes(key, bad):
    library, params = make_library(), sbn_parameters(2)
    params[key] = bad
    with pytest.raises(ValueError, match="flux_weights|dilution"):
        SpecModel(library)(params, observations(library))


def test_multicomponent_requires_weights_and_valid_components():
    library, params = make_library(), sbn_parameters(2)
    model, wave = SpecModel(library), observations(library)
    del params["flux_weights"]
    with pytest.raises(ValueError, match="explicit flux_weights"):
        model(params, wave)
    params["flux_weights"] = (1., .5)
    params["components"] = (params["components"][0], {})
    with pytest.raises(ValueError, match="atmosphere dict"):
        model(params, wave)


def test_every_component_retains_atmosphere_shape_and_coverage_checks():
    library, params = make_library(), sbn_parameters(2)
    model, wave = SpecModel(library), observations(library)
    params["components"][1]["atmosphere"]["custom_temperature"] = np.ones(2)
    with pytest.raises(ValueError, match="scalar per component"):
        model(params, wave)
    params["components"][1]["atmosphere"]["custom_temperature"] = .4
    params["components"][1]["rv"] = 400.
    with pytest.raises(ValueError, match="insufficient model wavelength coverage"):
        model(params, wave)


def test_single_region_decomposition_retains_region_axis():
    library, params = make_library(regions=1), sbn_parameters(2)
    model, wave = SpecModel(library), observations(library)[0]
    parts = model.decompose(params, wave)
    assert parts.components.shape == (2, 1, len(wave))
    assert parts.total.shape == (1, len(wave))
    np.testing.assert_array_equal(model(params, wave), parts.total[0])
    with pytest.raises(ValueError, match="stage must be"):
        model.decompose(params, wave, stage="invalid")


@pytest.mark.parametrize("count", [1, 2, 3])
def test_autodiff_all_parameters_weight_dilution_finite_difference_and_scale_null(count):
    library, params = make_library(), sbn_parameters(count)
    model, wave = SpecModel(library), observations(library)
    wavelength_weight = jnp.linspace(.3, 1.7, wave.size).reshape(wave.shape)
    objective = lambda m, p, w: jnp.mean(m(p, w)**2 * wavelength_weight)
    value, gradient = jax.jit(jax.value_and_grad(objective, argnums=1))(model, params, wave)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
    for component in gradient["components"]:
        assert set(component["atmosphere"]) == {"custom_temperature"}
        assert set(component["broadening"]) == {"vsini", "vmacro", "u1", "u2"}
        assert "rv" in component
    np.testing.assert_allclose(jnp.vdot(gradient["flux_weights"], params["flux_weights"]), 0, atol=1e-14)
    evaluate = jax.jit(objective)
    step = 1e-5
    for key in ("flux_weights", "dilution"):
        for index in range(count) if key == "flux_weights" else (None,):
            plus, minus = copy.deepcopy(params), copy.deepcopy(params)
            if index is None:
                plus[key] += step
                minus[key] -= step
                actual = gradient[key]
            else:
                plus[key] = plus[key].at[index].add(step)
                minus[key] = minus[key].at[index].add(-step)
                actual = gradient[key][index]
            expected = (evaluate(model, plus, wave) - evaluate(model, minus, wave)) / (2 * step)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-10)


@pytest.mark.parametrize("key,value,message", [
    ("flux_weights", jnp.zeros(2), "flux_weights must be finite nonnegative"),
    ("dilution", jnp.array(1.), "0 <= dilution < 1")])
def test_compiled_invalid_composition_fails(key, value, message):
    library, params = make_library(), sbn_parameters(2)
    model, wave = SpecModel(library), observations(library)
    evaluate = jax.jit(lambda m, p, w: m(p, w))
    jax.block_until_ready(evaluate(model, params, wave))
    params[key] = value
    with pytest.raises(Exception, match=message):
        jax.block_until_ready(evaluate(model, params, wave))


def test_arbitrary_component_count_and_independent_region_controls():
    library, params = make_library(), sbn_parameters(9)
    for i, component in enumerate(params["components"]):
        component["atmosphere"]["custom_temperature"] = .1 + .08 * i
    model, wave = SpecModel(library), observations(library)
    assert model.decompose(params, wave, stage="intrinsic").components.shape == (9, *wave.shape)
    params = sbn_parameters(3)
    original = model.decompose(params, wave)
    params["components"][1]["rv"] = jnp.array([-1.2, 12.7])
    modified = model.decompose(params, wave)
    np.testing.assert_array_equal(original.components[jnp.array([0, 2])], modified.components[jnp.array([0, 2])])
    # Scalar RV and a typed per-region array may differ at float64 roundoff.
    np.testing.assert_allclose(original.components[1, 0], modified.components[1, 0], rtol=0, atol=5e-15)
    assert not np.array_equal(original.components[1, 1], modified.components[1, 1])


def test_per_region_composition_gradients():
    library, params = make_library(), sbn_parameters(3)
    params.update(flux_weights=jnp.array([[1., 2.], [.3, .7], [.1, .2]]), dilution=jnp.array([.1, .2]))
    model, wave = SpecModel(library), observations(library)
    gradient = jax.jit(jax.grad(lambda m, p, w: jnp.mean(m(p, w)**2), argnums=1))(model, params, wave)
    assert gradient["flux_weights"].shape == (3, 2)
    assert gradient["dilution"].shape == (2,)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
    np.testing.assert_allclose(jnp.sum(gradient["flux_weights"] * params["flux_weights"], axis=0), 0, atol=1e-14)
