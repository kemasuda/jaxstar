"""Deterministic stage, shape, physical response, coverage and JAX contracts."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import SpecModel
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit.model import _doppler_factor
from jaxstar.specfit.sampling import _C_KMS


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def make_library(dtype=np.float64, regions=2, log=True, pixels=1025):
    velocity = np.linspace(-250, 250, pixels)
    centers = np.array([5000., 6000.])[:regions]
    wave = centers[:, None] * np.exp(velocity / _C_KMS)
    if not log:
        wave = np.stack([np.linspace(row[0], row[-1], pixels) for row in wave])
    profile = np.exp(-0.5 * (velocity / 3)**2)
    flux = np.stack([np.broadcast_to(1 - depth * profile, wave.shape) for depth in (.3, .6)])
    grid = RectilinearGrid(axes={"custom_temperature": np.array([0., 1.], dtype=dtype)},
        fields={"flux": Field(flux.astype(dtype), ("custom_temperature",), payload_dims=("region", "pixel"))})
    return _SpectralLibrary(grid, wave.astype(dtype), tuple(range(regions)), "synthetic", "vacuum", "normalized")


def parameters():
    return {"components": ({"atmosphere": {"custom_temperature": .4},
                            "broadening": {"vsini": 6.3, "vmacro": 3.1, "u1": .5, "u2": .2},
                            "rv": .43},), "instrument": {"resolving_power": 70000.}}


def observations(library):
    return np.asarray(library.wavelength)[:, 340:685:2]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stages_alias_jit_gradients_and_dynamic_arrays(dtype):
    spectra = make_library(dtype)
    model, params, wave = SpecModel(spectra), parameters(), observations(spectra)
    for stage in ("intrinsic", "broadened", "full"):
        function = lambda m, p, w: getattr(m, stage)(p, w)
        eager = function(model, params, wave)
        compiled = jax.jit(function)(model, params, wave)
        assert compiled.shape == wave.shape
        np.testing.assert_allclose(compiled, eager, rtol=2e-6, atol=2e-7)
        value, gradient = jax.jit(jax.value_and_grad(lambda p: function(model, p, wave).sum()))(params)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(gradient))
    np.testing.assert_array_equal(model(params, wave), model.full(params, wave))
    compiled = jax.jit(lambda m, p, w: m(p, w))
    flatter = SpecModel(make_library(dtype))
    # Replacement model data are ordinary dynamic leaves, not static self.
    leaves, treedef = jax.tree_util.tree_flatten(flatter)
    changed = [x * .99 if x.shape == spectra.grid.field("flux").values.shape else x for x in leaves]
    replaced = jax.tree_util.tree_unflatten(treedef, changed)
    np.testing.assert_allclose(compiled(replaced, params, wave), .99 * compiled(model, params, wave), rtol=2e-6)
    program = jax.make_jaxpr(lambda m, p, w: m(p, w))(model, params, wave)
    assert all(np.size(x) < spectra.grid.field("flux").values.size for x in program.consts)


def test_negligible_broadening_zero_rv_and_stage_specific_inputs():
    library = make_library()
    model, params, wave = SpecModel(library), parameters(), observations(library)
    params["components"][0]["broadening"].update(vsini=0., vmacro=0.)
    params["instrument"]["resolving_power"] = np.inf
    params["components"][0]["rv"] = 0.
    intrinsic = model.intrinsic(params, wave)
    np.testing.assert_allclose(model.broadened(params, wave), intrinsic, rtol=0, atol=2e-13)
    np.testing.assert_array_equal(model.full(params, wave), model.broadened(params, wave))
    minimal = {"components": ({"atmosphere": params["components"][0]["atmosphere"]},)}
    np.testing.assert_array_equal(model.intrinsic(minimal, wave), intrinsic)
    del params["components"][0]["rv"]
    np.testing.assert_allclose(model.broadened(params, wave), intrinsic, atol=2e-13)


def test_single_region_1d_convenience_and_dynamic_wavelength():
    library = make_library(regions=1)
    model, params, wave = SpecModel(library), parameters(), observations(library)
    fn = jax.jit(lambda m, p, w: m(p, w))
    for stage in ("intrinsic", "broadened", "full"):
        np.testing.assert_array_equal(getattr(model, stage)(params, wave[0]), getattr(model, stage)(params, wave)[0])
    np.testing.assert_allclose(fn(model, params, wave[0]), model.full(params, wave)[0])
    shifted_wave = wave * 1.000001
    assert not np.array_equal(fn(model, params, shifted_wave), fn(model, params, wave))


def test_rv_feature_translation_and_relativistic_convention():
    library = make_library()
    model, params, wave = SpecModel(library), parameters(), observations(library)
    for velocity in (-12., 12.):
        params["components"][0]["rv"] = velocity
        factor = _doppler_factor(velocity)
        np.testing.assert_allclose(factor, np.sqrt((1+velocity/_C_KMS)/(1-velocity/_C_KMS)), rtol=0, atol=0)
        # Evaluating at shifted wavelengths preserves the entire line profile.
        np.testing.assert_allclose(model.full(params, wave * factor), model.broadened(params, wave), rtol=0, atol=2e-12)
        line_index = np.argmin(model.full(params, wave), axis=1)
        centers = np.take_along_axis(wave, line_index[:, None], axis=1)[:, 0]
        expected = np.array([5000., 6000.]) * factor
        np.testing.assert_allclose(centers, expected, rtol=4e-6)


@pytest.mark.parametrize("control,low,high", [("vsini", 1., 15.), ("vmacro", 1., 10.), ("resolving_power", 100000., 20000.)])
def test_broadening_increases_line_width(control, low, high):
    library = make_library()
    model, params, wave = SpecModel(library), parameters(), observations(library)
    widths = []
    for value in (low, high):
        target = params["instrument"] if control == "resolving_power" else params["components"][0]["broadening"]
        target[control] = value
        deficit = 1 - np.asarray(model.broadened(params, wave))[0]
        velocity = _C_KMS * np.log(wave[0] / 5000.)
        widths.append(np.sqrt(np.sum(deficit * velocity**2) / np.sum(deficit)))
    assert widths[1] > widths[0] * 1.1


@pytest.mark.parametrize("control", ["vsini", "vmacro", "u1", "u2", "rv", "resolving_power"])
def test_scalar_array_equivalence_and_region_isolation(control):
    library = make_library()
    model, params, wave = SpecModel(library), parameters(), observations(library)
    shared = model(params, wave)
    target = (params["instrument"] if control == "resolving_power" else params["components"][0]
              if control == "rv" else params["components"][0]["broadening"])
    initial = target[control]
    target[control] = np.full(2, initial)
    np.testing.assert_array_equal(model(params, wave), shared)
    target[control][1] = initial * 1.3
    modified = model(params, wave)
    np.testing.assert_array_equal(modified[0], shared[0])
    assert not np.array_equal(modified[1], shared[1])


@pytest.mark.parametrize("control", ["vsini", "vmacro", "u1", "u2", "rv", "resolving_power"])
def test_invalid_post_parameter_shapes(control):
    library, params = make_library(), parameters()
    target = (params["instrument"] if control == "resolving_power" else params["components"][0]
              if control == "rv" else params["components"][0]["broadening"])
    target[control] = np.ones((2, 1))
    with pytest.raises(ValueError, match="scalar or shape"):
        SpecModel(library)(params, observations(library))


@pytest.mark.parametrize("components", [(), ({}, {}), {"atmosphere": {}}])
def test_component_count_contract(components):
    library = make_library()
    with pytest.raises(ValueError, match="exactly one component"):
        SpecModel(library).intrinsic({"components": components}, observations(library))


@pytest.mark.parametrize("coordinate", [np.ones(2), np.ones(1)])
def test_atmosphere_region_queries_rejected(coordinate):
    library, params = make_library(), parameters()
    params["components"][0]["atmosphere"]["custom_temperature"] = coordinate
    with pytest.raises(ValueError, match="scalar per component"):
        SpecModel(library)(params, observations(library))


@pytest.mark.parametrize("stage", ["intrinsic", "broadened", "full"])
@pytest.mark.parametrize("compiled", [False, True])
def test_coverage_failures_are_explicit(stage, compiled):
    library, params = make_library(), parameters()
    model = SpecModel(library)
    wave = np.asarray(library.wavelength) if stage != "intrinsic" else np.asarray(library.wavelength) * 1.01
    function = lambda m, p, w: getattr(m, stage)(p, w)
    if compiled:
        function = jax.jit(function)
    with pytest.raises(Exception, match="insufficient model wavelength coverage"):
        function(model, params, wave).block_until_ready()


def test_dynamic_rv_coverage_failure_and_invalid_setup():
    library, params = make_library(), parameters()
    params["components"][0]["rv"] = 500.
    with pytest.raises(Exception, match="insufficient model wavelength coverage"):
        jax.jit(lambda m, p, w: m(p, w))(SpecModel(library), params, observations(library)).block_until_ready()
    with pytest.raises(ValueError, match="log-uniform fitting"):
        SpecModel(make_library(log=False))
    with pytest.raises(ValueError, match="too short"):
        SpecModel(library, vmax=400.)
    with pytest.raises(ValueError, match="too coarse"):
        SpecModel(library, vmax=.01)


def test_custom_operator_boundary_without_an_analytic_kernel():
    library, params = make_library(), parameters()
    def identity(wavelength, flux, **unused):
        return wavelength[:, 2:-2], flux[:, 2:-2]
    model = SpecModel(library, broadening_operator=identity)
    wave = observations(library)
    np.testing.assert_array_equal(model.broadened(params, wave), model.intrinsic(params, wave))
    np.testing.assert_allclose(jax.jit(lambda m, p, w: m.broadened(p, w))(model, params, wave),
                               model.intrinsic(params, wave), rtol=0, atol=3e-16)


def test_forward_gradients_against_finite_differences():
    library, params = make_library(), parameters()
    model, wave = SpecModel(library), observations(library)
    weights = jnp.linspace(.2, 1.3, wave.size).reshape(wave.shape)
    objective = lambda p: jnp.sum(model(p, wave) * weights)
    gradient = jax.jit(jax.grad(objective))(params)
    for path, step in [(('components', 0, 'atmosphere', 'custom_temperature'), 1e-4),
                       (('components', 0, 'broadening', 'vsini'), 1e-3),
                       (('components', 0, 'rv'), 1e-3),
                       (('instrument', 'resolving_power'), 10.)]:
        differences = []
        for sign in (-1, 1):
            altered = copy.deepcopy(params)
            target = altered
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] += sign * step
            differences.append(objective(altered))
        expected = (differences[1] - differences[0]) / (2 * step)
        actual = gradient
        for key in path:
            actual = actual[key]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-9)


def test_x64_disabled_forward_and_reverse_mode(x64_context):
    with x64_context(False):
        library = make_library(np.float32)
        model = SpecModel(library)
        params = jax.tree_util.tree_map(jnp.asarray, parameters())
        wave = observations(library)
        function = jax.jit(lambda m, p, w: m(p, w))
        result = function(model, params, wave)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, model(params, wave), rtol=2e-6, atol=2e-7)
        value, gradient = jax.jit(jax.value_and_grad(lambda m, p, w: m(p, w).mean(), argnums=1))(model, params, wave)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(gradient))


def test_outer_parameter_vmap_keeps_scalar_atmosphere_queries():
    library, params = make_library(), parameters()
    model, wave = SpecModel(library), observations(library)
    batch = jax.tree_util.tree_map(lambda value: jnp.stack([jnp.asarray(value)] * 2), params)
    batch["components"][0]["atmosphere"]["custom_temperature"] = jnp.array([.4, .6])
    output = jax.jit(jax.vmap(lambda p: model(p, wave)))(batch)
    assert output.shape == (2, *wave.shape)
    np.testing.assert_allclose(output[0], model(params, wave), rtol=0, atol=5e-15)


@pytest.mark.parametrize("path,value,message", [
    (("components", 0, "atmosphere", "custom_temperature"), 2., "inside the prepared grid"),
    (("components", 0, "broadening", "vsini"), -1., "nonnegative speeds"),
    (("components", 0, "broadening", "vmacro"), np.nan, "finite nonnegative"),
    (("components", 0, "broadening", "u1"), 4., "positive disk intensity"),
    (("instrument", "resolving_power"), 0., "positive resolving_power"),
    (("components", 0, "rv"), 299792.458, "abs.rv. < c"),
    (("components", 0, "broadening", "vsini"), 100., "increase SpecModel vmax"),
])
def test_invalid_physical_inputs_fail(path, value, message):
    library, params = make_library(), parameters()
    target = params
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        SpecModel(library)(params, observations(library))


def test_requested_wavelength_shape_and_values_fail_clearly():
    library = make_library()
    model, params, wave = SpecModel(library), parameters(), observations(library)
    with pytest.raises(ValueError, match="shape .n_region, n_pixel."):
        model(params, wave[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        model(params, wave[:, ::-1])
