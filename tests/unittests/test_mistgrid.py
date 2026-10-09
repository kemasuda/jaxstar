import importlib
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxstar.grid import GridResult
from jaxstar.mistfit import MistFit, MistModel
from _mistgrid_reference import MistGridIso
from jaxstar.mistfit.mistgrid.paths import (
    MISTGRID_CACHE_VERSION,
    MISTGRID_ENV_VAR,
)


isochrone_module = importlib.import_module("jaxstar.mistfit.mistgrid.isochrone")
paths_module = importlib.import_module("jaxstar.mistfit.mistgrid.paths")


@pytest.fixture
def tiny_grid_path(tmp_path):
    """Create a 2x2x2 grid whose values follow an obvious linear rule."""
    logage = np.array([8.0, 9.0], dtype=np.float32)
    feh = np.array([-0.5, 0.5], dtype=np.float32)
    eep = np.array([100.0, 200.0], dtype=np.float32)

    age_values, feh_values, eep_values = np.meshgrid(
        logage, feh, eep, indexing="ij"
    )
    mass = age_values + 2.0 * feh_values + eep_values / 100.0
    teff = 5000.0 + 100.0 * mass

    path = tmp_path / "tiny_mistgrid.npz"
    np.savez(
        path,
        logagrid=logage,
        fgrid=feh,
        eepgrid=eep,
        mass=mass,
        teff=teff,
    )
    return path


def raw_values(model, age, feh, eep, *, keys=None):
    return model.grid.interpolate(
        {"age": age, "feh": feh, "eep": eep}, keys=keys,
        legacy_interpolation=model.legacy_interpolation,
    )


def raw_values_given_mass(model, logage, feh_init, mass_init, *, keys=None):
    eep = model.eep_given_mass(logage, feh_init, mass_init)
    return raw_values(model, logage, feh_init, eep, keys=keys)


def expected_mass(age, feh, eep):
    """The physical-coordinate rule used to construct ``tiny_grid_path``."""
    return age + 2.0 * feh + eep / 100.0


# Frozen legacy characterization, including its known defects. Production
# model/loader coverage follows below; this baseline never downloads grids.


def test_values_interpolate_selected_fields_in_requested_order(tiny_grid_path):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["teff", "mass"])

    values = grid.values(age=8.5, feh=0.0, eep=150.0)
    assert isinstance(values, list)
    teff, mass = values

    np.testing.assert_allclose([teff, mass], [6000.0, 10.0])
    assert mass.dtype == jnp.float32


def test_set_keys_changes_fields_after_the_first_evaluation(tiny_grid_path):
    """Expose the known interaction between mutable keys and the JIT cache."""
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass"])
    grid.values(age=8.5, feh=0.0, eep=150.0)[0].block_until_ready()

    grid.set_keys(["teff"])
    (actual,) = grid.values(age=8.5, feh=0.0, eep=150.0)

    if np.allclose(actual, 10.0):
        pytest.xfail("current JIT cache keeps the field selected on the first call")

    np.testing.assert_allclose(actual, 6000.0)


def test_values_broadcast_query_coordinates(tiny_grid_path):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass"])

    age = 8.5
    feh = jnp.array([[-0.25], [0.25]])
    eep = jnp.array([[125.0, 150.0, 175.0]])

    (actual,) = grid.values(age=age, feh=feh, eep=eep)
    expected = expected_mass(age, feh, eep)

    assert actual.shape == (2, 3)
    np.testing.assert_allclose(actual, expected)


def test_values_are_jittable_and_differentiable(tiny_grid_path):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass"])

    @jax.jit
    def evaluate(coordinates):
        age, feh, eep = coordinates
        return grid.values(age=age, feh=feh, eep=eep)[0]

    coordinates = jnp.array([8.5, 0.0, 150.0], dtype=jnp.float32)

    np.testing.assert_allclose(evaluate(coordinates), 10.0)
    np.testing.assert_allclose(
        jax.grad(evaluate)(coordinates),
        jnp.array([1.0, 2.0, 0.01]),
        rtol=1e-5,
    )


def test_values_include_the_exact_lower_grid_node(tiny_grid_path):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass"])

    (actual,) = grid.values(age=8.0, feh=-0.5, eep=100.0)

    np.testing.assert_allclose(actual, expected_mass(8.0, -0.5, 100.0))


@pytest.mark.parametrize(
    ("age", "feh", "eep"),
    [
        (7.9, 0.0, 150.0),
        (9.1, 0.0, 150.0),
        (8.5, -0.6, 150.0),
        (8.5, 0.6, 150.0),
        (8.5, 0.0, 90.0),
        (8.5, 0.0, 210.0),
    ],
    ids=[
        "age-below",
        "age-above",
        "feh-below",
        "feh-above",
        "eep-below",
        "eep-above",
    ],
)
def test_values_return_negative_infinity_outside_each_axis(
    tiny_grid_path, age, feh, eep
):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass"])

    (actual,) = grid.values(age=age, feh=feh, eep=eep)

    assert np.isneginf(actual)


def test_values_include_the_exact_upper_grid_node(tiny_grid_path):
    """Record the known upper-boundary bug without making it desired behavior."""
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass"])

    (actual,) = grid.values(age=9.0, feh=0.0, eep=150.0)

    if np.isnan(actual):
        pytest.xfail("current constant-fill interpolation returns NaN here")

    np.testing.assert_allclose(actual, expected_mass(9.0, 0.0, 150.0))


def test_isochrone_values_return_all_fields_without_set_keys(tiny_grid_path):
    grid = MistModel.from_isochrones(path=tiny_grid_path)

    result = raw_values(grid, age=8.5, feh=0.0, eep=150.0)

    assert isinstance(result, GridResult)
    assert tuple(result) == ("mass", "teff")
    np.testing.assert_allclose([result["mass"], result["teff"]], [10.0, 6000.0])


def test_isochrone_values_select_fields_per_call(tiny_grid_path):
    grid = MistModel.from_isochrones(path=tiny_grid_path)
    first = raw_values(grid, 8.5, 0.0, 150.0, keys="mass")
    second = raw_values(grid, 8.5, 0.0, 150.0, keys=["teff", "mass"])

    assert tuple(first) == ("mass",)
    assert tuple(second) == ("teff", "mass")
    np.testing.assert_allclose([second["teff"], second["mass"]], [6000.0, 10.0])


def test_isochrone_values_include_endpoints_and_fill_outside(tiny_grid_path):
    grid = MistModel.from_isochrones(path=tiny_grid_path)
    age = jnp.array([8.0, 8.5, 9.0, 7.9, 9.1, jnp.nan])

    result = raw_values(grid, age, 0.0, 150.0, keys="mass")

    np.testing.assert_allclose(result["mass"], [9.5, 10.0, 10.5, -np.inf, -np.inf, -np.inf])
    upper_corner = raw_values(grid, 9.0, 0.5, 200.0, keys="mass")
    np.testing.assert_allclose(upper_corner["mass"], 12.0)


def test_isochrone_values_support_jit_grad_and_vmap(tiny_grid_path):
    grid = MistModel.from_isochrones(path=tiny_grid_path)

    @jax.jit
    def evaluate(point):
        return raw_values(grid, *point, keys="mass")["mass"]

    point = jnp.array([8.5, 0.0, 150.0], dtype=jnp.float32)
    np.testing.assert_allclose(evaluate(point), 10.0)
    np.testing.assert_allclose(jax.grad(evaluate)(point), [1.0, 2.0, 0.01], rtol=1e-5)
    points = jnp.stack([point, point + jnp.array([0.25, 0.0, 0.0])])
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(points), [10.0, 10.25])


def test_isochrone_mass_lookup_returns_eep_and_selected_fields(tiny_grid_path):
    grid = MistModel.from_isochrones(path=tiny_grid_path)

    eep = grid.eep_given_mass(logage=8.5, feh_init=0.0, mass_init=10.0)
    assert eep.shape == ()
    np.testing.assert_allclose(eep, 150.0)

    result = raw_values_given_mass(grid, 8.5, 0.0, 10.0)
    assert isinstance(result, GridResult)
    assert tuple(result) == ("mass", "teff")
    np.testing.assert_allclose([result["mass"], result["teff"]], [10.0, 6000.0])
    # Inversion still needs mass when only teff is requested for the output.
    selected = raw_values_given_mass(grid, 8.5, 0.0, 10.0, keys="teff")
    assert tuple(selected) == ("teff",)
    np.testing.assert_allclose(selected["teff"], 6000.0)
    reordered = raw_values_given_mass(grid, 8.5, 0.0, 10.0, keys=["teff", "mass"])
    assert tuple(reordered) == ("teff", "mass")


def test_isochrone_mass_lookup_supports_jit_grad_and_vmap(tiny_grid_path):
    grid = MistModel.from_isochrones(path=tiny_grid_path)
    point = jnp.array([8.5, 0.0, 10.0], dtype=jnp.float32)

    evaluate_eep = jax.jit(lambda point: grid.eep_given_mass(*point))
    np.testing.assert_allclose(evaluate_eep(point), 150.0)
    np.testing.assert_allclose(jax.grad(evaluate_eep)(point), [-100.0, -200.0, 100.0])

    evaluate_teff = jax.jit(
        lambda point: raw_values_given_mass(grid, *point, keys="teff")["teff"]
    )
    np.testing.assert_allclose(jax.grad(evaluate_teff)(point), [0.0, 0.0, 100.0], atol=1e-4)
    points = jnp.stack([point, point + jnp.array([0.0, 0.0, 0.25])])
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate_eep))(points), [150.0, 175.0])
    results = jax.jit(jax.vmap(
        lambda point: raw_values_given_mass(grid, *point, keys="teff")
    ))(points)
    np.testing.assert_allclose(results["teff"], [6000.0, 6025.0])


@pytest.mark.parametrize("argument", [0, 1, 2], ids=["age", "feh", "mass"])
def test_isochrone_mass_lookup_requires_scalar_queries(tiny_grid_path, argument):
    grid = MistModel.from_isochrones(path=tiny_grid_path)
    point = [8.5, 0.0, 10.0]
    point[argument] = jnp.array([point[argument]])

    with pytest.raises(ValueError, match="scalars.*jax.vmap"):
        grid.eep_given_mass(*point)


def test_isochrone_mass_lookup_uses_physical_nonuniform_eep_nodes(tmp_path):
    age, feh, eep = np.meshgrid(
        [8.0, 9.0], [-0.5, 0.5], [100.0, 140.0, 200.0, 300.0], indexing="ij"
    )
    mass = age + 2.0 * feh + eep / 100.0
    path = tmp_path / "nonuniform_eep.npz"
    np.savez(path, logagrid=[8.0, 9.0], fgrid=[-0.5, 0.5],
             eepgrid=[100.0, 140.0, 200.0, 300.0], mass=mass)
    grid = MistModel.from_isochrones(path=path)

    np.testing.assert_allclose(grid.eep_given_mass(8.5, 0.0, 10.2), 170.0, atol=1e-4)
    np.testing.assert_allclose(raw_values_given_mass(grid, 8.5, 0.0, 10.2)["mass"], 10.2)


def test_isochrone_mass_lookup_retains_custom_interp_for_nan_curve(tmp_path):
    """Keep a valid local bracket even when the curve has an invalid prefix."""
    mass = np.broadcast_to([np.nan, 1.0, 2.0, 3.0], (2, 2, 4))
    path = tmp_path / "nan_mass_curve.npz"
    np.savez(path, logagrid=[8.0, 9.0], fgrid=[-0.5, 0.5],
             eepgrid=[0.0, 10.0, 20.0, 30.0], mass=mass, teff=5000.0 + 100.0 * mass)
    legacy, named = MistGridIso(path=path), MistModel.from_isochrones(path=path)
    legacy.set_keys(["teff", "mass"])

    np.testing.assert_allclose(named.eep_given_mass(8.5, 0.0, 1.5), 15.0)
    np.testing.assert_allclose(legacy.eep_given_mass(8.5, 0.0, 1.5)[0], 15.0)
    result = raw_values_given_mass(named, 8.5, 0.0, 1.5, keys=["teff", "mass"])
    np.testing.assert_allclose([result["teff"], result["mass"]], [5150.0, 1.5])
    np.testing.assert_allclose(legacy.values_given_mass(8.5, 0.0, 1.5), [5150.0, 1.5])


def test_isochrone_mass_lookup_retains_inverse_endpoint_limitations(tiny_grid_path):
    """Characterize the retained helper; these are not new validity guarantees."""
    grid = MistModel.from_isochrones(path=tiny_grid_path)
    np.testing.assert_allclose(grid.eep_given_mass(8.5, 0.0, 9.5), 100.0)
    # At the last mass node, interp's clipped upper index gives a zero denominator.
    assert np.isnan(grid.eep_given_mass(8.5, 0.0, 10.5))
    assert np.isnan(grid.eep_given_mass(7.9, 0.0, 10.0))
    result = raw_values_given_mass(grid, 7.9, 0.0, 10.0, keys="mass")
    assert np.isneginf(result["mass"])


def test_mistfit_uses_named_grid_with_optional_legacy_arithmetic(tiny_grid_path):
    named = MistFit(observations={}, path=tiny_grid_path)
    legacy = MistFit(observations={}, path=tiny_grid_path, legacy_interpolation=True)

    assert named.physical_model.kind == legacy.physical_model.kind == "isochrone"
    assert not hasattr(named, "mg") and not hasattr(legacy, "mg")
    assert named.physical_model.legacy_interpolation is False
    assert legacy.physical_model.legacy_interpolation is True
    with pytest.raises(TypeError, match="legacy_interpolation"):
        MistFit(observations={}, path=tiny_grid_path, legacy_interpolation="legacy")
    # Do not silently reinterpret the removed full legacy backend as the new option.
    with pytest.raises(TypeError, match="grid_backend"):
        MistFit(observations={}, path=tiny_grid_path, grid_backend="legacy")


def test_mistfit_modes_have_matching_model_values_and_gradients(tiny_grid_path, tmp_path):
    """Exercise interpolation modes, observations and per-call outputs through the real model."""
    from numpyro.infer.util import log_density

    with np.load(tiny_grid_path) as data:
        fields = {name: data[name] for name in data.files}
    mass = 0.5 + fields["mass"] / 20.0
    fields["mass"] = mass
    for name, value in {
        "kmag": 1.0, "jmag": 2.0, "logg": 4.0, "radius": 1.0,
        "feh_photosphere": 0.0, "dmdeep": 0.0005, "mmin": 0.8, "mmax": 1.2,
        "bpmag2": 1.5, "rpmag2": 1.0,
    }.items():
        fields[name] = np.full_like(mass, value)
    fields["star_mass"] = mass
    path = tmp_path / "model_grid.npz"
    np.savez(path, **fields)

    results = []
    point = jnp.array([10**8.5 / 1e9, 0.0, 150.0, 0.1])
    for backend in ("legacy", "named"):
        fit = MistFit(path=path, legacy_interpolation=(backend == "legacy"),
                      observations={"teff": (6100., 100.), "parallax": (10., .1)})

        def log_joint(point):
            params = dict(zip(("age", "feh_init", "eep", "distance"), point))
            return log_density(fit.model, (), {"output_keys": ("jmag",)}, params)[0]

        result = jax.jit(jax.value_and_grad(log_joint))(point)
        assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(result))
        results.append(result)

    for old, new in zip(jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(results[1])):
        np.testing.assert_allclose(old, new, rtol=1e-5, atol=1e-5)


# Grid-path and first-use behavior. These must stay offline in the default suite.


def test_mistfit_accepts_explicit_grid_path(tiny_grid_path, monkeypatch):
    monkeypatch.setenv(
        MISTGRID_ENV_VAR, str(tiny_grid_path.with_name("missing_grid.npz"))
    )

    fit = MistFit(observations={}, path=tiny_grid_path)

    np.testing.assert_array_equal(fit.physical_model.grid.axis("age"), [8.0, 9.0])


def test_explicit_missing_path_does_not_generate_a_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(isochrone_module, "create_mistgrid",
                        lambda **kw: pytest.fail("an explicit missing path must not generate a grid"))
    with pytest.raises(FileNotFoundError):
        MistModel.from_isochrones(tmp_path / "missing.npz")


def test_factory_rejects_missing_axis_before_constructing_grid(tmp_path):
    path = tmp_path / "missing_axis.npz"
    np.savez(path, logagrid=[8., 9.], fgrid=[-.5, .5], mass=np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="missing axes.*eepgrid"):
        MistModel.from_isochrones(path)


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_environment_grid_path_is_reused(tiny_grid_path, monkeypatch, backend):
    monkeypatch.setenv(MISTGRID_ENV_VAR, str(tiny_grid_path))
    monkeypatch.setattr(
        isochrone_module,
        "create_mistgrid",
        lambda path: pytest.fail("an existing environment grid was regenerated"),
    )

    fit = MistFit(observations={}, legacy_interpolation=(backend == "legacy"))

    np.testing.assert_allclose(raw_values(fit.physical_model, 8.5, 0.0, 150.0, keys="mass")["mass"], 10.0)


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_missing_grid_is_created_in_user_cache(tiny_grid_path, tmp_path, monkeypatch, backend):
    cache_directory = tmp_path / "cache"
    expected_path = (
        cache_directory / MISTGRID_CACHE_VERSION / "mistgrid_iso.npz"
    )
    generated_paths = []

    monkeypatch.delenv(MISTGRID_ENV_VAR, raising=False)
    monkeypatch.setattr(
        paths_module,
        "user_cache_path",
        lambda *args, **kwargs: cache_directory,
    )

    def create_test_grid(path):
        generated_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tiny_grid_path, path)
        return path

    monkeypatch.setattr(isochrone_module, "create_mistgrid", create_test_grid)

    fit = MistFit(observations={}, legacy_interpolation=(backend == "legacy"))
    second_fit = MistFit(observations={}, legacy_interpolation=(backend == "legacy"))

    assert generated_paths == [expected_path]
    assert expected_path.is_file()
    for instance in (fit, second_fit):
        np.testing.assert_allclose(raw_values(instance.physical_model, 8.5, 0.0, 150.0, keys="mass")["mass"], 10.0)


@pytest.mark.full_grid
@pytest.mark.skipif(
    not os.environ.get(MISTGRID_ENV_VAR),
    reason=f"set {MISTGRID_ENV_VAR} to run the full-grid regression test",
)
def test_full_grid_matches_historical_reference_values():
    """Check the generated MIST v1.2 grid against the original test fixture."""
    reference = pd.read_csv(Path(__file__).with_name("test_data.txt")).iloc[300]
    model = MistModel.from_isochrones(os.environ[MISTGRID_ENV_VAR])
    keys = ("kmag_abs", "teff", "logg", "mass", "star_mass", "feh_photosphere")
    result = model({"logage": 9.3, "feh_init": 0.1, "eep": reference.EEP}, keys=keys)
    actual = np.asarray([result[key] for key in keys])
    expected = reference[
        ["2MASS_Ks", "teff", "log_g", "initial_mass", "star_mass", "[Fe/H]"]
    ].to_numpy(dtype=float)

    np.testing.assert_allclose(actual[:5], expected[:5], rtol=1e-3)
    np.testing.assert_allclose(actual[5], expected[5], rtol=0.15)
