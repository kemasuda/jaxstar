import importlib
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxstar.mistfit import MistFit, MistGridIso
from jaxstar.mistfit.mistgrid.paths import (
    MISTGRID_CACHE_VERSION,
    MISTGRID_ENV_VAR,
)


mistfit_module = importlib.import_module("jaxstar.mistfit.mistfit")
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


def expected_mass(age, feh, eep):
    """The physical-coordinate rule used to construct ``tiny_grid_path``."""
    return age + 2.0 * feh + eep / 100.0


# Forward-interpolation characterization. These tests record current behavior
# so a replacement keeps or changes each part explicitly.


def test_values_interpolate_selected_fields_in_requested_order(tiny_grid_path):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["teff", "mass"])

    teff, mass = grid.values(age=8.5, feh=0.0, eep=150.0)

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


# Grid-path and first-use behavior. These must stay offline in the default suite.


def test_mistfit_accepts_explicit_grid_path(tiny_grid_path, monkeypatch):
    monkeypatch.setenv(
        MISTGRID_ENV_VAR, str(tiny_grid_path.with_name("missing_grid.npz"))
    )

    fit = MistFit(path=tiny_grid_path)

    np.testing.assert_array_equal(fit.mg.dgrid["logagrid"], [8.0, 9.0])


def test_environment_grid_path_is_reused(tiny_grid_path, monkeypatch):
    monkeypatch.setenv(MISTGRID_ENV_VAR, str(tiny_grid_path))
    monkeypatch.setattr(
        mistfit_module,
        "create_mistgrid",
        lambda path: pytest.fail("an existing environment grid was regenerated"),
    )

    fit = MistFit()

    np.testing.assert_array_equal(fit.mg.dgrid["fgrid"], [-0.5, 0.5])


def test_missing_grid_is_created_in_user_cache(tiny_grid_path, tmp_path, monkeypatch):
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

    monkeypatch.setattr(mistfit_module, "create_mistgrid", create_test_grid)

    fit = MistFit()
    second_fit = MistFit()

    assert generated_paths == [expected_path]
    assert expected_path.is_file()
    np.testing.assert_array_equal(fit.mg.dgrid["eepgrid"], [100.0, 200.0])
    np.testing.assert_array_equal(second_fit.mg.dgrid["eepgrid"], [100.0, 200.0])


@pytest.mark.full_grid
@pytest.mark.skipif(
    not os.environ.get(MISTGRID_ENV_VAR),
    reason=f"set {MISTGRID_ENV_VAR} to run the full-grid regression test",
)
def test_full_grid_matches_historical_reference_values():
    """Check the generated MIST v1.2 grid against the original test fixture."""
    reference = pd.read_csv(Path(__file__).with_name("test_data.txt")).iloc[300]
    grid = MistGridIso(path=os.environ[MISTGRID_ENV_VAR])
    grid.set_keys(
        ["kmag", "teff", "logg", "mass", "star_mass", "feh_photosphere"]
    )

    actual = np.asarray(grid.values(age=9.3, feh=0.1, eep=reference.EEP))
    expected = reference[
        ["2MASS_Ks", "teff", "log_g", "initial_mass", "star_mass", "[Fe/H]"]
    ].to_numpy(dtype=float)

    np.testing.assert_allclose(actual[:5], expected[:5], rtol=1e-3)
    np.testing.assert_allclose(actual[5], expected[5], rtol=0.15)
