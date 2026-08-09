import importlib
import os
import shutil
from pathlib import Path

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
    """Create a deterministic grid small enough for unit tests."""
    logage = np.array([8.0, 9.0])
    feh = np.array([-0.5, 0.5])
    eep = np.array([100.0, 200.0])

    age_index, feh_index, eep_index = np.indices((2, 2, 2))
    mass = 1.0 + age_index + 2.0 * feh_index + 4.0 * eep_index
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


def test_values_from_explicit_grid(tiny_grid_path):
    grid = MistGridIso(path=tiny_grid_path)
    grid.set_keys(["mass", "teff"])

    values = np.asarray(grid.values(age=8.5, feh=0.0, eep=150.0))

    np.testing.assert_allclose(values, [4.5, 5450.0])


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
