import numpy as np
import pytest

from jaxstar.mistfit import MistFit, MistGridIso


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


def test_mistfit_accepts_explicit_grid_path(tiny_grid_path):
    fit = MistFit(path=tiny_grid_path)

    np.testing.assert_array_equal(fit.mg.dgrid["logagrid"], [8.0, 9.0])
