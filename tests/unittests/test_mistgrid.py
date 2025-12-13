import numpy as np
from jaxstar.mistfit.mistfit import MistGridIso, MistFit


def _template_value(config, age, feh, eep):
    agrid = config["agrid"]
    fgrid = config["fgrid"]
    egrid = config["eepgrid"]
    aidx = (age - agrid[0]) / (agrid[1] - agrid[0])
    fidx = (feh - fgrid[0]) / (fgrid[1] - fgrid[0])
    eidx = (eep - egrid[0]) / (egrid[1] - egrid[0])
    return aidx + 10.0 * fidx + 100.0 * eidx


def test_mistgrid_values_interpolate_without_real_downloads(synthetic_mistgrid):
    cfg = synthetic_mistgrid
    grid = MistGridIso(path=str(cfg["path"]))
    grid.set_keys(["kmag"])
    age, feh, eep = 9.25, -0.25, 505.0

    value = float(np.asarray(grid.values(age, feh, eep)[0]))
    expected = _template_value(cfg, age, feh, eep)
    assert np.isclose(value, expected)


def test_values_given_mass_consistent_with_direct_lookup(synthetic_mistgrid):
    cfg = synthetic_mistgrid
    grid = MistGridIso(path=str(cfg["path"]))
    grid.set_keys(["mass", "radius"])
    age, feh, eep = 9.25, -0.25, 505.0

    mass_value, radius_direct = [
        float(np.asarray(val)) for val in grid.values(age, feh, eep)
    ]
    radius_from_mass = float(
        np.asarray(grid.values_given_mass(age, feh, mass_value)[1])
    )
    expected_radius = (
        cfg["radius_offset"] + cfg["radius_scale"] * _template_value(cfg, age, feh, eep)
    )

    assert np.isclose(radius_direct, expected_radius)
    assert np.isclose(radius_from_mass, expected_radius)


def test_eep_given_mass_tracks_mass_grid(synthetic_mistgrid):
    cfg = synthetic_mistgrid
    grid = MistGridIso(path=str(cfg["path"]))
    grid.set_keys(["mass"])
    age, feh = 9.25, -0.25
    target_mass = 1.05

    eep, aidx, fidx = grid.eep_given_mass(age, feh, target_mass)
    # EEP should land inside the synthetic grid bounds
    assert cfg["eepgrid"][0] <= eep <= cfg["eepgrid"][-1]
    # Spot-check that plugging the EEP back in yields the same mass
    mass_back = float(np.asarray(grid.values(age, feh, eep)[0]))
    assert np.isclose(mass_back, target_mass, rtol=1e-2, atol=1e-3)


def test_mistfit_uses_supplied_grid_and_tracks_custom_keys(synthetic_mistgrid):
    cfg = synthetic_mistgrid
    mistfit = MistFit(path=str(cfg["path"]))
    mistfit.set_data(
        keys=["kmag", "custom_flux"],
        vals=np.array([0.0, 1.0]),
        errs=np.array([0.1, 0.2]),
    )
    assert "kmag" in mistfit.outkeys
    assert "custom_flux" in mistfit.outkeys

    mistfit.mg.set_keys(["kmag"])
    age, feh, eep = 9.25, -0.25, 505.0
    value = float(np.asarray(mistfit.mg.values(age, feh, eep)[0]))
    expected = _template_value(cfg, age, feh, eep)
    assert np.isclose(value, expected)
