import numpy as np
import pytest

from jaxstar.mistfit.mistfit import MistGridIso, MistFit


def _template_value(config, age, feh, eep):
    agrid = config["agrid"]
    fgrid = config["fgrid"]
    egrid = config["eepgrid"]
    aidx = (age - agrid[0]) / (agrid[1] - agrid[0])
    fidx = (feh - fgrid[0]) / (fgrid[1] - fgrid[0])
    eidx = (eep - egrid[0]) / (egrid[1] - egrid[0])
    return aidx + 10.0 * fidx + 100.0 * eidx


@pytest.fixture
def synthetic_mistgrid(tmp_path):
    agrid = np.array([9.0, 9.5])
    fgrid = np.array([-0.5, 0.0])
    eepgrid = np.array([500.0, 510.0, 515.0])
    age_idx = np.arange(len(agrid), dtype=float)[:, None, None]
    feh_idx = np.arange(len(fgrid), dtype=float)[None, :, None]
    eep_idx = np.arange(len(eepgrid), dtype=float)[None, None, :]
    template = age_idx + 10.0 * feh_idx + 100.0 * eep_idx

    def make_grid(scale=1.0, offset=0.0):
        return offset + scale * template

    mass_base = np.array([0.85, 1.0, 1.15], dtype=float)
    mass_age_coeff = 0.05
    mass_feh_coeff = 0.02
    mass_grid = (
        mass_base.reshape(1, 1, -1)
        + mass_age_coeff * age_idx
        + mass_feh_coeff * feh_idx
    )

    radius_scale = 0.5
    radius_offset = 2.0

    grid_data = {
        "logagrid": agrid,
        "fgrid": fgrid,
        "eepgrid": eepgrid,
        "kmag": make_grid(),
        "teff": 5000.0 + 20.0 * make_grid(),
        "logt": make_grid(0.01),
        "logg": make_grid(0.02),
        "logage": make_grid(0.03),
        "logl": make_grid(0.04),
        "jmag": make_grid(0.1),
        "hmag": make_grid(0.2),
        "gmag": make_grid(0.3),
        "bpmag": make_grid(0.4),
        "rpmag": make_grid(0.5),
        "gmag3": make_grid(0.6),
        "bpmag3": make_grid(0.7),
        "rpmag3": make_grid(0.8),
        "umag": make_grid(0.9),
        "bmag": make_grid(1.0),
        "vmag": make_grid(1.1),
        "rmag": make_grid(1.2),
        "imag": make_grid(1.3),
        "dmdeep": make_grid(1.4),
        "mmin": np.full_like(template, 0.2),
        "mmax": np.full_like(template, 3.0),
        "eepmin": np.full_like(template, eepgrid[0]),
        "eepmax": np.full_like(template, eepgrid[-1]),
        "star_mass": make_grid(1.5),
        "feh_photosphere": make_grid(1.6),
        "radius": radius_offset + radius_scale * template,
        "mass": mass_grid,
    }

    path = tmp_path / "mistgrid_iso.npz"
    np.savez(path, **grid_data)
    return {
        "path": path,
        "agrid": agrid,
        "fgrid": fgrid,
        "eepgrid": eepgrid,
        "mass_base": mass_base,
        "mass_age_coeff": mass_age_coeff,
        "mass_feh_coeff": mass_feh_coeff,
        "radius_scale": radius_scale,
        "radius_offset": radius_offset,
    }


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
