import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if SRC.exists():
    sys.path.insert(0, str(SRC))


import numpy as np
import pytest


@pytest.fixture
def synthetic_mistgrid(tmp_path):
    """Synthetic mistgrid fixture to avoid external downloads."""
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
