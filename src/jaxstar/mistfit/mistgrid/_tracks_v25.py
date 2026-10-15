"""MIST v2.5 native EEP tracks and fixed-alpha UBVRIplus BC v2 tables.

BC temperatures are already log10(K). Alpha is selected at build time, not
interpolated. Keep the release-specific chemistry separate from v1.2.
"""

from pathlib import PurePosixPath
import re
import tarfile

import numpy as np

from ._tracks_v12 import BANDS, MBOL_SUN, _read_track, bc_coverage, track_bc

# Solar-scaled CMD normalization, checked against paired theoretical/CMD
# outputs from https://mist.science/ (2026-10-15). These are photospheric
# conversion constants, not the initial X/Z in a native track's header.
# Nonzero alpha needs a separate normalization and is not yet enabled in the
# photometric builder. See benchmarks/mist_v25/README.md for the reference gate.
ZSUN = .01685
XSUN = .73465

# The 66 native non-isotope columns, in the official v2.5 EEP header order.
# Keep original names/units, including phase as a numerical diagnostic rather
# than a categorical classification. The 102 surface/central isotope columns
# are deliberately not part of this optional selection.
STRUCTURE_COLUMNS = (
    "star_age", "star_mass", "star_mdot", "he_core_mass", "c_core_mass", "o_core_mass",
    "log_L", "log_L_div_Ledd", "log_LH", "log_LHe", "log_LZ", "log_Teff",
    "log_abs_Lgrav", "log_R", "log_g", "log_surf_cell_z", "surf_avg_omega",
    "surf_avg_v_rot", "surf_num_c12_div_num_o16", "surf_avg_omega_crit",
    "surf_avg_omega_div_omega_crit", "surf_avg_v_crit", "surf_avg_v_div_v_crit",
    "surf_avg_Lrad_div_Ledd", "v_div_csound_surf", "surf_r_equatorial_div_r",
    "surf_r_polar_div_r", "total_angular_momentum", "grav_dark_L_polar",
    "grav_dark_Teff_polar", "grav_dark_L_equatorial", "grav_dark_Teff_equatorial",
    "log_center_T", "log_center_Rho", "center_degeneracy", "center_omega",
    "center_gamma", "mass_conv_core", "pp", "cno", "tri_alfa", "burn_c", "burn_n",
    "burn_o", "c12_c12", "apsidal_constant_k2", "delta_nu", "delta_Pg", "nu_max",
    "acoustic_cutoff", "max_conv_vel_div_csound", "max_gradT_div_grada",
    "gradT_excess_alpha", "min_Pgas_div_P", "max_L_rad_div_Ledd", "e_thermal",
    "conv_env_top_mass", "conv_env_bot_mass", "conv_env_top_radius",
    "conv_env_bot_radius", "conv_env_turnover_time_l_t", "conv_env_turnover_time_l_b",
    "conv_env_turnover_time_g", "envelope_binding_energy", "total_moment_of_inertia",
    "phase",
)


def surface_composition(data, initial_feh):
    """Solar-scaled CMD convention; retain missing chemistry as NaN.

    The BC coordinate is log(Z/Zsun), capped at initial FeH+0.1 for dredge-up.
    The reported surface FeH additionally accounts for surface hydrogen. This
    is the CMD convention, not an elemental surface_fe56/H measurement.
    """
    log_z = data["log_surf_cell_z"]
    metal = np.where(np.isfinite(log_z),
                     np.minimum(log_z-np.log10(ZSUN), initial_feh+.1), np.nan)
    hydrogen = data["surface_h1"]
    hydrogen = np.where(np.isfinite(hydrogen) & (hydrogen > 0), hydrogen, np.nan)
    return metal, metal-np.log10(hydrogen/XSUN)


def read_track(stream):
    """Validate the native file header and retain named v2.5 chemistry columns."""
    return _read_track(stream, version="2.5", composition_columns=(
        "log_surf_cell_z", "surface_h1", "surface_fe56"))


def read_bc_archive(path, *, afe=0.):
    """Read one fixed-alpha Av=0/Rv=3.1 slice, validating units and composition."""
    planes = []
    with tarfile.open(path, "r|xz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".UBVRIplus"):
                continue
            match = re.fullmatch(r"feh([+-][0-9.]+)_afe([+-][0-9.]+)\.UBVRIplus",
                                 PurePosixPath(member.name).name)
            if match is None:
                raise ValueError(f"Invalid BC v2 member name: {member.name}")
            named_feh, named_afe = map(float, match.groups())
            if named_afe != afe:
                continue
            with archive.extractfile(member) as stream:
                for raw in stream:
                    words = raw.decode("ascii").lstrip("#").split()
                    if words[:6] == ["lgTef", "logg", "Fe_H", "a_Fe", "Av", "Rv"]:
                        names = words
                        break
                else:
                    raise ValueError(f"BC v2 column header not found: {member.name}")
                values = np.loadtxt(stream, ndmin=2)
            if values.shape[1] != len(names) or len(set(names)) != len(names):
                raise ValueError(f"BC column count mismatch: {member.name}")
            if not set(BANDS).issubset(names):
                raise ValueError(f"Missing UBVRIplus bands in {member.name}")
            if (not np.all(np.isfinite(values[:, :6]))
                    or not np.all(values[:, 2] == named_feh)
                    or not np.all(values[:, 3] == named_afe)):
                raise ValueError(f"BC composition/axes disagree with member name: {member.name}")
            selected = values[(values[:, 4] == 0.) & (values[:, 5] == 3.1)]
            if not len(selected):
                raise ValueError(f"Missing Av=0/Rv=3.1 BC slice: {member.name}")
            temperature, gravity = (np.unique(selected[:, i]) for i in (0, 1))
            order = np.lexsort((selected[:, 1], selected[:, 0]))
            selected = selected[order]
            mesh = np.stack(np.meshgrid(temperature, gravity, indexing="ij"), -1)
            if not np.array_equal(selected[:, :2], mesh.reshape(-1, 2)):
                raise ValueError("BC temperature/gravity grid has missing or duplicate nodes")
            fields = selected[:, [names.index(band) for band in BANDS]]
            if not np.all(np.isfinite(fields)):
                raise ValueError("Nonfinite BC table values")
            planes.append((named_feh, temperature, gravity,
                           fields.reshape(len(temperature), len(gravity), -1)))
    if len(planes) < 4:
        raise ValueError(f"At least four BC metallicities are needed at afe={afe}")
    planes.sort(key=lambda item: item[0])
    z = np.array([p[0] for p in planes])
    t, g = planes[0][1:3]
    if len(t) < 4 or len(g) < 2 or not np.all(np.diff(z) > 0):
        raise ValueError("BC axes need four temperatures/metallicities and two gravities; no duplicates")
    if any(not np.array_equal(t, p[1]) or not np.array_equal(g, p[2]) for p in planes):
        raise ValueError("BC atmosphere axes differ between metallicities")
    return (z, t, g), np.stack([p[3] for p in planes])
