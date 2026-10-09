
__all__ = ["create_mistgrid"]

# %%
import numpy as np
import pandas as pd
from astropy.constants import M_sun, R_sun, G
from scipy.interpolate import interp1d
from .cache import extract_archive
from .paths import default_mistgrid_path, resolve_mistgrid_path
from .sources import isochrone_archive
logg_sun = np.log10((G * M_sun / R_sun**2).cgs.value)

# %%
keys = ['2MASS_J', '2MASS_H', '2MASS_Ks', 'logT', 'logg', 'teff', 'logage',
        'mass', 'dmdeep', 'logL', 'radius', 'mmin', 'mmax', 'eepmin', 'eepmax']
keys += ['Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev']
keys += ['Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3']
keys += ['star_mass', 'feh_photosphere']
keys += ['Bessell_U',  'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I']


def _cmd_directory(output, raw_cache_dir, offline):
    source = isochrone_archive()
    directory_name = source.filename.removesuffix(".txz")
    # Preserve earlier downloads/extractions, including when the NPZ output
    # changes. New archives/extractions live in the shared raw cache.
    parents = tuple(dict.fromkeys((output.parent, default_mistgrid_path().parent)))
    for parent in parents:
        directory = parent / directory_name
        if directory.is_dir() and any(directory.glob("*.cmd")):
            return directory
    local = next((parent / source.filename for parent in parents
                  if (parent / source.filename).is_file()), None)
    root = extract_archive(source, cache_dir=raw_cache_dir, local_path=local, offline=offline)
    directory = root / directory_name
    if not directory.is_dir() or not any(directory.glob("*.cmd")):
        raise ValueError(f"No CMD files in the expected archive directory: {directory}")
    return directory


def create_mistgrid(path=None, *, raw_cache_dir=None, offline=False):
    """Create the historical v1.2 isochrone grid, reusing cached source CMDs.

    Args:
        path: output path for the generated grid. If omitted, use the
            environment override or the platform-specific user cache.
        raw_cache_dir: shared raw-data cache; defaults to JAXSTAR_MIST_RAW_DIR
            or the platform cache. Independent of the NPZ output path.
        offline: require cached/local source files; never download.

    Returns:
        Path: path to the generated grid
    """
    mistgrid_path = resolve_mistgrid_path(path)
    mistgrid_path.parent.mkdir(parents=True, exist_ok=True)
    mistdir_path = _cmd_directory(mistgrid_path, raw_cache_dir, offline)
    filenames = sorted(mistdir_path.glob("*.cmd"))
    print("CMD files found in %s" % mistdir_path)
    print("creating grid for mistfit...")

    # %%
    header = ['EEP', 'log10_isochrone_age_yr', 'initial_mass', 'star_mass', 'log_Teff', 'log_g', 'log_L', '[Fe/H]_init', '[Fe/H]',  'Bessell_U',  'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I', '2MASS_J', '2MASS_H', '2MASS_Ks', 'Kepler_Kp',
              'Kepler_D51', 'Hipparcos_Hp', 'Tycho_B', 'Tycho_V', 'Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev', 'Gaia_G_MAW', 'Gaia_BP_MAWb', 'Gaia_BP_MAWf', 'Gaia_RP_MAW', 'TESS', 'Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3', 'phase']

    # %%
    df_all = pd.DataFrame(data={})
    for filename in filenames:
        d = pd.read_csv(filename, sep=r'\s+',
                        comment='#', header=None, names=header)
        d['mass'] = d['initial_mass']
        d['feh'] = d['[Fe/H]_init']
        d['feh_photosphere'] = d['[Fe/H]']
        d['teff'] = 10 ** d['log_Teff']
        d['radius'] = np.sqrt(d['star_mass'] / 10 ** (d['log_g'] - logg_sun))
        d['logage'] = d['log10_isochrone_age_yr']
        d['age'] = 10**(d.logage) / 1e9
        df_all = pd.concat([df_all, d], ignore_index=True)
    df_all = df_all.reset_index(drop=True)

    # %% Fe/H grid cut at -1.5
    d = df_all.sort_values(["logage", "feh", "EEP"]).reset_index(drop=True)
    agrid = np.sort(list(set(d.logage)))
    fgrid = np.sort(list(set(d.feh)))[6:]
    eepgrid = np.sort(list(set(d.EEP)))
    print("logage(yr) grid:", agrid)
    print("feh grid:", fgrid)
    print("EEP grid:", eepgrid)

    # %%
    def eepderivative(y):
        # Unit-spaced EEP: centered differences at interior nodes, copying the
        # neighboring difference at each endpoint. dmdeep is stored and later
        # interpolated independently of mass values to approximate the smooth
        # physical relation; do not replace it with AD of the linear interpolant.
        dy = 0.5 * (y[2:] - y[:-2])
        return np.array([dy[0]]+list(dy)+[dy[-1]])

    # %%
    d = d.rename({"log_Teff": 'logT', "log_L": "logL",
                 "log_g": "logg"}, axis='columns')

    # %%
    pgrids2d = []
    for key in keys:
        pgrid2d = np.zeros((len(agrid), len(fgrid), len(eepgrid)))
        for i, a in enumerate(agrid):
            for j, f in enumerate(fgrid):
                _d = d[(d.logage == a) & (d.feh == f)]
                eeparr = np.ones_like(eepgrid) * -np.inf
                eep0, eep1 = int(_d.EEP.min()), int(_d.EEP.max())+1
                if key == 'dmdeep':
                    _marr = interp1d(_d.EEP, _d['mass'])(np.arange(eep0, eep1))
                    eeparr[eep0:eep1] = eepderivative(_marr)
                elif key == 'mmin':
                    eeparr = _d['mass'].min()
                elif key == 'mmax':
                    eeparr = _d['mass'].max()
                elif key == 'eepmin':
                    eeparr = eep0
                elif key == 'eepmax':
                    eeparr = eep1 - 1
                else:
                    eeparr[eep0:eep1] = interp1d(
                        _d.EEP, _d[key])(np.arange(eep0, eep1))
                pgrid2d[i][j] = eeparr
                # pgrid2d.append(eeparr.reshape(len(mgrid), len(fgrid)))
        # pgrid2d = np.array(pgrid2d)
        pgrids2d.append(pgrid2d)

    # %%
    np.savez(mistgrid_path, logagrid=agrid, fgrid=fgrid, eepgrid=eepgrid,
             jmag=pgrids2d[0], hmag=pgrids2d[1], kmag=pgrids2d[2],
             logt=pgrids2d[3], logg=pgrids2d[4], teff=pgrids2d[5], logage=pgrids2d[6], mass=pgrids2d[7],
             dmdeep=pgrids2d[8], logl=pgrids2d[9], radius=pgrids2d[10], mmin=pgrids2d[11], mmax=pgrids2d[
                 12], eepmin=pgrids2d[13], eepmax=pgrids2d[14], gmag2=pgrids2d[15], bpmag2=pgrids2d[16], rpmag2=pgrids2d[17],
             gmag3=pgrids2d[18], bpmag3=pgrids2d[19], rpmag3=pgrids2d[20], star_mass=pgrids2d[21], feh_photosphere=pgrids2d[22],
             umag=pgrids2d[23], bmag=pgrids2d[24], vmag=pgrids2d[25], rmag=pgrids2d[26], imag=pgrids2d[27])
    return mistgrid_path
