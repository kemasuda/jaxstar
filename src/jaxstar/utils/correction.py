
__all__ = ["correct_gedr3_parallax", "correct_kmag", "correct_gedr3_parallax_error", "correct_gedr3_parallax_zeropoint", "extinction_mag_vector"]

import numpy as np
import pandas as pd


def parallax_error_correction_gedr3(gmags, a=0.21, g0=12.65, b=0.90, p=[-0.00062, 0.0040, 1.141]):
    """ eq.16 of El-Badry, Rix, Heinz MNRAS 506, 2269–2295 (2021)

        Args:
            gmags: Gaia G magnitude

    """
    return a * np.exp(-(gmags-g0)**2 / b**2) + np.poly1d(p)(gmags)


def correct_gedr3_parallax_error(parallax_error, phot_g_mean_mag):
    """ parallax error correction following eq.16 of El-Badry, Rix, Heinz MNRAS 506, 2269–2295 (2021)

        Args:
            parallax_error: parallax error (mas) in the Gaia catalog
            phot_g_mean_mag: Gaia G magnitude

        Returns:
            corrected parallax error (mas)

    """
    return parallax_error * parallax_error_correction_gedr3(phot_g_mean_mag)


def _first_non_missing(values):
    for val in np.atleast_1d(values):
        if pd.isna(val):
            continue
        return val
    return None


def _ensure_quantity(values, default_unit, units_module):
    """Ensure scalars/arrays/Series carry the required astropy unit."""
    units = units_module
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)

    raw = series.values
    if isinstance(raw, units.Quantity):
        return raw.to(default_unit)

    if series.dtype == object:
        template = _first_non_missing(raw)
        if template is not None and hasattr(template, "to"):
            converted = [np.nan if pd.isna(val) else val.to_value(default_unit) for val in raw]
            return units.Quantity(converted, unit=default_unit)

    numeric = pd.to_numeric(series, errors="coerce")
    return units.Quantity(numeric.to_numpy(), unit=default_unit, copy=False)


def correct_gedr3_parallax_zeropoint(d):
    """ parallax zeropoint correction using gaiaedr3_zeropoint

        Args:
            d: gaia catalog

        Returns:
            corrected parallax (mas)

    """
    from zero_point import zpt
    zpt.load_tables()
    d["zpt"] = d.apply(zpt.zpt_wrapper, axis=1)
    return d.parallax - d.zpt


def extinction_mag_vector(l, b, distpc, version='2019'):
    """ reddening vector using dustmaps

        Args:
            l, b: galactic coordinates
            distpc: distance in parsec

        Returns:
            extinction(mag) in grizy+2MASS JHKs, error (assumed to be 30% of A)
            grizyJHKs

    """
    from dustmaps.bayestar import BayestarQuery as bquery
    from astropy.coordinates import SkyCoord
    import astropy.units as units
    # extinction vector from Table 1 of Green et al. (2019) https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G/abstract
    # Pan-STARRS 1 grizy, 2MASS JHKs
    Rvect_b19 = np.array([3.518, 2.617, 1.971, 1.549, 1.263, 0.7927, 0.4690, 0.3026])
    Rvect_b17 = np.array([3.384, 2.483, 1.838, 1.414, 1.126, 0.650, 0.327, 0.161])

    bayestar = bquery(version='bayestar'+version)
    if version == '2019':
        Rvect = Rvect_b19
    else:
        version == '2017'
        Rvect = Rvect_b17
    print ("# bayestar%s is used."%version)

    l_vals = _ensure_quantity(l, units.deg, units)
    b_vals = _ensure_quantity(b, units.deg, units)
    distance_vals = _ensure_quantity(distpc, units.pc, units)
    coords = SkyCoord(l=l_vals, b=b_vals, distance=distance_vals, frame='galactic')
    reddening = bayestar(coords, mode='median')
    ak = reddening * Rvect
    ak_err = 0.3 * ak # Fulton & Petigura
    return ak, ak_err


# eq.16 from El-Badry et al. (2021) MNRAS 506, 2269
def edr3error(gmags, a=0.21, g0=12.65, b=0.90, p=[-0.00062, 0.0040, 1.141]):
    return a * np.exp(-(gmags-g0)**2 / b**2) + np.poly1d(p)(gmags)

#%%
def correct_gedr3_parallax(d):
    from zero_point import zpt
    zpt.load_tables()
    d["zpt"] = d.apply(zpt.zpt_wrapper, axis=1)
    d["parallax_zpcorrected"] = d.parallax - d.zpt
    print ("# zp correction failed for %s stars (original parallax used)"%np.sum(d.parallax_zpcorrected != d.parallax_zpcorrected))
    d['parallax_corrected'] = d['parallax_zpcorrected'].fillna(d.parallax)
    d['parallax_error_corrected'] = d.parallax_error * edr3error(d.phot_g_mean_mag)
    return d

#%%
# extinction vector from Table 1 of Green et al. (2019) https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G/abstract
# Pan-STARRS 1 grizy, 2MASS JHKs
Rvect_b19 = np.array([3.518, 2.617, 1.971, 1.549, 1.263, 0.7927, 0.4690, 0.3026])
Rvect_b17 = np.array([3.384, 2.483, 1.838, 1.414, 1.126, 0.650, 0.327, 0.161])

#%%
def correct_kmag(d, version='2019'):
    from dustmaps.bayestar import BayestarQuery as bquery
    from astropy.coordinates import SkyCoord
    import astropy.units as units

    print ("# %d stars include nan."%np.sum((d.kmag_err!=d.kmag_err)&(d.parallax!=d.parallax)))

    d['distpc'] = 1e3 / d.parallax
    d.loc[d['distpc']<0, 'distpc'] = 8e3
    d['kmag_err'] = d['kmag_err'].fillna(np.nanmedian(d.kmag_err))

    bayestar = bquery(version='bayestar'+version)
    if version == '2019':
        Rvect = Rvect_b19
    elif version == '2017':
        Rvect = Rvect_b17
    else:
        version = '2017'
        Rvect = Rvect_b17
    print ("# bayestar%s is used."%version)
    l_vals = _ensure_quantity(d['l'], units.deg, units)
    b_vals = _ensure_quantity(d['b'], units.deg, units)
    dist_vals = _ensure_quantity(d['distpc'], units.pc, units)
    coords = SkyCoord(l=l_vals, b=b_vals, distance=dist_vals, frame='galactic')
    reddening = bayestar(coords, mode='median')
    ak = reddening * Rvect[-1]
    d['ak'] = ak
    d['ak_err'] = 0.3 * ak # Fulton & Petigura
    d['kmag_corrected'] = d.kmag - ak
    d['kmag_err_corrected'] = np.sqrt(d.kmag_err**2 + d.ak_err**2)

    return d
