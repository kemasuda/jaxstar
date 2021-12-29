__all__ = ["correct_gedr3_parallax", "correct_kmag"]

#%%
import numpy as np
import pandas as pd

#%% for Gaia EDR3
from zero_point import zpt

# eq.16 from El-Badry et al. (2021) MNRAS 506, 2269
def edr3error(gmags, a=0.21, g0=12.65, b=0.90, p=[-0.00062, 0.0040, 1.141]):
    return a * np.exp(-(gmags-g0)**2 / b**2) + np.poly1d(p)(gmags)

#%%
def correct_gedr3_parallax(d):
    zpt.load_tables()
    d["zpt"] = d.apply(zpt.zpt_wrapper, axis=1)
    d["parallax_zpcorrected"] = d.parallax - d.zpt
    print ("# zp correction failed for %s stars (original parallax used)"%np.sum(d.parallax_zpcorrected != d.parallax_zpcorrected))
    d['parallax_corrected'] = d['parallax_zpcorrected'].fillna(d.parallax)
    d['parallax_error_corrected'] = d.parallax_error * edr3error(d.phot_g_mean_mag)
    return d

#%%
from dustmaps.bayestar import BayestarQuery as bquery
from astropy.coordinates import SkyCoord
import astropy.units as units
# extinction vector from Table 1 of Green et al. (2019) https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G/abstract
# Pan-STARRS 1 grizy, 2MASS JHKs
Rvect_b19 = np.array([3.518, 2.617, 1.971, 1.549, 1.263, 0.7927, 0.4690, 0.3026])
Rvect_b17 = np.array([3.384, 2.483, 1.838, 1.414, 1.126, 0.650, 0.327, 0.161])

#%%
def correct_kmag(d, version='2019'):
    print ("# %d stars include nan."%np.sum((d.kmag_err!=d.kmag_err)&(d.parallax!=d.parallax)))

    d['distpc'] = 1e3 / d.parallax
    d.distpc[d['distpc']<0] = 8e3
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
    coords = SkyCoord(l=d['l']*units.deg, b=d['b']*units.deg, distance=d['distpc']*units.pc, frame='galactic')
    reddening = bayestar(coords, mode='median')
    ak = reddening * Rvect[-1]
    d['ak'] = ak
    d['ak_err'] = 0.3 * ak # Fulton & Petigura
    d['kmag_corrected'] = d.kmag - ak
    d['kmag_err_corrected'] = np.sqrt(d.kmag_err**2 + d.ak_err**2)

    return d

#%%
"""
filename = "/Users/k_masuda/Dropbox/astrodata/gyrochrone/cks/all_edr3_binflag.csv"
d = pd.read_csv(filename)
d = correct_gedr3_parallax(d)
d = correct_kmag(d)

#%%
d2 = pd.read_csv(filename)
d2 = correct_gedr3_parallax(d2)
d2 = correct_kmag(d2, version='2017')
#d.to_csv("isoinput_cks.csv", index=False)

#%%
import matplotlib.pyplot as plt
plt.xscale("log")
plt.plot(d.distpc, d.ak, '.', label='b2019')
plt.plot(d2.distpc, d2.ak, '.', label='b2017')
plt.legend()
"""
