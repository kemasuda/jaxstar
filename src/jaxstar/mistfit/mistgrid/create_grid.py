
__all__ = ["create_mistgrid"]

#%%
import numpy as np
import pandas as pd
import sys, os, glob, pathlib
from astropy.constants import M_sun, R_sun, G
from scipy.interpolate import interp1d
logg_sun = np.log10((G * M_sun / R_sun**2).cgs.value)

#%%
keys = ['2MASS_J', '2MASS_H', '2MASS_Ks', 'logT', 'logg', 'teff', 'logage', 'mass', 'dmdeep', 'logL', 'radius', 'mmin', 'mmax', 'eepmin', 'eepmax']
keys += ['Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev']
keys += ['Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3']
#keys = list(df_all.keys())
#print (keys)
#print (df_all.keys())

#%%
def create_mistgrid():
    """ function to create mistgrid_iso.npz for jaxstar.mistfit
    """
    url_cmd = "http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_UBVRIplus.txz"
    filename_cmd = "MIST_v1.2_vvcrit0.4_UBVRIplus.txz"
    mistgriddir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    mistgrid_path = mistgriddir_path/'mistgrid_iso.npz'
    mistdir_path = mistgriddir_path/'MIST_v1.2_vvcrit0.4_UBVRIplus'
    print (mistdir_path)

    #%%
    if not os.path.exists(mistdir_path):
        print ("CMD files not found. Downloading to %s..."%str(mistgriddir_path))
        import urllib.error
        import urllib.request
        import tarfile
        cmdfile_tar = str(mistgriddir_path/filename_cmd)
        cmdfile_tar
        try:
            with urllib.request.urlopen(url_cmd) as download_file:
                data = download_file.read()
                with open(cmdfile_tar, mode='wb') as save_file:
                    save_file.write(data)
        except urllib.error.URLError as errormsg:
            print (errormsg)

        import tarfile
        with tarfile.open(cmdfile_tar, 'r:xz') as t:
            t.extractall(path=mistgriddir_path)

    if os.path.exists(mistdir_path):
        print ("CMD files found in %s"%str(mistdir_path))
        filenames = glob.glob(str(mistdir_path/'*.cmd'))
        print (filenames)
        print ("creating grid for mistfit...")

    #%%
    header = ['EEP', 'log10_isochrone_age_yr', 'initial_mass', 'star_mass', 'log_Teff', 'log_g', 'log_L', '[Fe/H]_init', '[Fe/H]',  'Bessell_U',  'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I', '2MASS_J', '2MASS_H', '2MASS_Ks', 'Kepler_Kp', 'Kepler_D51', 'Hipparcos_Hp', 'Tycho_B', 'Tycho_V', 'Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev', 'Gaia_G_MAW', 'Gaia_BP_MAWb', 'Gaia_BP_MAWf', 'Gaia_RP_MAW', 'TESS', 'Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3', 'phase']

    #%%
    df_all = pd.DataFrame(data={})
    for filename in filenames:
        feh = float(filename.split('/')[-1].split("_")[3].replace("m", "-").replace("p", "+"))
        d = pd.read_csv(filename, delim_whitespace=True, comment='#', header=None, names=header)
        d['mass'] = d['initial_mass']
        d['feh'] = d['[Fe/H]_init']
        d['teff'] = 10 ** d['log_Teff']
        d['radius'] = np.sqrt(d['star_mass'] / 10 ** (d['log_g'] - logg_sun))
        d['logage'] = d['log10_isochrone_age_yr']
        d['age'] = 10**(d.logage) / 1e9
        df_all = df_all.append(d)
    df_all = df_all.reset_index(drop=True)

    #%% Fe/H grid cut at -1.5
    d = df_all.sort_values(["logage", "feh", "EEP"]).reset_index(drop=True)
    agrid = np.sort(list(set(d.logage)))
    fgrid = np.sort(list(set(d.feh)))[6:]
    eepgrid = np.sort(list(set(d.EEP)))
    print (agrid, len(agrid))
    print (fgrid, len(fgrid))
    print (eepgrid, len(eepgrid))

    #%%
    def eepderivative(y):
        dy = 0.5 * (y[2:] - y[:-2])
        return np.array([dy[0]]+list(dy)+[dy[-1]])

    #%%
    d = d.rename({"log_Teff": 'logT', "log_L": "logL", "log_g": "logg"}, axis='columns')

    #%%
    pgrids2d = []
    for key in keys:
        pgrid2d = np.zeros((len(agrid), len(fgrid), len(eepgrid)))
        for i,a in enumerate(agrid):
            for j,f in enumerate(fgrid):
                _d = d[(d.logage==a)&(d.feh==f)]
                eeparr = np.ones_like(eepgrid) * -np.inf
                eep0, eep1 = int(_d.EEP.min()), int(_d.EEP.max())+1
                if key=='dmdeep':
                    _marr = interp1d(_d.EEP, _d['mass'])(np.arange(eep0, eep1))
                    eeparr[eep0:eep1] = eepderivative(_marr)
                elif key=='mmin':
                    eeparr = _d['mass'].min()
                elif key=='mmax':
                    eeparr = _d['mass'].max()
                elif key=='eepmin':
                    eeparr = eep0
                elif key=='eepmax':
                    eeparr = eep1 - 1
                else:
                    eeparr[eep0:eep1] = interp1d(_d.EEP, _d[key])(np.arange(eep0, eep1))
                pgrid2d[i][j] = eeparr
                #pgrid2d.append(eeparr.reshape(len(mgrid), len(fgrid)))
        #pgrid2d = np.array(pgrid2d)
        pgrids2d.append(pgrid2d)

    #%%
    np.savez(mistgrid_path, logagrid=agrid, fgrid=fgrid, eepgrid=eepgrid,
            jmag=pgrids2d[0], hmag=pgrids2d[1], kmag=pgrids2d[2],
            logt=pgrids2d[3], logg=pgrids2d[4], teff=pgrids2d[5], logage=pgrids2d[6], mass=pgrids2d[7],
            dmdeep=pgrids2d[8], logl=pgrids2d[9], radius=pgrids2d[10], mmin=pgrids2d[11], mmax=pgrids2d[12], eepmin=pgrids2d[13], eepmax=pgrids2d[14], gmag=pgrids2d[15], bpmag=pgrids2d[16], rpmag=pgrids2d[17],
            gmag3=pgrids2d[18], bpmag3=pgrids2d[19], rpmag3=pgrids2d[20])
