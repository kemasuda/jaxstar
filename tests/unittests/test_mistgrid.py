import jaxstar
import numpy as np
import pandas as pd


def test_values():
    # file_test, logage_test, feh_test = "MIST_iso_67a746fd71c02.iso.cmd", 9.3, 0.02
    # file_test, logage_test, feh_test = "MIST_iso_67a74f472815f.iso.cmd", 9.3, 0.0
    # file_test, logage_test, feh_test = "MIST_iso_67a74fd255b6e.iso.cmd", 9.3, 0.05
    # file_test, logage_test, feh_test = "MIST_iso_67a751c050e0e.iso.cmd", 9.3, 0.1
    # d = pd.read_csv(file_test, comment='#', sep='\s+')
    # d['teff'] = 10**d['log_Teff']
    # di = d.iloc[300:700]
    # di.to_csv("test_data.txt", index=False)
    di = pd.read_csv("test_data.txt").iloc[300]
    logage_test, feh_test, eep_test = 9.3, 0.1, di.EEP

    mf = jaxstar.mistfit.MistFit()
    keys = ['kmag', 'teff', 'logg', 'mass', 'radius', 'star_mass', 'feh_photosphere',
            'dmdeep', 'mmin', 'mmax', 'bpmag', 'rpmag']
    mf.mg.set_keys(keys)

    kmag, teff, logg, mass, _, star_mass, feh_photosphere, _, _, _, _, _ = mf.mg.values(
        logage_test, feh_test, eep_test)
    output = np.array([kmag, teff, logg, mass, star_mass])
    print(output, feh_photosphere)
    print(di[['2MASS_Ks', 'teff', 'log_g', 'initial_mass', 'star_mass', '[Fe/H]']])
    assert np.allclose(output, np.array(
        di[['2MASS_Ks', 'teff', 'log_g', 'initial_mass', 'star_mass']]), rtol=1e-3, atol=0)
    assert np.isclose(feh_photosphere, di['[Fe/H]'], rtol=0.15)


if __name__ == '__main__':
    test_values()
