__all__ = ["MistGridIso", "MistFit"]

#%%
import numpy as np
import pandas as pd
import os
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import (random, jit)
from numpyro.infer import init_to_value
from jax.scipy.ndimage import map_coordinates as mapc
from scipy.optimize import minimize
#from jax.scipy.optimize import minimize
from functools import partial
from .gyrochrone_likelihood import loglike_gyro
from jaxstar.mistfit.mistgrid.create_grid import *


def check_mistgrid_path():
    """ check the existence of mistgrid_iso.npz
    if the file does not exist, download CMD files and create the file by calling create_mistgrid()
    """
    gridfile_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mistgrid/mistgrid_iso.npz')
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mistgrid/create_grid.py')
    if not os.path.exists(gridfile_path):
        print ("mistgrid_iso.npz not found.")
        create_mistgrid()
    return gridfile_path


class MistGridIso:
    """ class to store model grid """
    def __init__(self, path=None):
        """ initialization

            Args:
                path: path to npz grid file, if not specified, defaults to gridfile_path

        """
        if path is None:
            path = check_mistgrid_path()
        self.dgrid = np.load(path)
        self.a0, self.da = self.dgrid['logagrid'][0], np.diff(self.dgrid['logagrid'])[0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[0]
        self.eep0, self.deep = self.dgrid['eepgrid'][0], np.diff(self.dgrid['eepgrid'])[0]
        self.amin, self.amax = np.min(self.dgrid['logagrid']), np.max(self.dgrid['logagrid'])
        self.fmin, self.fmax = np.min(self.dgrid['fgrid']), np.max(self.dgrid['fgrid'])
        self.eepmin, self.eepmax = np.min(self.dgrid['eepgrid']), np.max(self.dgrid['eepgrid'])
        self.keys = None

    def set_keys(self, keys):
        """ set keys for the stellar parameters to be evaluated

            Args:
                keys: list of strings, should be one of those listed in mistgrid/create_grid.py

        """
        self.keys = keys

    @partial(jit, static_argnums=(0,))
    def values(self, age, feh, eep):
        """ compute stellar parameters for given age, feh, and eep

            Args:
                age: log10(stellar age in yr)
                feh: metallicity (dex)
                eep: equivalent evolutionary point

            Returns:
                interpolated values for the parameters in self.keys

        """
        aidx = (age - self.a0) / self.da
        fidx = (feh - self.f0) / self.df
        eepidx = (eep - self.eep0) / self.deep
        idxs = [aidx, fidx, eepidx]
        return [mapc(self.dgrid[key], idxs, order=1, cval=-jnp.inf) for key in self.keys]

    def values2(self, age, feh, mass, keys):
        """ compute stellar parameters for given age, feh, and mass

            Args:
                age:  log10(stellar age in yr)
                feh:  metallicity (dex)
                mass: stellar mass (Msun)

            Returns:
                interpolated values for the parameters in self.keys

        """
        eep_new = self.find_eep(age, feh, mass)
        self.set_keys(keys)
        return self.values(age,feh,eep_new)[0]

    def _mass_res(self, eep, age, feh, mass):
        self.set_keys(["mass"])
        mass_guess = self.values(age, feh, eep)
        return (mass - np.array(mass_guess)) ** 2
 
    def find_eep(self, age, feh, mass):
        """ find eep value given age, feh, and mass

            Args:
                age:  log10(stellar age in yr)
                feh:  metallicity (dex)
                mass: stellar mass (Msun)

            Returns:
                eep value given age, feh, and mass

        """
        eep0=300
        resid_tol=0.001
        aidx   = (age - self.a0) / self.da
        fidx   = (feh - self.f0) / self.df
        result = minimize(self._mass_res, eep0, args=(age, feh, mass), method="Nelder-Mead",tol=resid_tol)
        if result.success and result.fun < resid_tol:
            return float(result.x)
        else:
            return np.nan

@jit
def smbound(x, low, upp, s=20, depth=30):
    """ sigmoid bound

        Args:
            x: parameter to be bounded
            low, upp: lower and upper bounds
            s: smoothness of the bound
            depth: depth of the bounds

        Returns:
            box-shaped penality term for the log-likelihood
            const if low < x < upp, const+depth otherwise

    """
    return -depth*(1./(jnp.exp(s*(x-low))+1)+1./(jnp.exp(-s*(x-upp))+1))


class MistFit:
    """ main class for isochrone fitting """
    def __init__(self, path=None):
        """ initialization

            Args:
                path: path to npz grid file, if not specified, defaults to gridfile_path

        """
        if path is None:
            path = check_mistgrid_path()
        self.mg = MistGridIso(path)

    def set_data(self, keys, vals, errs):
        """ set observational data to be fitted

            Args:
                keys: names of the parameters (list of str)
                vals: observed values of the parameters
                errs: errors of the parameters (assumed to be Gaussian widths)

        """
        self.obskeys = keys
        self.obsvals = vals
        self.obserrs = errs
        outkeys = ['kmag', 'teff', 'logg', 'mass', 'radius', 'dmdeep', 'mmin', 'mmax', 'bpmag', 'rpmag']
        #outkeys = ['eepmin', 'eepmax', 'kmag', 'teff', 'logg', 'mass', 'radius', 'dmdeep', 'mmin', 'mmax']
        for k in keys:
            if k not in outkeys and k!='parallax' and k!='feh':
                outkeys += [k]
        self.outkeys = outkeys
        self.mg.set_keys(outkeys)

    def add_keys(self, keys):
        """ set keys for the stellar parameters to be evaluated

            Args:
                keys: keys to be added manually (list of str)
                    note that 'outkeys' defined in set_data are automatically added

        """
        outkeys = self.outkeys
        for k in keys:
            if k not in outkeys:
                outkeys += [k]
        self.outkeys = outkeys
        self.mg.set_keys(outkeys)

    def model(self, nodata=False, linear_age=True, flat_age_marginal=False, logamin=8, logamax=10.14,
            fmin=-1, fmax=0.5, eepmin=0, eepmax=600, massmin=0.1, massmax=2.5, dist_scale=1.35, prot=None, prot_err=0.05, rho=None, rho_err=None):
        """ model for NumPyro HMC

            Args:
                nodata: if True, data is ignored (i.e. sampling from prior)
                linear_age: if True/False, prior flat in age/logage is used
                flag_age_marginal: if True, the prior is set so that the marginal age/logage prior is flat.
                    Otherwise, PDF has a constant value in the mass-(log)age-eep space.
                logamin, logmax: bounds for log10(age/yr)
                fmin, fmax: bounds for FeH
                eepmin, eepmax: bounds for EEP
                massmin, massmax: bounds for stellar mass
                dist_scale: length scale L in the distance prior (Bailer-Jones 2015; Astraatmadja & Bailer-Jones 2016)
                prot: if specified, gyrochronal log-likelihood is added folloing Angus et al. (2019), AJ 158, 173
                prot_err: error in Prot assumed in evaluating gyro log-likelihood
                rho, rho_err: if specified, gaussian prior on the mean density can be imposed (solar units)

        """
        if linear_age:
            age = numpyro.sample("age", dist.Uniform(10**logamin/1e9, 10**logamax/1e9))
            logage = jnp.log10(age * 1e9)
            numpyro.deterministic("logage", logage)
        else:
            logage= numpyro.sample("logage", dist.Uniform(logamin, logamax))
            numpyro.deterministic("age", 10**logage/1e9)

        feh = numpyro.sample("feh", dist.Uniform(fmin, fmax))
        eep = numpyro.sample("eep", dist.Uniform(eepmin, eepmax))

        distance = numpyro.sample("distance", dist.Gamma(3, rate=1./dist_scale)) # BJ18, kpc
        parallax = numpyro.deterministic("parallax", 1. / distance) # mas

        params = dict(zip(self.outkeys, self.mg.values(logage, feh, eep)))
        for key in self.outkeys:
            if 'mag' in key:
                params[key] = params[key] - 5 * jnp.log10(parallax) + 10 # apparent mag
            params[key] = jnp.where(params[key]==params[key], params[key], -jnp.inf) # nan
            numpyro.deterministic(key, params[key])

        if not nodata:
            params['parallax'], params['feh'] = parallax, feh
            obsparams = jnp.array([params[key] for key in self.obskeys])
            numpyro.sample("obs", dist.Normal(obsparams, jnp.array(self.obserrs)), obs=jnp.array(self.obsvals))

        # mass prior
        logjac = jnp.log(params['dmdeep'])
        logjac += smbound(params['mass'], massmin, massmax)
        if flat_age_marginal:
            params['mmax'] = jnp.where(params['mmax'] < massmax, params['mmax'], massmax)
            logjac -= jnp.log(params['mmax'] - params['mmin'])

        logjac = jnp.where(logjac==logjac, logjac, -jnp.inf)
        numpyro.factor("logjac", logjac)

        if prot is not None:
            bprp = params['bpmag'] - params['rpmag']
            numpyro.factor("loglike_gyro", loglike_gyro(prot, bprp, params['mass'], eep, logage, feh, sigma=prot_err))

        if rho is not None and rho_err is not None:
            rho_model = params['mass'] / params['radius']**3
            numpyro.factor("loglike_rho", -0.5 * (rho - rho_model)**2 / rho_err**2)

    def setup_hmc(self, target_accept_prob=0.95, num_warmup=1000, num_samples=1000, init_logage=9.3, init_feh=0, init_eep=300):
        """ setup NumPyro HMC

            Args:
                target_accept_prob: target acceptance probability in HMC/NUTS
                num_warmup: # of warmup steps
                num_samples: # of sampling steps
                init_logage: initival value of log10(age/yr)
                init_feh: initial value of FeH
                init_eep: initial value of EEP

        """
        # initialize parameters for HMC
        _parallax_obs = np.array(self.obsvals)[np.array(self.obskeys)=='parallax']
        init_dist = float(np.where(_parallax_obs > 0, 1./_parallax_obs, 8))
        initdict = {"distance": init_dist}
        for key, val in zip(self.obskeys, self.obsvals):
            if key=="parallax":
                continue
            initdict[key] = val
        init_strategy = init_to_value(values=initdict)

        kernel = numpyro.infer.NUTS(self.model, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        self.mcmc = mcmc

    def run_hmc(self, rng_key, **kwargs):
        """ run HMC/NUTS

            Args:
                rng_key: PRNG key from jax.random.PRNGKey()

        """
        self.mcmc.run(rng_key, **kwargs)
        self.mcmc.print_summary()
        self.samples = pd.DataFrame(data=self.mcmc.get_samples())
