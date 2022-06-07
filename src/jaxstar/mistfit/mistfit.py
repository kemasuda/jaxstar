__all__ = ["MistGridIso", "MistFit"]

#%%
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import (random, jit)
from numpyro.infer import init_to_value
from jax.scipy.ndimage import map_coordinates as mapc
from functools import partial
from .gyrochrone_likelihood import loglike_gyro

#%% here age is logage
class MistGridIso:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mistgrid/mistgrid_iso.npz')
        self.dgrid = np.load(path)
        self.a0, self.da = self.dgrid['logagrid'][0], np.diff(self.dgrid['logagrid'])[0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[0]
        self.eep0, self.deep = self.dgrid['eepgrid'][0], np.diff(self.dgrid['eepgrid'])[0]
        self.amin, self.amax = np.min(self.dgrid['logagrid']), np.max(self.dgrid['logagrid'])
        self.fmin, self.fmax = np.min(self.dgrid['fgrid']), np.max(self.dgrid['fgrid'])
        self.eepmin, self.eepmax = np.min(self.dgrid['eepgrid']), np.max(self.dgrid['eepgrid'])
        self.keys = None

    def set_keys(self, keys):
        self.keys = keys

    @partial(jit, static_argnums=(0,))
    def values(self, age, feh, eep):
        aidx = (age - self.a0) / self.da
        fidx = (feh - self.f0) / self.df
        eepidx = (eep - self.eep0) / self.deep
        idxs = [aidx, fidx, eepidx]
        return [mapc(self.dgrid[key], idxs, order=1, cval=-jnp.inf) for key in self.keys]

#%%
@jit
def smbound(x, low, upp, s=20, depth=30):
    return -depth*(1./(jnp.exp(s*(x-low))+1)+1./(jnp.exp(-s*(x-upp))+1))

#%%
import os
class MistFit:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mistgrid/mistgrid_iso.npz')
        self.mg = MistGridIso(path)

    def set_data(self, keys, vals, errs):
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
        outkeys = self.outkeys
        for k in keys:
            if k not in outkeys:
                outkeys += [k]
        self.outkeys = outkeys
        self.mg.set_keys(outkeys)

    def model(self, nodata=False, linear_age=True, flat_age_marginal=False, logamin=8, logamax=10.14,
            fmin=-1, fmax=0.5, eepmin=0, eepmax=600, massmin=0.1, massmax=2.5, dist_scale=1.35, prot=None, prot_err=0.05):
        if linear_age:
            age = numpyro.sample("age", dist.Uniform(10**logamin/1e9, 10**logamax/1e9))
            logage = jnp.log10(age * 1e9)
            numpyro.deterministic("logage", logage)
        else:
            logage= numpyro.sample("logage", dist.Uniform(logamin, logamax))
            numpyro.deterministic("age", 10**logage/1e9)

        feh = numpyro.sample("feh", dist.Uniform(fmin, fmax))
        #tmp = self.mg.values(logage, feh, 0)
        #eep = numpyro.sample("eep", dist.Uniform(tmp[0], tmp[1]))
        eep = numpyro.sample("eep", dist.Uniform(eepmin, eepmax))

        #distance = 10**numpyro.sample("distance", dist.Uniform(-2, 1)) # kpc
        distance = numpyro.sample("distance", dist.Gamma(3, rate=1./dist_scale)) # BJ18, kpc
        #parallax = 1. / distance # mas
        #numpyro.deterministic("parallax", parallax)
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

    def setup_hmc(self, target_accept_prob=0.95, num_warmup=1000, num_samples=1000, init_logage=9.3, init_feh=0, init_eep=300):
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
        self.mcmc.run(rng_key, **kwargs)
        self.mcmc.print_summary()
        self.samples = pd.DataFrame(data=self.mcmc.get_samples())
