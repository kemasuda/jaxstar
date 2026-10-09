"""Frozen pre-refactor MistFit likelihood/prior oracle, from 443352e69c56b88708bbd1376002bfe73705b489.

Only imports and the class name differ from the original. Never update this
reference to match a new implementation; tests use tiny local NPZ inputs.
"""

import jax.numpy as jnp
from jax import jit
import numpyro
import numpyro.distributions as dist

from _mistgrid_reference import MistGridIso, NamedMistGridIso, check_mistgrid_path
from jaxstar.mistfit.gyrochrone_likelihood import loglike_gyro

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


class ReferenceMistFit:
    """ main class for isochrone fitting """

    def __init__(self, path=None, *, grid_backend="legacy"):
        """initialization

            Args:
                path: path to an existing npz grid file. If omitted, use the
                    environment override or user cache, downloading on first use.
                grid_backend: "legacy" (default) uses MistGridIso; "named" uses
                    NamedMistGridIso. Choose the backend at initialization.

        """
        grid_types = {"legacy": MistGridIso, "named": NamedMistGridIso}
        if grid_backend not in grid_types:
            raise ValueError("grid_backend must be 'legacy' or 'named'")
        self.grid_backend = grid_backend
        if path is None:
            path = check_mistgrid_path()
        self.mg = grid_types[grid_backend](path)

    def set_data(self, keys, vals, errs):
        """set observational data to be fitted

            Args:
                keys: names of the parameters (list of str)
                vals: observed values of the parameters
                errs: errors of the parameters (assumed to be Gaussian widths)

        """
        self.obskeys = keys
        self.obsvals = vals
        self.obserrs = errs
        outkeys = ['kmag', 'teff', 'logg', 'mass', 'radius', 'feh_photosphere', 'star_mass',
                   'dmdeep', 'mmin', 'mmax', 'bpmag2', 'rpmag2']  # DR2 BP-RP needed for gyro likelihood
        # outkeys = ['eepmin', 'eepmax', 'kmag', 'teff', 'logg', 'mass', 'radius', 'dmdeep', 'mmin', 'mmax']
        for k in keys:
            if k not in outkeys and k != 'parallax' and k != 'feh':
                outkeys += [k]
        self.outkeys = outkeys
        if self.grid_backend == "legacy":
            self.mg.set_keys(outkeys)

    def add_keys(self, keys):
        """set keys for the stellar parameters to be evaluated

            Args:
                keys: keys to be added manually (list of str)
                    note that 'outkeys' defined in set_data are automatically added

        """
        outkeys = self.outkeys
        for k in keys:
            if k not in outkeys:
                outkeys += [k]
        self.outkeys = outkeys
        if self.grid_backend == "legacy":
            self.mg.set_keys(outkeys)

    def model(self, nodata=False, linear_age=True, flat_age_marginal=False, logamin=8, logamax=10.14,
              fmin=-1, fmax=0.5, eepmin=0, eepmax=600, massmin=0.1, massmax=2.5, dist_scale=1.35, prot=None, prot_err=0.05, rho_obs=None, rho_err=None, age_obs=None, age_err=None):
        """model for NumPyro HMC

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
            age = numpyro.sample("age", dist.Uniform(
                10**logamin/1e9, 10**logamax/1e9))
            # logage = jnp.log10(age * 1e9)
            # numpyro.deterministic("logage", logage)
            logage = numpyro.deterministic("logage", jnp.log10(age * 1e9))
        else:
            logage = numpyro.sample("logage", dist.Uniform(logamin, logamax))
            age = numpyro.deterministic("age", 10**logage/1e9)

        feh_init = numpyro.sample("feh_init", dist.Uniform(fmin, fmax))
        eep = numpyro.sample("eep", dist.Uniform(eepmin, eepmax))

        distance = numpyro.sample("distance", dist.Gamma(
            3, rate=1./dist_scale))  # BJ18, kpc
        parallax = numpyro.deterministic("parallax", 1. / distance)  # mas

        if self.grid_backend == "named":
            params = dict(self.mg.values(logage, feh_init, eep, keys=self.outkeys))
        else:
            params = dict(zip(self.outkeys, self.mg.values(logage, feh_init, eep)))
        for key in self.outkeys:
            if 'mag' in key:
                params[key] = params[key] - 5 * \
                    jnp.log10(parallax) + 10  # apparent mag
            params[key] = jnp.where(
                params[key] == params[key], params[key], -jnp.inf)  # nan
            numpyro.deterministic(key, params[key])
        # "feh" is set to be photospheric value
        params["feh"] = numpyro.deterministic("feh", params["feh_photosphere"])

        if not nodata:
            # params['parallax'], params['feh'] = parallax, feh
            params['parallax'] = parallax
            obsparams = jnp.array([params[key] for key in self.obskeys])
            numpyro.sample("obs", dist.Normal(obsparams, jnp.array(
                self.obserrs)), obs=jnp.array(self.obsvals))

        # mass prior
        logjac = jnp.log(params['dmdeep'])
        logjac += smbound(params['mass'], massmin, massmax)
        if flat_age_marginal:
            params['mmax'] = jnp.where(
                params['mmax'] < massmax, params['mmax'], massmax)
            logjac -= jnp.log(params['mmax'] - params['mmin'])

        logjac = jnp.where(logjac == logjac, logjac, -jnp.inf)
        numpyro.factor("logjac", logjac)

        if prot is not None:
            bprp = params['bpmag2'] - params['rpmag2']
            numpyro.factor("loglike_gyro", loglike_gyro(
                prot, bprp, params['mass'], eep, logage, params['feh'], sigma=prot_err))

        if rho_obs is not None and rho_err is not None:
            rho_model = params['mass'] / params['radius']**3
            numpyro.factor("loglike_rho", -0.5 *
                           (rho_obs - rho_model)**2 / rho_err**2)

        if age_obs is not None and age_err is not None:
            age_model = age
            numpyro.factor("loglike_age", -0.5 *
                           (age_obs - age_model)**2 / age_err**2)
