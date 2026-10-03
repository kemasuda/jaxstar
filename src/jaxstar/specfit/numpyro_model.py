"""Optional single-star priors on the ordinary deterministic/continuum APIs.

No optimizer, sampled continuum coefficients, GP, or fitting state lives here.
Users can instead write their own NumPyro model using exactly the same helpers.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .continuum import marginalized_continuum_log_likelihood
from .model import SpecModel, _require
from .observation import Observation


def single_star_params(atmosphere, *, vsini, vmacro, q1, q2, rv, resolving_power):
    """Build one-component physical params, retaining frozen q1/q2 convention.

    This pure JAX helper is also useful in custom NumPyro models and outside
    inference when reconstructing spectra from a sample-site dictionary.
    Atmosphere coordinates are scalar; other values may be scalar or per region.
    """
    q1, q2 = jnp.asarray(q1), jnp.asarray(q2)
    _require(jnp.all(jnp.isfinite(q1) & (q1 >= 0) & (q1 <= 1))
             & jnp.all(jnp.isfinite(q2) & (q2 >= 0) & (q2 <= 1)),
             "q1 and q2 must be finite in [0, 1]")
    u1 = 2 * jnp.sqrt(q1) * q2
    u2 = jnp.sqrt(q1) - u1
    return {"components": ({"atmosphere": dict(atmosphere),
             "broadening": {"vsini": vsini, "vmacro": vmacro, "u1": u1, "u2": u2},
             "rv": rv},), "instrument": {"resolving_power": resolving_power}}


def _draw(name, prior_or_value, *, regions=None):
    if isinstance(prior_or_value, dist.Distribution):
        # Region vectors are one joint site, without implicit plate broadcasting.
        prior = prior_or_value.to_event(len(prior_or_value.batch_shape))
        value = numpyro.sample(name, prior)
    else:
        value = numpyro.deterministic(name, jnp.asarray(prior_or_value))
    value = jnp.asarray(value)
    if value.ndim != 0 and (regions is None or value.shape != (regions,)):
        contract = "scalar" if regions is None else f"scalar or shape ({regions},)"
        raise ValueError(f"{name} must be {contract}, got {value.shape}")
    return value


def model_single(observation, specmodel, *, atmosphere, broadening, rv,
                 resolving_power, sigma_constant, sigma_continuum, jitter=0.0,
                 degree=4, basis=None, save_model_flux=False):
    """Single-star NumPyro model with analytically marginalized continuum.

    Every physical/noise argument is a fixed numeric value or a NumPyro
    distribution. ``atmosphere`` maps exactly the loaded library's axis names
    to scalar values/distributions. ``broadening`` supplies exactly ``vsini``,
    ``vmacro``, ``q1``, ``q2``. These, RV, resolving power, jitter and continuum
    scales can be scalar or (n_region,). No scientific prior ranges or universal
    continuum/jitter prior are chosen here; pass them explicitly. Jitter defaults
    to zero and is additive absolute noise in observation flux units.

    Site names are the atmosphere axis names, vsini/vmacro/q1/q2, rv,
    resolving_power, sigma_constant/sigma_continuum and jitter. Fixed values
    produce deterministic sites. u1/u2 are deterministic transformations.
    ``spectrum`` is a normalized marginalized-likelihood factor. Continuum
    coefficients never become latent sites. The returned physical parameter
    dict is useful when tracing/replaying a fitted point. Large ``model_flux``
    traces are opt-in; reconstruct continuum and data-space spectra separately.

    Rows must already be aligned with the SpecModel library. Optional region
    labels must agree. This is an ordinary NumPyro model, usable with NUTS,
    SVI or user-selected inference tools; numpyro-inferutils is not required.
    """
    if not isinstance(observation, Observation) or not isinstance(specmodel, SpecModel):
        raise TypeError("model_single requires Observation and SpecModel")
    count = observation.n_regions
    if count != len(specmodel.spectra.regions):
        raise ValueError("observation rows must match SpecModel regions")
    if observation.region is not None and observation.region != specmodel.spectra.regions:
        raise ValueError("observation region labels must match SpecModel row order")
    axes = specmodel.spectra.grid.axis_names
    if not isinstance(atmosphere, Mapping) or set(atmosphere) != set(axes):
        raise ValueError(f"atmosphere must supply exactly the named axes {axes}")
    names = ("vsini", "vmacro", "q1", "q2")
    reserved = (*names, "rv", "resolving_power", "sigma_constant", "sigma_continuum",
                "jitter", "u1", "u2", "spectrum", "model_flux")
    if set(axes).intersection(reserved):
        raise ValueError("atmosphere axis names conflict with single-star sample sites")
    if not isinstance(broadening, Mapping) or set(broadening) != set(names):
        raise ValueError("broadening must supply exactly vsini, vmacro, q1, q2")
    coordinates = {name: _draw(name, atmosphere[name]) for name in axes}
    broad = {name: _draw(name, broadening[name], regions=count) for name in names}
    velocity = _draw("rv", rv, regions=count)
    resolution = _draw("resolving_power", resolving_power, regions=count)
    s0 = _draw("sigma_constant", sigma_constant, regions=count)
    sc = _draw("sigma_continuum", sigma_continuum, regions=count)
    noise = _draw("jitter", jitter, regions=count)
    params = single_star_params(coordinates, **broad, rv=velocity, resolving_power=resolution)
    for name in ("u1", "u2"):
        numpyro.deterministic(name, params["components"][0]["broadening"][name])
    flux = specmodel(params, observation.wavelength)
    if save_model_flux:
        numpyro.deterministic("model_flux", flux)
    logp = marginalized_continuum_log_likelihood(
        observation, flux, degree=degree, basis=basis,
        sigma_constant=s0, sigma_continuum=sc, jitter=noise)
    numpyro.factor("spectrum", logp)
    return params
