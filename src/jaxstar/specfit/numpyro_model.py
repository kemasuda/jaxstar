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


def physical_logg_max(teff):
    """Return the legacy Teff-dependent upper bound on logg (log10, cgs).

    ``teff`` is in kelvin. This is the polynomial used by
    ``jaxspec.numpyro_model.model_single(physical_logg_max=True)``, whose
    source notes an applicability range of 4500--7000 K. It is evaluated
    without clipping or range checks; the caller chooses the lower bound
    and ensures that the resulting prior lies inside the spectral grid.
    Scalar and array inputs support JAX transformations.
    """
    teff = jnp.asarray(teff)
    return -2.34638497e-08 * teff**2 + 1.58069918e-04 * teff + 4.53251890


def empirical_vmic(teff, logg, feh):
    """Return the legacy microturbulent velocity in km/s.

    Inputs are Teff in kelvin, logg in log10 cgs, and metallicity ``feh`` in
    dex. The relation is copied from ``jaxspec.numpyro_model.get_empirical_vmic``,
    which notes Teff > 5000 K and logg > 3.5. Its BOSZ caller supplied ``mh``
    as ``feh``; this helper performs no metallicity conversion.

    The polynomial is evaluated without clipping or applicability checks.
    Inputs broadcast under JAX; the caller decides whether the result is
    fixed deterministically or used to construct a prior with scatter.
    """
    dt = jnp.asarray(teff) - 5500
    dg = jnp.asarray(logg) - 4.0
    feh = jnp.asarray(feh)
    return (1.05 + 2.51e-4 * dt + 1.5e-7 * dt**2
            - 0.14 * dg - 0.05e-1 * dg**2 + 0.05 * feh + 0.01 * feh**2)


def empirical_vmacro_valenti_fischer2005(teff, *, sigma=1.0):
    """Return a macroturbulence prior around the Valenti & Fischer (2005) relation.

    ``teff`` is in kelvin; the returned NumPyro distribution is in km/s:
    ``TruncatedNormal(loc=3.98 + (teff - 5770) / 650, scale=sigma, low=0)``.
    The Teff relation is from Valenti & Fischer (2005), ApJS 159, 141,
    doi:10.1086/430500. The default ``sigma=1.0`` and the truncation reproduce
    ``jaxspec.numpyro_model.model_single``; they are prior choices from that
    model, not a scatter measurement attributed to the paper.

    ``sigma`` must be positive and is the standard deviation before truncation.
    Teff and sigma may be scalars or broadcastable arrays. The helper creates
    no sample sites; use ``numpyro.sample("vmacro", helper(teff))`` inside a
    custom model. Sigma is fixed by default, but an already sampled value may
    be passed explicitly. Applicability and spectral coverage are the caller's
    responsibility. Future Teff/logg relations can have separate named helpers.
    """
    teff = jnp.asarray(teff)
    sigma = jnp.asarray(sigma)
    location = 3.98 + (teff - 5770.) / 650.
    return dist.TruncatedNormal(loc=location, scale=sigma, low=0.)


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
    """Record a distribution draw or fixed value, then check its scalar/region shape."""
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


def _validate_single_config(observation, specmodel, atmosphere, broadening):
    """Check model configuration before sampling; return axis names and region count."""
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
    return axes, count


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
    axes, n_regions = _validate_single_config(observation, specmodel, atmosphere, broadening)

    # Scalar atmosphere coordinates, in the loaded library's axis order.
    coordinates = {}
    for name in axes:
        coordinates[name] = _draw(name, atmosphere[name])

    # Broadening and kinematics: shared values or one value per region.
    vsini = _draw("vsini", broadening["vsini"], regions=n_regions)
    vmacro = _draw("vmacro", broadening["vmacro"], regions=n_regions)
    q1 = _draw("q1", broadening["q1"], regions=n_regions)
    q2 = _draw("q2", broadening["q2"], regions=n_regions)
    rv = _draw("rv", rv, regions=n_regions)
    resolving_power = _draw("resolving_power", resolving_power, regions=n_regions)

    # Continuum prior scales and additive noise; coefficients are marginalized.
    sigma_constant = _draw("sigma_constant", sigma_constant, regions=n_regions)
    sigma_continuum = _draw("sigma_continuum", sigma_continuum, regions=n_regions)
    jitter = _draw("jitter", jitter, regions=n_regions)

    # Convert limb darkening and evaluate the physical spectrum.
    params = single_star_params(
        coordinates, vsini=vsini, vmacro=vmacro, q1=q1, q2=q2,
        rv=rv, resolving_power=resolving_power)
    for name in ("u1", "u2"):
        numpyro.deterministic(name, params["components"][0]["broadening"][name])
    flux = specmodel(params, observation.wavelength)
    if save_model_flux:
        numpyro.deterministic("model_flux", flux)

    # Normalized likelihood with the continuum integrated out.
    log_likelihood = marginalized_continuum_log_likelihood(
        observation, flux, degree=degree, basis=basis,
        sigma_constant=sigma_constant, sigma_continuum=sigma_continuum, jitter=jitter)
    numpyro.factor("spectrum", log_likelihood)
    return params
