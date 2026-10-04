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


def _validate_single_config(observation, specmodel, atmosphere, broadening, *,
                            use_physical_logg_max, use_empirical_vmic,
                            use_empirical_vmacro, vmacro_empirical_sigma):
    """Check configuration and resolve BOSZ-auto vmic before sampling."""
    if not isinstance(observation, Observation) or not isinstance(specmodel, SpecModel):
        raise TypeError("model_single requires Observation and SpecModel")
    count = observation.n_regions
    if count != len(specmodel.spectra.regions):
        raise ValueError("observation rows must match SpecModel regions")
    if observation.region is not None and observation.region != specmodel.spectra.regions:
        raise ValueError("observation region labels must match SpecModel row order")

    for name, value in (("use_physical_logg_max", use_physical_logg_max),
                        ("use_empirical_vmacro", use_empirical_vmacro)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    if use_empirical_vmic is not None and not isinstance(use_empirical_vmic, bool):
        raise TypeError("use_empirical_vmic must be None or a bool")

    axes = specmodel.spectra.grid.axis_names
    empirical_vmic_active = (specmodel.spectra.library == "bosz"
                             if use_empirical_vmic is None else use_empirical_vmic)
    if use_physical_logg_max and not {"teff", "logg"}.issubset(axes):
        raise ValueError("use_physical_logg_max requires teff and logg atmosphere axes")
    if empirical_vmic_active and not {"teff", "logg", "mh", "vmic"}.issubset(axes):
        raise ValueError("empirical vmic requires teff, logg, mh and vmic atmosphere axes")
    if use_empirical_vmacro and "teff" not in axes:
        raise ValueError("use_empirical_vmacro requires a teff atmosphere axis")

    required_axes = tuple(name for name in axes if not (empirical_vmic_active and name == "vmic"))
    if isinstance(atmosphere, Mapping) and empirical_vmic_active and "vmic" in atmosphere:
        raise ValueError("omit atmosphere['vmic'] when empirical vmic is active; "
                         "set use_empirical_vmic=False to supply an explicit vmic")
    if not isinstance(atmosphere, Mapping) or set(atmosphere) != set(required_axes):
        raise ValueError(f"atmosphere must supply exactly the named axes {required_axes}")
    if use_physical_logg_max and isinstance(atmosphere["logg"], dist.Distribution):
        prior = atmosphere["logg"]
        if not isinstance(prior, dist.Uniform):
            raise ValueError("use_physical_logg_max supports only a scalar Uniform prior "
                             "or fixed scalar logg")
        if prior.batch_shape or prior.event_shape:
            raise ValueError("logg must be scalar")

    names = ("vsini", "vmacro", "q1", "q2")
    reserved = (*names, "rv", "resolving_power", "sigma_constant", "sigma_continuum",
                "jitter", "u1", "u2", "spectrum", "model_flux")
    if set(axes).intersection(reserved):
        raise ValueError("atmosphere axis names conflict with single-star sample sites")
    required_broadening = tuple(name for name in names if not (use_empirical_vmacro and name == "vmacro"))
    if isinstance(broadening, Mapping) and use_empirical_vmacro and "vmacro" in broadening:
        raise ValueError("omit broadening['vmacro'] when use_empirical_vmacro=True")
    if not isinstance(broadening, Mapping) or set(broadening) != set(required_broadening):
        raise ValueError(f"broadening must supply exactly {', '.join(required_broadening)}")
    if use_empirical_vmacro:
        sigma = jnp.asarray(vmacro_empirical_sigma)
        if sigma.ndim != 0 or sigma.dtype.kind not in "fiu":
            raise ValueError("vmacro_empirical_sigma must be a finite positive scalar")
        _require(jnp.isfinite(sigma) & (sigma > 0),
                 "vmacro_empirical_sigma must be a finite positive scalar")
    return axes, count, empirical_vmic_active


def model_single(observation, specmodel, *, atmosphere, broadening, rv,
                 resolving_power, sigma_constant, sigma_continuum, jitter=0.0,
                 use_physical_logg_max=False, use_empirical_vmic=None,
                 use_empirical_vmacro=False, vmacro_empirical_sigma=1.0,
                 degree=4, basis=None, save_model_flux=False):
    """Single-star NumPyro model with analytically marginalized continuum.

    Every physical/noise argument is a fixed numeric value or a NumPyro
    distribution. ``atmosphere`` maps the loaded library's axis names to scalar
    values/distributions, omitting ``vmic`` when empirical vmic is active.
    ``broadening`` supplies exactly ``vsini``, ``vmacro``, ``q1``, ``q2``,
    omitting ``vmacro`` when ``use_empirical_vmacro=True``. These, RV,
    resolving power, jitter and continuum
    scales can be scalar or (n_region,). No scientific prior ranges or universal
    continuum/jitter prior are chosen here; pass them explicitly. Jitter defaults
    to zero and is additive absolute noise in observation flux units.

    ``use_physical_logg_max=True`` replaces the upper bound of a scalar Uniform
    logg prior with ``physical_logg_max(teff)``, retaining its lower bound.
    The supplied Uniform upper bound is ignored, as in jaxspec. Fixed scalar
    logg is also allowed if finite and at or below that ceiling. Other logg
    distributions are unsupported in this mode; logg keeps its native site name.

    ``use_empirical_vmic=None`` enables the deterministic relation automatically
    for ``specmodel.spectra.library == "bosz"``. True requests it explicitly
    for any library with teff/logg/mh/vmic axes; False restores explicit vmic
    handling. The helper receives mh as its feh argument, as in jaxspec.
    Explicit vmic conflicts with either active empirical mode, including auto.

    ``use_empirical_vmacro=True`` samples one shared scalar vmacro from the
    Valenti & Fischer (2005) prior with positive scalar
    ``vmacro_empirical_sigma`` (default 1 km/s) and lower bound zero. Explicit
    vmacro conflicts with this option. Empirical relations are not clipped to
    calibration or grid bounds; the caller must choose compatible prior support.

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
    axes, n_regions, empirical_vmic_active = _validate_single_config(
        observation, specmodel, atmosphere, broadening,
        use_physical_logg_max=use_physical_logg_max, use_empirical_vmic=use_empirical_vmic,
        use_empirical_vmacro=use_empirical_vmacro, vmacro_empirical_sigma=vmacro_empirical_sigma)

    # Resolve stellar coordinates in dependency order: Teff -> logg -> mh/vmic.
    coordinates = {}
    if "teff" in axes:
        teff = _draw("teff", atmosphere["teff"])
        coordinates["teff"] = teff

    if "logg" in axes:
        logg_prior = atmosphere["logg"]
        if use_physical_logg_max:
            upper = physical_logg_max(teff)
            if isinstance(logg_prior, dist.Uniform):
                lower = logg_prior.low
                _require(jnp.isfinite(lower) & jnp.isfinite(upper) & (lower < upper),
                         "logg lower bound must be finite and below physical_logg_max(teff)")
                logg = _draw("logg", dist.Uniform(lower, upper))
            else:
                logg = _draw("logg", logg_prior)
                _require(jnp.isfinite(logg) & jnp.isfinite(upper) & (logg <= upper),
                         "fixed logg must be finite and at or below physical_logg_max(teff)")
        else:
            logg = _draw("logg", logg_prior)
        coordinates["logg"] = logg

    if empirical_vmic_active:
        mh = _draw("mh", atmosphere["mh"])
        coordinates["mh"] = mh
        vmic = numpyro.deterministic("vmic", empirical_vmic(teff, logg, mh))
        coordinates["vmic"] = vmic

    # Explicit vmic and all remaining coordinates use their supplied priors/values.
    for name in axes:
        if name not in coordinates:
            coordinates[name] = _draw(name, atmosphere[name])

    # Broadening and kinematics: shared values or one value per region.
    vsini = _draw("vsini", broadening["vsini"], regions=n_regions)
    if use_empirical_vmacro:
        vmacro = numpyro.sample("vmacro", empirical_vmacro_valenti_fischer2005(
            teff, sigma=vmacro_empirical_sigma))
    else:
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
