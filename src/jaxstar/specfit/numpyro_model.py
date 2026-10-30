"""Single-star priors with Gaussian or Gaussian-mixture spectral likelihoods.

No optimizer or fitting state lives here.
Users can instead write their own NumPyro model using exactly the same helpers.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .broadening import default_broadening
from .continuum import _basis, _degree, chebyshev_basis, continuum_prior, apply_continuum
from .likelihood import (
    _validate_gp_solver, _mixture_parameters, gaussian_mixture_log_likelihood,
    gp_marginalized_continuum_log_likelihood,
    marginalized_continuum_log_likelihood, prepare_gp_observation,
)
from .model import SpecModel, _limb_darkening_from_q, _require
from .observation import Observation
from .sampling import _C_KMS


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
    """Return the empirical microturbulent velocity in km/s.

    Inputs are Teff in kelvin, logg in log10 cgs, and ``feh`` = [Fe/H] in
    dex; no metallicity conversion is performed. This is the Gaia-ESO
    prescription of Bergemann & Hill (v2, 2013), given in Bergemann & Hoppe
    (2026), Living Reviews in Computational Astrophysics 12, 7,
    https://doi.org/10.1007/s41115-026-00029-3. Its stated validity range is
    4000 < Teff < 7000 K, 0 < logg < 5, and -4.5 < [Fe/H] < 1.

    For logg >= 3.5, dwarfs/subgiants with Teff >= 5000 K use the original
    hot-dwarf polynomial; cooler dwarfs use its temperature terms evaluated
    at 5000 K, retaining the gravity and metallicity terms. Giants (logg <
    3.5) use a separate polynomial at their actual Teff. Values are continuous
    at 5000 K, with a slope change, but discontinuous across logg = 3.5.

    Apart from the prescribed cool-dwarf temperature freeze, no input or
    output clipping or validity checks are applied. Inputs broadcast under
    JAX; the caller chooses deterministic use or a prior with scatter.
    """
    teff = jnp.asarray(teff)
    logg = jnp.asarray(logg)
    feh = jnp.asarray(feh)
    dt = teff - 5500
    dg = logg - 4.0
    dt_dwarf = jnp.where(teff >= 5000, dt, -500)
    dwarf = (1.05 + 2.51e-4 * dt_dwarf + 1.5e-7 * dt_dwarf**2
             - 0.14 * dg - 0.005 * dg**2 + 0.05 * feh + 0.01 * feh**2)
    giant = (1.25 + 4.01e-4 * dt + 3.1e-7 * dt**2
             - 0.14 * dg - 0.005 * dg**2 + 0.05 * feh + 0.01 * feh**2)
    return jnp.where(logg >= 3.5, dwarf, giant)


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
    responsibility.
    """
    teff = jnp.asarray(teff)
    sigma = jnp.asarray(sigma)
    location = 3.98 + (teff - 5770.) / 650.
    return dist.TruncatedNormal(loc=location, scale=sigma, low=0.)


def empirical_vmacro_doyle2014(teff, logg, *, sigma=0.37):
    """Return a macroturbulence prior around the Doyle et al. (2014) relation.

    Inputs are Teff in kelvin and logg in log10 cgs; the returned NumPyro
    TruncatedNormal is in km/s, with the empirical relation as its location
    and lower bound zero. Equation (8) of MNRAS 444, 3592,
    https://doi.org/10.1093/mnras/stu1692, is calibrated over 5200--6400 K
    and logg 4.0--4.6 for radial-tangential macroturbulence.

    The default sigma=0.37 km/s adopts the observed residual RMS as a prior
    width, not a measurement-error-corrected intrinsic scatter. It does not
    use the paper's combined error of 0.73 km/s. Sigma must be positive and
    sets the standard deviation before truncation. Inputs broadcast under
    JAX; no sample sites, input/location clipping or validity checks are
    added. Applicability is the caller's responsibility.
    """
    dt = jnp.asarray(teff) - 5777.
    logg = jnp.asarray(logg)
    sigma = jnp.asarray(sigma)
    location = 3.21 + 2.33e-3 * dt + 2.00e-6 * dt**2 - 2.00 * (logg - 4.44)
    return dist.TruncatedNormal(loc=location, scale=sigma, low=0.)


def empirical_vmacro_lubin2026(teff, logg, *, sigma=0.09):
    """Return a macroturbulence prior around the Lubin et al. (2026) relation.

    Inputs are Teff in kelvin and logg in log10 cgs; the returned NumPyro
    TruncatedNormal is in km/s, with location 5.32 - 2.30*r and lower bound
    zero. Here r = 10**(logg - 4.44) * (teff / 5772)**(-0.5) is the
    dimensionless solar nu_max ratio, not a frequency in microhertz.
    This follows equations (2)--(3) and the table values in
    https://arxiv.org/abs/2609.17693 (v1); equation (3)'s printed slope units
    are inconsistent with that normalization. The radial-tangential
    calibration is restricted to r=0.4--1.4 in F/G/K dwarfs.

    The default sigma=0.09 km/s adopts the gold sample's observed residual
    RMS as a prior width, not a measurement-error-corrected intrinsic scatter.
    Sigma must be positive and sets the standard deviation before truncation.
    Inputs broadcast under JAX; no sample sites, input/location clipping or
    validity checks are added. Applicability is the caller's responsibility.
    """
    teff = jnp.asarray(teff)
    logg = jnp.asarray(logg)
    sigma = jnp.asarray(sigma)
    ratio = 10.**(logg - 4.44) * (teff / 5772.)**(-0.5)
    return dist.TruncatedNormal(loc=5.32 - 2.30 * ratio, scale=sigma, low=0.)


def _resolve_empirical_vmacro(option, sigma):
    """Resolve the static selector and its optional scalar prior width."""
    if option is False:
        return None, (), sigma
    if option is True:
        option = "valenti_fischer2005"
    if not isinstance(option, str):
        raise TypeError("use_empirical_vmacro must be a bool or a relation name")
    relations = {
        "valenti_fischer2005": (empirical_vmacro_valenti_fischer2005, ("teff",), 1.0),
        "doyle2014": (empirical_vmacro_doyle2014, ("teff", "logg"), 0.37),
        "lubin2026": (empirical_vmacro_lubin2026, ("teff", "logg"), 0.09),
    }
    if option not in relations:
        raise ValueError(f"use_empirical_vmacro must be False, True or one of {tuple(relations)}")
    helper, dependencies, default_sigma = relations[option]
    sigma = default_sigma if sigma is None else sigma
    value = jnp.asarray(sigma)
    if value.ndim != 0 or value.dtype.kind not in "fiu":
        raise ValueError("vmacro_empirical_sigma must be a finite positive scalar")
    _require(jnp.isfinite(value) & (value > 0),
             "vmacro_empirical_sigma must be a finite positive scalar")
    return helper, dependencies, sigma


def single_star_params(atmosphere, *, vsini, vmacro, q1, q2, rv, resolving_power=None):
    """Build one-component physical params, retaining frozen q1/q2 convention.

    This pure JAX helper is also useful in custom NumPyro models and outside
    inference when reconstructing spectra from a sample-site dictionary.
    Atmosphere coordinates are scalar; other values may be scalar or per region.
    Omit resolving_power when the SpecModel has a fixed_ip calibration.
    """
    u1, u2 = _limb_darkening_from_q(q1, q2)
    return {"components": ({"atmosphere": dict(atmosphere),
                            "broadening": {"vsini": vsini, "vmacro": vmacro, "u1": u1, "u2": u2},
                            "rv": rv},), "instrument": (
                                {} if resolving_power is None else {"resolving_power": resolving_power})}


def _draw(name, prior_or_value, *, regions=None, infer=None):
    """Record a distribution draw or fixed value, then check its scalar/region shape."""
    if isinstance(prior_or_value, dist.Distribution):
        # Region vectors are one joint site, without implicit plate broadcasting.
        prior = prior_or_value.to_event(len(prior_or_value.batch_shape))
        value = numpyro.sample(name, prior, infer=infer)
    else:
        value = numpyro.deterministic(name, jnp.asarray(prior_or_value))
    value = jnp.asarray(value)
    if value.ndim != 0 and (regions is None or value.shape != (regions,)):
        contract = "scalar" if regions is None else f"scalar or shape ({regions},)"
        raise ValueError(f"{name} must be {contract}, got {value.shape}")
    return value


def _validate_single_config(observation, specmodel, atmosphere, broadening, *,
                            use_physical_logg_max, use_empirical_vmic,
                            use_empirical_vmacro, vmacro_dependencies):
    """Check configuration and resolve BOSZ-auto vmic before sampling."""
    if not isinstance(observation, Observation) or not isinstance(specmodel, SpecModel):
        raise TypeError("model_single requires Observation and SpecModel")
    count = observation.n_regions
    if count != len(specmodel.spectra.regions):
        raise ValueError("observation rows must match SpecModel regions")
    if observation.region is not None and observation.region != specmodel.spectra.regions:
        raise ValueError(
            "observation region labels must match SpecModel row order")

    if not isinstance(use_physical_logg_max, bool):
        raise TypeError("use_physical_logg_max must be a bool")
    if use_empirical_vmic is not None and not isinstance(use_empirical_vmic, bool):
        raise TypeError("use_empirical_vmic must be None or a bool")

    axes = specmodel.spectra.grid.axis_names
    empirical_vmic_active = (specmodel.spectra.library == "bosz"
                             if use_empirical_vmic is None else use_empirical_vmic)
    if use_physical_logg_max and not {"teff", "logg"}.issubset(axes):
        raise ValueError(
            "use_physical_logg_max requires teff and logg atmosphere axes")
    if empirical_vmic_active and not {"teff", "logg", "mh", "vmic"}.issubset(axes):
        raise ValueError(
            "empirical vmic requires teff, logg, mh and vmic atmosphere axes")
    for name in vmacro_dependencies:
        if name not in axes:
            raise ValueError(f"use_empirical_vmacro requires a {name} atmosphere axis")

    required_axes = tuple(name for name in axes if not (
        empirical_vmic_active and name == "vmic"))
    if isinstance(atmosphere, Mapping) and empirical_vmic_active and "vmic" in atmosphere:
        raise ValueError("omit atmosphere['vmic'] when empirical vmic is active; "
                         "set use_empirical_vmic=False to supply an explicit vmic")
    if not isinstance(atmosphere, Mapping) or set(atmosphere) != set(required_axes):
        raise ValueError(
            f"atmosphere must supply exactly the named axes {required_axes}")
    if use_physical_logg_max and isinstance(atmosphere["logg"], dist.Distribution):
        prior = atmosphere["logg"]
        if not isinstance(prior, dist.Uniform):
            raise ValueError("use_physical_logg_max supports only a scalar Uniform prior "
                             "or fixed scalar logg")
        if prior.batch_shape or prior.event_shape:
            raise ValueError("logg must be scalar")

    names = ("vsini", "vmacro", "q1", "q2")
    reserved = (*names, "rv", "resolving_power", "sigma_constant", "sigma_continuum",
                "jitter", "u1", "u2", "spectrum", "model_flux",
                "mismatch_fraction", "mismatch_scale", "continuum_coefficients")
    if set(axes).intersection(reserved):
        raise ValueError(
            "atmosphere axis names conflict with single-star sample sites")
    required_broadening = tuple(name for name in names if not (
        use_empirical_vmacro and name == "vmacro"))
    if isinstance(broadening, Mapping) and use_empirical_vmacro and "vmacro" in broadening:
        raise ValueError(
            "omit broadening['vmacro'] when empirical vmacro is active")
    if not isinstance(broadening, Mapping) or set(broadening) != set(required_broadening):
        raise ValueError(
            f"broadening must supply exactly {', '.join(required_broadening)}")
    return axes, count, empirical_vmic_active


def _validate_gp_config(axes, gp_amplitude, gp_scale):
    """Select GP mode from the paired physical hyperparameter arguments."""
    if (gp_amplitude is None) != (gp_scale is None):
        raise ValueError(
            "supply both gp_amplitude and gp_scale, or neither for iid noise")
    if gp_amplitude is None:
        return False
    if set(axes).intersection(("gp_amplitude", "gp_scale")):
        raise ValueError("atmosphere axis names conflict with GP sample sites")
    return True


def _check_ip_choice(specmodel, resolving_power):
    if specmodel.fixed_ip is not None:
        if resolving_power is not None:
            raise ValueError("omit resolving_power when SpecModel has fixed_ip")
    elif resolving_power is None:
        raise ValueError("resolving_power is required unless SpecModel has fixed_ip")


def _validate_likelihood(likelihood, gp_enabled, mismatch_fraction, mismatch_scale):
    if not isinstance(likelihood, str) or likelihood not in ("gaussian", "gaussian_mixture"):
        raise ValueError("likelihood must be 'gaussian' or 'gaussian_mixture'")
    if likelihood == "gaussian_mixture":
        if gp_enabled:
            raise ValueError("likelihood='gaussian_mixture' requires use_gp=False and no GP parameters")
        if mismatch_fraction is None or mismatch_scale is None:
            raise ValueError("supply mismatch_fraction and mismatch_scale as numbers or distributions")
    elif mismatch_fraction is not None or mismatch_scale is not None:
        raise ValueError("mismatch parameters require likelihood='gaussian_mixture'")


def model_single(observation, specmodel, *, resolving_power=None,
                 atmosphere_bounds=None, rv_bounds=None, vsini_max=None,
                 vmacro_bounds=(0., 10.), fixed=None,
                 sigma_constant=0.1, sigma_continuum=0.05,
                 jitter_bounds=(1e-3, 0.05),
                 gp_amplitude_bounds=(1e-3, 0.05),
                 gp_scale_bounds=(np.exp(-5.), np.exp(2.)),
                 use_gp=None, gp_solver="auto", use_physical_logg_max=False,
                 likelihood="gaussian", mismatch_fraction=None, mismatch_scale=None,
                 use_empirical_vmic=None, use_empirical_vmacro=False,
                 vmacro_empirical_sigma=None, degree=4, basis=None,
                 prior_only=False, save_model_flux=False,
                 atmosphere=None, broadening=None, rv=None, jitter=None,
                 gp_amplitude=None, gp_scale=None):
    """Single-star priors and likelihood, written as an ordinary NumPyro model.

    Normally use make_model_single to prepare bounds from a grid and CCF.
    This function defines the distributions: uniform atmosphere/rotation/RV,
    uniform q1/q2, uniform or empirical vmacro, and log-uniform jitter/GP.
    Singleton atmosphere axes are fixed. ``fixed`` maps native site names to
    fixed values. Resolving power and continuum scales may also be supplied as
    numeric values or NumPyro distributions. With SpecModel(fixed_ip=...), omit
    resolving_power; no instrumental parameter sites are created.

    The explicit ``atmosphere``, ``broadening``, ``rv``, ``jitter`` and GP
    arguments remain available for low-level distribution overrides. Explicit
    atmosphere/broadening mappings must contain exactly the required names.
    With use_gp=None, supplying both GP arguments enables GP (neither disables
    it); True builds default GP priors, and False disables it.

    Physical logg replaces a Uniform's upper limit with physical_logg_max(teff),
    retaining its lower limit; fixed logg must satisfy that ceiling. BOSZ-auto
    vmic uses empirical_vmic(teff, logg, mh). These relations are evaluated
    without clipping. make_model_single checks the gravity support and prepares
    a compatible Teff interval; direct callers choose compatible bounds.

    use_empirical_vmacro=True (or "valenti_fischer2005") selects the legacy
    Teff-only prior; "doyle2014" and "lubin2026" also depend on logg. All
    return one shared vmacro sample with a zero-truncated Normal prior.
    vmacro_empirical_sigma=None uses widths of 1.0, 0.37 and 0.09 km/s,
    respectively; a positive scalar overrides the width. The latter two
    defaults are observed residual RMS values, not inferred intrinsic scatter.
    See the helpers for calibration ranges; inputs/locations are not clipped.

    With likelihood="gaussian" (default), continuum coefficients are analytically
    marginalized under Gaussian priors. With likelihood="gaussian_mixture", the
    same priors define explicit ``continuum_coefficients`` sites, optimized by
    run_svi or sampled by run_nuts. Per-pixel mixture
    indicators are summed out with logsumexp; there are no discrete sample sites.
    Supply global ``mismatch_fraction`` and absolute-flux ``mismatch_scale`` as
    numbers or scalar NumPyro distributions. Component variances are
    uncertainty**2+jitter**2 and uncertainty**2+jitter**2+mismatch_scale**2.
    Mixture mode does not support GP parameters; use_gp must be False or None.

    The optional Gaussian residual GP is Matérn-3/2, shared across independent
    regions. GP masks must be concrete when tracing. GP amplitude/jitter are in
    flux units, GP scale in wavelength units. All physical/noise sites retain
    their native names; fixed inputs become deterministic sites.

    gp_solver="vecchia1" opts into an experimental likelihood conditioning
    each observed residual on its preceding retained pixel. It retains local
    Matérn-3/2 covariances and the analytic continuum integral. Latent GP
    prediction is not supported with this approximation.

    ``prior_only=True`` records parameter sites and returns physical params
    before computing spectra or likelihoods. It supports check_prior. Normal
    evaluation returns the same params after recording the spectrum factor.
    """
    _validate_likelihood(
        likelihood, use_gp is True or gp_amplitude is not None or gp_scale is not None
        or bool({"gp_amplitude", "gp_scale"}.intersection(fixed or {})),
        mismatch_fraction, mismatch_scale)
    _check_ip_choice(specmodel, resolving_power)
    _validate_gp_solver(gp_solver)
    vmacro_helper, vmacro_dependencies, vmacro_empirical_sigma = _resolve_empirical_vmacro(
        use_empirical_vmacro, vmacro_empirical_sigma)
    fixed = dict(fixed or {})
    empirical_vmic_active = (specmodel.spectra.library == "bosz"
                             if use_empirical_vmic is None else use_empirical_vmic)
    axes = specmodel.spectra.grid.axis_names
    allowed_fixed = set(axes) | {"vsini", "vmacro", "q1", "q2", "rv", "jitter",
                                 "gp_amplitude", "gp_scale"}
    if set(fixed) - allowed_fixed:
        raise ValueError(
            f"unknown fixed parameters: {sorted(set(fixed) - allowed_fixed)}")
    if empirical_vmic_active and "vmic" in fixed:
        raise ValueError("omit fixed vmic when empirical vmic is active")
    if use_empirical_vmacro and "vmacro" in fixed:
        raise ValueError("omit fixed vmacro when empirical vmacro is active")

    # Uniform atmosphere priors over prepared bounds; singleton axes are fixed.
    if atmosphere is None:
        bounds = atmosphere_bounds if atmosphere_bounds is not None else {
            name: (float(specmodel.spectra.grid.axis(name)[0]),
                   float(specmodel.spectra.grid.axis(name)[-1])) for name in axes}
        atmosphere = {}
        for name in axes:
            if name == "vmic" and empirical_vmic_active:
                continue
            low, high = bounds[name]
            atmosphere[name] = (fixed[name] if name in fixed else
                                low if low == high else dist.Uniform(low, high))

    # Shared broadening priors. Explicit low-level mappings may replace them.
    if broadening is None:
        if vsini_max is None and "vsini" not in fixed:
            raise ValueError("supply vsini_max or fixed vsini")
        broadening = dict(
            vsini=fixed["vsini"] if "vsini" in fixed else dist.Uniform(
                0., vsini_max),
            q1=fixed["q1"] if "q1" in fixed else dist.Uniform(0., 1.),
            q2=fixed["q2"] if "q2" in fixed else dist.Uniform(0., 1.))
        if not use_empirical_vmacro:
            broadening["vmacro"] = (fixed["vmacro"] if "vmacro" in fixed
                                    else dist.Uniform(*vmacro_bounds))
    if rv is None:
        if "rv" in fixed:
            rv = fixed["rv"]
        elif rv_bounds is None:
            raise ValueError("supply rv_bounds or fixed rv")
        else:
            rv = dist.Uniform(*rv_bounds)
    if jitter is None:
        jitter = fixed["jitter"] if "jitter" in fixed else dist.LogUniform(
            *jitter_bounds)
    if use_gp is not None and not isinstance(use_gp, bool):
        raise TypeError("use_gp must be None or a bool")
    if use_gp is True:
        if gp_amplitude is None:
            gp_amplitude = (fixed["gp_amplitude"] if "gp_amplitude" in fixed
                            else dist.LogUniform(*gp_amplitude_bounds))
        if gp_scale is None:
            gp_scale = fixed["gp_scale"] if "gp_scale" in fixed else dist.LogUniform(
                *gp_scale_bounds)
    elif use_gp is False and (gp_amplitude is not None or gp_scale is not None
                              or {"gp_amplitude", "gp_scale"}.intersection(fixed)):
        raise ValueError("GP parameters conflict with use_gp=False")
    axes, n_regions, empirical_vmic_active = _validate_single_config(
        observation, specmodel, atmosphere, broadening,
        use_physical_logg_max=use_physical_logg_max, use_empirical_vmic=use_empirical_vmic,
        use_empirical_vmacro=use_empirical_vmacro, vmacro_dependencies=vmacro_dependencies)
    gp_enabled = _validate_gp_config(axes, gp_amplitude, gp_scale)

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
                logg = _draw("logg", dist.Uniform(lower, upper), infer={"prior": {
                    "depends_on": ("teff",),
                    "expression": "Uniform(logg_min, physical_logg_max(teff))"}})
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
        inputs = {name: coordinates[name] for name in vmacro_dependencies}
        vmacro = numpyro.sample("vmacro", vmacro_helper(
            **inputs, sigma=vmacro_empirical_sigma), infer={"prior": {
                "depends_on": vmacro_dependencies,
                "expression": (f"{vmacro_helper.__name__}({', '.join(vmacro_dependencies)}, "
                               f"sigma={vmacro_empirical_sigma})")}})
    else:
        vmacro = _draw("vmacro", broadening["vmacro"], regions=n_regions)
    q1 = _draw("q1", broadening["q1"], regions=n_regions)
    q2 = _draw("q2", broadening["q2"], regions=n_regions)
    rv = _draw("rv", rv, regions=n_regions)
    if resolving_power is not None:
        resolving_power = _draw(
            "resolving_power", resolving_power, regions=n_regions)

    # Continuum prior scales and additive noise.
    sigma_constant = _draw("sigma_constant", sigma_constant, regions=n_regions)
    sigma_continuum = _draw(
        "sigma_continuum", sigma_continuum, regions=n_regions)
    jitter = _draw("jitter", jitter, regions=n_regions)
    if gp_enabled:
        gp_amplitude = _draw("gp_amplitude", gp_amplitude)
        gp_scale = _draw("gp_scale", gp_scale)
    if likelihood == "gaussian_mixture":
        mismatch_fraction, mismatch_scale = _mixture_parameters(
            _draw("mismatch_fraction", mismatch_fraction), _draw("mismatch_scale", mismatch_scale))
        degree = _degree(degree)
        basis = chebyshev_basis(observation.wavelength, degree) if basis is None else _basis(basis)
        if basis.shape != observation.shape + (degree + 1,):
            raise ValueError("basis shape must be Observation.shape + (degree + 1,)")
        prior = continuum_prior(
            sigma_constant=sigma_constant, sigma_continuum=sigma_continuum,
            degree=degree, n_regions=None if observation.ndim == 1 else n_regions)
        coefficients = numpyro.sample(
            "continuum_coefficients", dist.Normal(prior.mean, prior.scale).to_event(prior.mean.ndim),
            infer={"prior": {
                "depends_on": ("sigma_constant", "sigma_continuum"),
                "expression": "Normal([1, 0, ...], [sigma_constant, sigma_continuum, ...])"}})

    # Convert limb darkening and evaluate the physical spectrum.
    params = single_star_params(
        coordinates, vsini=vsini, vmacro=vmacro, q1=q1, q2=q2,
        rv=rv, resolving_power=resolving_power)
    for name in ("u1", "u2"):
        numpyro.deterministic(
            name, params["components"][0]["broadening"][name])
    if prior_only:
        return params
    flux = specmodel(params, observation.wavelength)
    if save_model_flux:
        numpyro.deterministic("model_flux", flux)

    # Only the mixture samples continuum; its pixel indicators are summed out.
    if likelihood == "gaussian_mixture":
        usable_flux = jnp.where(observation.valid, flux, 0)
        log_likelihood = gaussian_mixture_log_likelihood(
            observation, apply_continuum(usable_flux, basis, coefficients),
            mismatch_fraction=mismatch_fraction, mismatch_scale=mismatch_scale, jitter=jitter)
    elif not gp_enabled:
        log_likelihood = marginalized_continuum_log_likelihood(
            observation, flux, degree=degree, basis=basis,
            sigma_constant=sigma_constant, sigma_continuum=sigma_continuum, jitter=jitter)
    else:
        try:
            gp_data = prepare_gp_observation(observation)
        except ValueError as exc:
            raise ValueError(
                "GP masks must be fixed before tracing model_single. Ordinary SVI/NUTS "
                "are supported; dynamically tracing Observation/mask, including "
                "jit_model_args=True, is not supported. Capture a concrete Observation "
                "for GP JIT evaluation.") from exc
        log_likelihood = gp_marginalized_continuum_log_likelihood(
            gp_data, flux, degree=degree, basis=basis,
            sigma_constant=sigma_constant, sigma_continuum=sigma_continuum, jitter=jitter,
            gp_amplitude=gp_amplitude, gp_scale=gp_scale, gp_solver=gp_solver)
    numpyro.factor("spectrum", log_likelihood)
    return params


def _bound_pair(value, name, *, shape=(), positive=False, singleton=False):
    """Validate static numeric bounds without constructing a distribution."""
    if len(value) != 2:
        raise ValueError(f"{name} must be a (lower, upper) pair")
    try:
        low, high = (np.broadcast_to(np.asarray(
            v, dtype=float), shape).copy() for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} endpoints must be scalar or shape {shape}") from exc
    ordered = low <= high if singleton else low < high
    if not np.all(np.isfinite(low) & np.isfinite(high) & ordered):
        raise ValueError(f"{name} must have finite ordered bounds")
    if positive and np.any(low <= 0):
        raise ValueError(f"{name} must be positive")
    return (float(low), float(high)) if not shape else (low, high)


def make_model_single(sf, *, resolving_power=None, atmosphere_bounds=None, rv_bounds=None,
                      vsini_max=None, vmacro_bounds=(1., 15.), fixed=None,
                      degree=4, basis=None, use_gp=True, gp_solver="auto",
                      likelihood="gaussian", mismatch_fraction=None, mismatch_scale=None,
                      use_physical_logg_max=False, use_empirical_vmic=None,
                      use_empirical_vmacro=False, vmacro_empirical_sigma=None,
                      sigma_constant=0.1, sigma_continuum=0.05,
                      jitter_bounds=None, gp_amplitude_bounds=None,
                      gp_scale_bounds=None):
    """Prepare a configured single-star model from a SpecFit's grid and CCF.

    The factory resolves numeric bounds/settings; model_single defines every
    distribution. Override selected atmosphere_bounds (remaining axes use grid
    endpoints), rv_bounds (scalar endpoints or one endpoint per region), and
    vsini_max. ``fixed`` supplies numeric values by native site name, e.g.
    {"logg": 4.4, "vmacro": 0.}. Atmosphere coordinates must be scalar; broadening,
    RV and jitter may be shared or per region. GP hyperparameters are shared.
    gp_solver="vecchia1" selects the experimental likelihood described in
    model_single. SpecFit.predict reconstructs its conditional continuum and
    returns mean_flux = model_flux, omitting the latent GP mean. Its marginal
    residual variance is available for the standard mixture outlier masking.
    Resolving power is required, as a positive scalar/region value or a NumPyro
    distribution. Infinity disables the Gaussian instrumental profile. With a
    fixed_ip SpecModel, omit resolving_power; its fixed mixture is used instead.

    ``likelihood="gaussian_mixture"`` requires ``use_gp=False`` and explicit
    ``mismatch_fraction``/``mismatch_scale`` numbers or scalar distributions.
    Both are global; the latter is an absolute flux amplitude. Continuum
    coefficients are explicit inference sites, while pixel mixture indicators
    are summed out. Both run_svi and run_nuts accept this configured model.

    CCF is required only when RV or vsini still needs a default. RV has one
    Uniform per region centered on median CCF RV, with half-width 5 km/s if
    CCF width <20 km/s, otherwise half that width. No coverage clipping occurs.
    vsini defaults to the CCF width capped at 90% of default kernel support.
    Explicit vsini_max must fit that support. Bounds and settings are snapshots;
    fit masks are always read anew by SpecFit when inference/reconstruction runs.

    Omitted jitter/amplitude bounds are (1e-3, 0.05) in normalized flux units,
    independent of observational errors and masks. Omitted GP
    scale bounds run from 5 km/s at each region's wavelength midpoint to one
    fifth of its full wavelength span. The shared interval uses the largest
    lower and smallest upper bound across regions. Explicit bounds override
    these defaults; fixed parameters and disabled GP need no derived bounds.

    With physical logg enabled, the selected Teff bounds are intersected with
    4500--7000 K; fixed Teff must already be in that interval. The resulting
    gravity prior must fit the loaded grid. The model's temperature-dependent
    ceiling replaces a Uniform logg upper bound, retaining its lower bound.
    Other empirical relations retain their unclipped behavior.
    use_empirical_vmacro accepts False, True/"valenti_fischer2005",
    "doyle2014" or "lubin2026" as in model_single. With
    vmacro_empirical_sigma=None, the selected relation supplies its default.

    Returns ConfiguredModel: call it as an ordinary NumPyro model, inspect
    model.settings/check_prior(), and pass model=model to SpecFit methods.
    """
    from .configured_model import ConfiguredModel
    from .reconstruction import reconstruct_single

    count = sf.observation.n_regions
    spectra = sf.specmodel.spectra
    if not isinstance(use_gp, bool):
        raise TypeError("use_gp must be a bool")
    _validate_likelihood(
        likelihood, use_gp or bool({"gp_amplitude", "gp_scale"}.intersection(fixed or {})),
        mismatch_fraction, mismatch_scale)
    degree = _degree(degree)
    _validate_gp_solver(gp_solver)
    if not isinstance(use_physical_logg_max, bool):
        raise TypeError("use_physical_logg_max must be a bool")
    _, _, vmacro_empirical_sigma = _resolve_empirical_vmacro(
        use_empirical_vmacro, vmacro_empirical_sigma)
    if use_empirical_vmic is not None and not isinstance(use_empirical_vmic, bool):
        raise TypeError("use_empirical_vmic must be None or a bool")
    empirical_vmic_active = spectra.library == "bosz" if use_empirical_vmic is None else use_empirical_vmic
    fixed = dict(fixed or {})
    overrides = dict(atmosphere_bounds or {})
    if set(overrides) - set(spectra.grid.axis_names):
        raise ValueError(
            f"unknown atmosphere bounds: {sorted(set(overrides) - set(spectra.grid.axis_names))}")
    if empirical_vmic_active and "vmic" in overrides:
        raise ValueError("omit vmic bounds when empirical vmic is active")
    bounds = {}
    for name in spectra.grid.axis_names:
        axis = np.asarray(spectra.grid.axis(name))
        low, high = _bound_pair(overrides.get(name, (axis[0], axis[-1])),
                                name, singleton=True)
        if low < axis[0] or high > axis[-1]:
            raise ValueError(f"{name} bounds extend outside the loaded grid")
        bounds[name] = (low, high)

    allowed = set(spectra.grid.axis_names) | {"vsini", "vmacro", "q1", "q2", "rv",
                                              "jitter", "gp_amplitude", "gp_scale"}
    if set(fixed) - allowed:
        raise ValueError(
            f"unknown fixed parameters: {sorted(set(fixed) - allowed)}")
    for name, value in fixed.items():
        value = np.asarray(value)
        shapes = ((),) if name in bounds or name.startswith(
            "gp_") else ((), (count,))
        if value.dtype.kind not in "fiu" or value.shape not in shapes or not np.all(np.isfinite(value)):
            raise ValueError(
                f"fixed {name} must be finite numeric with shape in {shapes}")
        if name in bounds and not bounds[name][0] <= value <= bounds[name][1]:
            raise ValueError(f"fixed {name} is outside its atmosphere bounds")
        if name in ("vsini", "vmacro", "jitter") and np.any(value < 0):
            raise ValueError(f"fixed {name} must be nonnegative")
        if name in ("q1", "q2") and np.any((value < 0) | (value > 1)):
            raise ValueError(f"fixed {name} must be in [0, 1]")
        if name.startswith("gp_") and np.any(value <= 0):
            raise ValueError(f"fixed {name} must be positive")

    _check_ip_choice(sf.specmodel, resolving_power)
    if resolving_power is not None:
        if isinstance(resolving_power, dist.Distribution):
            shape = resolving_power.batch_shape + resolving_power.event_shape
        else:
            resolving_power = np.array(resolving_power, copy=True)
            shape = resolving_power.shape
            if resolving_power.dtype.kind not in "fiu" or not np.all(
                    (resolving_power > 0) & ~np.isnan(resolving_power)):
                raise ValueError(
                    "resolving_power must be positive (infinity disables Gaussian IP)")
        if shape not in ((), (count,)):
            raise ValueError(f"resolving_power must be scalar or shape ({count},)")

    needs_rv = rv_bounds is None and "rv" not in fixed
    needs_vsini = vsini_max is None and "vsini" not in fixed
    if needs_rv or needs_vsini:
        if sf.ccf is None:
            raise ValueError(
                "run check_ccf or specify rv_bounds/fixed rv and vsini_max/fixed vsini")
        width = np.asarray(sf.ccf["width"])
        if width.ndim or not np.isfinite(width) or width <= 0:
            raise ValueError("CCF width must be finite and positive")
        if needs_rv:
            center = np.asarray(sf.ccf["rv"])
            if center.ndim or not np.isfinite(center):
                raise ValueError("CCF RV must be a finite scalar")
            half_width = 5. if width < 20. else .5 * width
            rv_bounds = (float(center - half_width),
                         float(center + half_width))
        if needs_vsini:
            vsini_max = float(width)
            if sf.specmodel.broadening_operator is default_broadening:
                vsini_max = min(
                    vsini_max, .9 * float(np.min(sf.specmodel.velocity_grid[:, -1])))
    if rv_bounds is not None:
        rv_bounds = _bound_pair(rv_bounds, "rv_bounds", shape=(count,))
    if vsini_max is not None:
        vsini_max = np.asarray(vsini_max)
        if vsini_max.ndim or not np.isfinite(vsini_max) or vsini_max <= 0:
            raise ValueError("vsini_max must be finite and positive")
        vsini_max = float(vsini_max)
    if sf.specmodel.broadening_operator is default_broadening:
        support = float(np.min(sf.specmodel.velocity_grid[:, -1]))
        if np.any(np.asarray(fixed.get("vsini", vsini_max)) > support):
            raise ValueError(
                "vsini exceeds broadening support; increase SpecModel vmax")

    if use_physical_logg_max:
        if not {"teff", "logg"}.issubset(bounds):
            raise ValueError(
                "use_physical_logg_max requires teff and logg atmosphere axes")
        low, high = bounds["teff"]
        if "teff" in fixed:
            low = high = float(fixed["teff"])
            if not 4500. <= low <= 7000.:
                raise ValueError(
                    "fixed teff is outside the physical_logg_max temperature range")
        else:
            singleton = low == high
            low, high = max(low, 4500.), min(high, 7000.)
            if low > high or (low == high and not singleton):
                raise ValueError(
                    "no grid support in the physical_logg_max temperature range 4500--7000 K")
            bounds["teff"] = (low, high)
        ceilings = np.asarray(physical_logg_max(np.array([low, high])))
        logg_low, logg_high = bounds["logg"]
        if "logg" in fixed or logg_low == logg_high:
            if fixed.get("logg", logg_low) > np.min(ceilings):
                raise ValueError(
                    "fixed logg exceeds physical_logg_max over the Teff prior")
        elif np.min(ceilings) <= logg_low or np.max(ceilings) > spectra.grid.axis("logg")[-1]:
            raise ValueError(
                "physical_logg_max prior extends outside the loaded logg grid")

    vmacro_bounds = _bound_pair(vmacro_bounds, "vmacro_bounds")
    if vmacro_bounds[0] < 0:
        raise ValueError("vmacro_bounds must be nonnegative")
    noise_bounds = {}
    for name, value, active in (("jitter", jitter_bounds, True),
                                ("gp_amplitude", gp_amplitude_bounds, use_gp),
                                ("gp_scale", gp_scale_bounds, use_gp)):
        if value is None and active and name not in fixed:
            if name == "gp_scale":
                wave = np.atleast_2d(np.asarray(sf.observation.wavelength))
                center = (wave[:, 0] + wave[:, -1]) / 2
                value = (np.max(center) * 5. / _C_KMS,
                         np.min(wave[:, -1] - wave[:, 0]) / 5.)
            else:
                value = (1e-3, 0.05)
        noise_bounds[f"{name}_bounds"] = (None if value is None else
            _bound_pair(value, f"{name}_bounds", positive=True))
    settings = dict(
        atmosphere_bounds=bounds, rv_bounds=rv_bounds, vsini_max=vsini_max,
        vmacro_bounds=vmacro_bounds, fixed=fixed,
        resolving_power=resolving_power, degree=degree, basis=basis,
        use_gp=use_gp, gp_solver=gp_solver, sigma_constant=sigma_constant,
        likelihood=likelihood, mismatch_fraction=mismatch_fraction, mismatch_scale=mismatch_scale,
        sigma_continuum=sigma_continuum,
        **noise_bounds,
        use_physical_logg_max=use_physical_logg_max, use_empirical_vmic=use_empirical_vmic,
        use_empirical_vmacro=use_empirical_vmacro, vmacro_empirical_sigma=vmacro_empirical_sigma)
    annotations = {
        "u1": dict(depends_on=("q1", "q2"), expression="2 * sqrt(q1) * q2"),
        "u2": dict(depends_on=("q1", "q2"), expression="sqrt(q1) - u1")}
    if empirical_vmic_active:
        annotations["vmic"] = dict(depends_on=("teff", "logg", "mh"),
                                   expression="empirical_vmic(teff, logg, mh)")
    model = ConfiguredModel(model_single, settings=settings, reconstruct_fn=reconstruct_single,
                            observation=sf.observation, specmodel=sf.specmodel,
                            site_annotations=annotations)
    # Validate the resolved model without spectra/likelihood work.
    model.prior_trace()
    return model
