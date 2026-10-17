"""NumPyro isochrone/track models and shared NUTS/MultiNest inference functions."""

__all__ = ["MistObservations", "InitialGridSearch", "model_isochrone", "model_track", "find_initial", "make_init_strategy",
           "run_nuts", "run_multinest"]

from dataclasses import dataclass, field
from functools import partial
import inspect
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit, random
from numpyro.infer import init_to_value
from numpyro.infer.util import log_density
from .gyrochrone_likelihood import loglike_gyro
from .results import _new_run_record, _finish_run_record, _json_value, _native_files, _file_hash


@dataclass(frozen=True)
class MistObservations:
    """Ordered independent Gaussian measurements in MistModel output units.

    ``keys``, ``values`` and ``errors`` are copied to immutable tuples. A mapping
    of ``{key: (value, error)}`` can be converted with ``from_mapping`` or passed
    directly to MistFit. An empty observation set is allowed for prior runs.
    """

    keys: tuple
    values: tuple
    errors: tuple

    def __post_init__(self):
        keys, values, errors = tuple(self.keys), tuple(
            self.values), tuple(self.errors)
        if any(not isinstance(k, str) or not k for k in keys):
            raise ValueError("observation keys must be nonempty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("observation keys must be unique")
        if len(keys) != len(values) or len(keys) != len(errors):
            raise ValueError(
                "observation keys, values and errors must have equal lengths")
        for name, items in (("values", values), ("errors", errors)):
            array = np.asarray(items)
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise ValueError(f"observation {name} must be real scalars")
            if not np.all(np.isfinite(array)) or (name == "errors" and np.any(array <= 0)):
                raise ValueError(
                    f"observation {name} must be finite" + (" and positive" if name == "errors" else ""))
        object.__setattr__(self, "keys", keys)
        # Scalar copies retain input precision without an early JAX conversion.
        object.__setattr__(self, "values", tuple(
            np.asarray(v)[()] for v in values))
        object.__setattr__(self, "errors", tuple(
            np.asarray(e)[()] for e in errors))

    @classmethod
    def from_mapping(cls, observations):
        """Build from ``{output_key: (value, Gaussian_error)}`` in mapping order."""
        keys = tuple(observations)
        pairs = tuple(observations[key] for key in keys)
        if any(np.shape(pair) != (2,) for pair in pairs):
            raise ValueError("each observation must be a (value, error) pair")
        return cls(keys, tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs))


# Historical posterior outputs. Inference dependencies are selected separately
# inside model_isochrone(), without storing a mutable selection on the fitter or grid.
_DEFAULT_OUTPUT_KEYS = (
    'kmag', 'teff', 'logg', 'mass', 'radius', 'feh_photosphere', 'star_mass',
    'dmdeep', 'mmin', 'mmax', 'bpmag2', 'rpmag2',
)


def _normalize_output_keys(keys):
    """Make a hashable, ordered additional-output selection for one call."""
    if keys is None:
        return ()
    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("output_keys must contain nonempty strings")
    return tuple(dict.fromkeys(keys))


@jit
def smbound(x, low=None, upp=None, s=20, depth=30):
    """Smooth log penalty; each bound contributes -depth/2 at its location.

    A missing bound contributes zero. The one-sided branches use stable
    sigmoids; the two-sided expression preserves historical isochrone math.
    """
    if low is None:
        return jnp.zeros_like(x) if upp is None else -depth * jax.nn.sigmoid(s * (x - upp))
    if upp is None:
        return -depth * jax.nn.sigmoid(s * (low - x))
    return -depth*(1./(jnp.exp(s*(x-low))+1)+1./(jnp.exp(-s*(x-upp))+1))


def model_isochrone(observations, physical_model, nodata=False, linear_age=True,
                    flat_age_marginal=False, logamin=8, logamax=10.14,
                    fmin=-1, fmax=0.5, eepmin=0, eepmax=600,
                    massmin=0.1, massmax=2.5, dist_scale=1.35,
                    prot=None, prot_err=0.05, rho_obs=None, rho_err=None,
                    age_obs=None, age_err=None, *, output_keys=None):
    """Isochrone probability model shared by NUTS and MultiNest.

    Args:
        observations: MistObservations containing independent Gaussian data.
        physical_model: Deterministic MistModel, independent of observations.
        nodata: Disable the main data likelihood; optional factors remain active.
        linear_age: Use a flat prior in age (True) or logage (False).
        flat_age_marginal: Normalize the mass interval at each age so the marginal
            age/logage prior is flat, retaining the historical mass-limit rule.
        logamin, logamax: Bounds for log10(age/yr).
        fmin, fmax: Bounds for initial FeH (dex).
        eepmin, eepmax: Bounds for EEP.
        massmin, massmax: Smooth bounds for initial mass (solar masses).
        dist_scale: Gamma distance-prior scale in kpc (Bailer-Jones 2015;
            Astraatmadja & Bailer-Jones 2016).
        prot: Optional rotation period, using Angus et al. (2019), AJ 158, 173.
        prot_err: Uncertainty used by the gyro likelihood.
        rho_obs, rho_err: Gaussian constraint on INITIAL mass / radius**3 in
            solar units, retained for historical compatibility.
        age_obs, age_err: Gaussian constraint on age in Gyr.
        output_keys: Additional deterministic MistModel outputs for this call.
            A name or sequence; None/empty keeps defaults and observed outputs.
            Duplicates are ignored. Keep this argument static under JIT.
    """
    saved_keys = tuple(dict.fromkeys((
        *_DEFAULT_OUTPUT_KEYS,
        *(key for key in observations.keys if key not in ('parallax', 'feh')),
        *_normalize_output_keys(output_keys),
    )))
    required_keys = ['parallax', 'feh', 'mass', 'dmdeep']
    if flat_age_marginal:
        required_keys += ['mmin', 'mmax']
    if prot is not None:
        required_keys += ['bpmag2', 'rpmag2']  # DR2 BP-RP for gyro likelihood
    if rho_obs is not None and rho_err is not None:
        required_keys += ['radius']

    if linear_age:
        age = numpyro.sample("age", dist.Uniform(
            10**logamin/1e9, 10**logamax/1e9))
        logage = numpyro.deterministic("logage", jnp.log10(age * 1e9))
    else:
        logage = numpyro.sample("logage", dist.Uniform(logamin, logamax))
        age = numpyro.deterministic("age", 10**logage/1e9)

    feh_init = numpyro.sample("feh_init", dist.Uniform(fmin, fmax))
    eep = numpyro.sample("eep", dist.Uniform(eepmin, eepmax))

    distance = numpyro.sample("distance", dist.Gamma(
        3, rate=1./dist_scale))  # BJ18, kpc
    params = dict(physical_model(
        {"logage": logage, "feh_init": feh_init, "eep": eep, "distance": distance},
        keys=tuple(dict.fromkeys((*saved_keys, *required_keys))),
    ))
    numpyro.deterministic("parallax", params["parallax"])  # mas
    for key in saved_keys:
        # Coordinates/parallax already have sites; public derived keys may
        # also be observed or requested as outputs without duplicate sites.
        if key not in ("age", "logage", "feh_init", "eep", "distance", "parallax", "feh"):
            numpyro.deterministic(key, params[key])
    # "feh" is set to be photospheric value
    numpyro.deterministic("feh", params["feh"])

    if not nodata:
        obsparams = jnp.array([params[key] for key in observations.keys])
        numpyro.sample("obs", dist.Normal(obsparams, jnp.array(
            observations.errors)), obs=jnp.array(observations.values))

    # The prior uses the separately interpolated centered dm/dEEP field.
    # It deliberately approximates the smooth physical relation, rather
    # than differentiating the piecewise-linear mass value interpolant.
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


_TRACK_OUTPUT_KEYS = (
    'kmag', 'teff', 'logg', 'mass', 'mass_init', 'radius', 'feh_photosphere',
    'star_mass', 'dtdeep', 'bpmag2', 'rpmag2', 'age', 'logage',
)


def _require_concrete_valid(valid, message):
    """Raise for concrete invalid options; traced options get a support factor."""
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(message)


def _track_bounds(axis, low, high, name):
    low = axis[0] if low is None else jnp.asarray(low)
    high = axis[-1] if high is None else jnp.asarray(high)
    if jnp.ndim(low) != 0 or jnp.ndim(high) != 0:
        raise ValueError(f"{name} bounds must be scalars")
    valid = jnp.isfinite(low) & jnp.isfinite(high) & (
        axis[0] <= low) & (low < high) & (high <= axis[-1])
    _require_concrete_valid(
        valid, f"{name} bounds must lie inside the available track grid")
    return (jnp.where(valid, low, axis[0]), jnp.where(valid, high, axis[-1])), valid


def _track_age_bounds(low, high, width, depth):
    """Validate optional Gyr bounds and keep traced invalid options finite."""
    values = [v for v in (low, high, width, depth) if v is not None]
    if any(jnp.ndim(jnp.asarray(v)) != 0 for v in values) or width is None or depth is None:
        raise ValueError(
            "age bounds, age_bound_width and age_bound_depth must be scalars")
    valid = jnp.isfinite(width) & (
        width > 0) & jnp.isfinite(depth) & (depth > 0)
    if low is not None:
        valid &= jnp.isfinite(low) & (low >= 0)
    if high is not None:
        valid &= jnp.isfinite(high) & (high > 0)
    if low is not None and high is not None:
        valid &= low < high
    _require_concrete_valid(valid, "age bounds must be finite, nonnegative and increasing; "
                            "age_max, age_bound_width and age_bound_depth must be positive")
    return (None if low is None else jnp.where(valid, low, 0.),
            None if high is None else jnp.where(valid, high, 1.),
            jnp.where(valid, width, .1), jnp.where(valid, depth, 30.), valid)


def model_track(observations, physical_model, nodata=False, linear_age=True,
                fmin=None, fmax=None,
                eepmin=None, eepmax=None, massmin=None, massmax=None,
                dist_scale=1.35, prot=None, prot_err=0.05,
                rho_obs=None, rho_err=None, age_obs=None, age_err=None,
                *, output_keys=None, age_min=None, age_max=13.8,
                age_bound_width=0.1, age_bound_depth=30.):
    """Track probability model with initial mass/FeH/EEP/distance coordinates.

    Mass (site ``mass``) and initial FeH have Uniform priors. Their bounds,
    and the EEP bounds, default to the grid endpoints; explicit bounds must
    lie inside the grid. Unlike model_isochrone's smooth mass bounds, these
    are hard sampling limits. EEP has a fixed Uniform prior for both samplers.
    Age is an output in Gyr, with a soft upper bound of 13.8 Gyr by default
    (age_max=None disables it). There is no default lower bound. linear_age selects
    weights dtdeep (True) or dtdeep / (age * ln(10)) (False), using the separately
    interpolated centered derivative, not autodiff of age. No age-span
    normalization or flat_age_marginal correction is applied: grid coverage
    still shapes the marginal mass and age distributions.

    age_min / age_max apply one-sided smbound penalties to age in Gyr; None
    disables that side. age_bound_width is the sigmoid scale in Gyr (default 0.1); each bound
    contributes -age_bound_depth/2 at the bound and approaches -age_bound_depth
    outside (default depth 30). These are soft weights, not hard exclusions.
    They apply even with nodata=True and never change the EEP sampling bounds.

    Observations, optional gyro/initial-density/age constraints, distance
    prior, and per-call additional output_keys follow model_isochrone.
    nodata disables only the main Gaussian observations. JAX x64 is required.
    Missing age/derivative stencils or required likelihood outputs have zero
    density. Unsupported saved-only outputs do not restrict the prior.
    """
    if physical_model.kind != "track":
        raise ValueError("model_track requires MistModel.from_tracks(...)")
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for track inference")
    mass_bounds, mass_valid = _track_bounds(
        physical_model.grid.axis("initial_mass"), massmin, massmax, "mass")
    feh_bounds, feh_valid = _track_bounds(
        physical_model.grid.axis("initial_feh"), fmin, fmax, "FeH")
    eep_bounds, eep_valid = _track_bounds(
        physical_model.grid.axis("eep"), eepmin, eepmax, "EEP")
    age_min, age_max, age_bound_width, age_bound_depth, age_valid = _track_age_bounds(
        age_min, age_max, age_bound_width, age_bound_depth)
    if jnp.ndim(dist_scale) != 0:
        raise ValueError("dist_scale must be a scalar")
    distance_valid = jnp.isfinite(dist_scale) & (dist_scale > 0)
    _require_concrete_valid(
        distance_valid, "dist_scale must be finite and positive")

    saved_keys = tuple(dict.fromkeys((*_TRACK_OUTPUT_KEYS, *observations.keys,
                                     *_normalize_output_keys(output_keys))))
    required_keys = ("parallax", "feh", "age", "logage", "dtdeep")
    mass = numpyro.sample("mass", dist.Uniform(*mass_bounds))
    feh_init = numpyro.sample("feh_init", dist.Uniform(*feh_bounds))
    eep_lo, eep_hi = eep_bounds
    eep = numpyro.sample("eep", dist.Uniform(eep_lo, eep_hi))
    numpyro.deterministic("eep_lower", eep_lo)
    numpyro.deterministic("eep_upper", eep_hi)
    distance = numpyro.sample("distance", dist.Gamma(
        3, rate=1. / jnp.where(distance_valid, dist_scale, 1.35)))
    params = physical_model(
        {"mass_init": mass, "feh_init": feh_init,
            "eep": eep, "distance": distance},
        keys=tuple(dict.fromkeys((*saved_keys, *required_keys))),
    )
    for key in dict.fromkeys((*saved_keys, "parallax", "feh")):
        if key not in ("mass", "feh_init", "eep", "distance"):
            numpyro.deterministic(key, params[key])
    likelihood_keys = list(observations.keys) if not nodata else []
    if prot is not None:
        likelihood_keys.extend(("bpmag2", "rpmag2", "logage", "feh"))
    if rho_obs is not None and rho_err is not None:
        likelihood_keys.append("radius")
    likelihood_valid = jnp.array(True)
    for key in dict.fromkeys(likelihood_keys):
        likelihood_valid &= jnp.isfinite(params[key])
    # Keep rejected evaluations numerically finite until the support factor.
    # In particular Normal(-inf, sigma) would produce NaN reverse gradients.
    safe = {key: jnp.where(jnp.isfinite(value), value, 1.) for key, value in params.items()}
    if not nodata:
        obsparams = jnp.array([safe[key] for key in observations.keys])
        numpyro.sample("obs", dist.Normal(obsparams, jnp.array(observations.errors)),
                       obs=jnp.array(observations.values))

    dt = params["dtdeep"]
    valid = (mass_valid & feh_valid & eep_valid & age_valid & distance_valid & likelihood_valid
             & (mass >= mass_bounds[0]) & (mass <= mass_bounds[1])
             & (feh_init >= feh_bounds[0]) & (feh_init <= feh_bounds[1])
             & (eep >= eep_lo) & (eep <= eep_hi)
             & jnp.isfinite(params["age"]) & (params["age"] > 0)
             & jnp.isfinite(dt) & (dt > 0))
    logjac = jnp.log(jnp.where(valid, dt, 1.))
    safe_age = jnp.where(valid, params["age"], 1.)
    if not linear_age:
        logjac -= jnp.log(safe_age * jnp.log(10.))
    numpyro.factor("logjac", jnp.where(valid, logjac, -jnp.inf))
    if age_min is not None or age_max is not None:
        numpyro.factor("logage_bound",
                       smbound(safe_age, low=age_min, s=1. /
                               age_bound_width, depth=age_bound_depth)
                       + smbound(safe_age, upp=age_max, s=1. / age_bound_width, depth=age_bound_depth))

    if prot is not None:
        numpyro.factor("loglike_gyro", loglike_gyro(
            prot, jnp.where(likelihood_valid, safe['bpmag2'] - safe['rpmag2'], .8),
            mass, eep, safe['logage'], safe['feh'], sigma=prot_err))
    if rho_obs is not None and rho_err is not None:
        numpyro.factor("loglike_rho", -0.5 * (rho_obs - mass /
                       safe['radius']**3)**2 / rho_err**2)
    if age_obs is not None and age_err is not None:
        numpyro.factor("loglike_age", -0.5 *
                       (age_obs - safe['age'])**2 / age_err**2)


def _default_model(physical_model):
    """Resolve once in Python; kind is static PyTree metadata under JIT."""
    if physical_model.kind == "isochrone":
        return model_isochrone
    if physical_model.kind == "track":
        return model_track
    raise ValueError(f"unsupported MIST model kind: {physical_model.kind}")


def _initial_axis(axis, low, high, name):
    """Native nodes inside open prior bounds, or a midpoint for a narrow range."""
    axis = np.asarray(axis)
    if np.ndim(low) or np.ndim(high) or not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(f"{name} bounds must be finite increasing scalars")
    left, right = max(low, axis[0]), min(high, axis[-1])
    if left >= right:
        raise ValueError(f"{name} bounds have no interior overlap with the grid")
    nodes = axis[(axis > low) & (axis < high)]
    if not nodes.size:
        nodes = np.asarray([left + (right - left) / 2], dtype=axis.dtype)
    if not np.all((nodes > low) & (nodes < high)):
        raise ValueError(f"{name} bounds are too narrow for grid precision")
    return nodes


@dataclass(frozen=True)
class InitialGridSearch:
    """In-memory search scores and their target, for optional coverage checks.

    Scores are constrained joint log densities, not marginalized probabilities.
    log_density is the maximum over feh_values at each two-dimensional node;
    best_feh retains its maximizing FeH (NaN where no slice has finite density).
    log_density_by_feh has shape (FeH, first axis, EEP).
    Axes are mass/EEP for tracks or log10(age/yr)/EEP for isochrones. Holding
    the model here shares its existing arrays; it does not duplicate the grid.
    """

    axis_names: tuple
    axes: tuple
    bounds: tuple
    log_density: np.ndarray = field(repr=False)
    fixed_values: dict
    observations: MistObservations = field(repr=False)
    physical_model: object = field(repr=False)
    model_kwargs: dict = field(repr=False)
    feh_values: np.ndarray
    feh_step: object
    log_density_by_feh: np.ndarray = field(repr=False)
    best_feh: np.ndarray = field(repr=False)


def _feh_scan_axis(axis, low, high, step, extra=None):
    """Step-spaced dex nodes anchored at zero, plus an optional starting value."""
    fallback = _initial_axis(axis, low, high, 'FeH')
    if (isinstance(step, (bool, np.bool_)) or np.ndim(step)
            or not np.isfinite(step) or step <= 0):
        raise ValueError('feh_step must be a finite positive scalar')
    left, right = max(low, float(axis[0])), min(high, float(axis[-1]))
    nodes = step * np.arange(np.ceil(left / step), np.floor(right / step) + 1)
    # Correct grid-edge roundoff, but keep starts away from open prior endpoints.
    nodes = np.clip(nodes, left, right)
    tolerance = 4 * np.finfo(float).eps * max(1., abs(low), abs(high))
    nodes = nodes[(nodes > low + tolerance) & (nodes < high - tolerance)]
    if not nodes.size:
        nodes = np.array([left + (right - left) / 2])
        if not low < nodes[0] < high:
            nodes = fallback
    if extra is not None:
        nodes = np.r_[nodes, extra]
    return np.unique(nodes)


def find_initial(observations, physical_model, *, feh_init=None, distance=None,
                 model_kwargs=None, return_grid=False, feh_step=None):
    """Find constrained NUTS starting values, optionally scanning initial FeH.

    Search all native initial-mass x EEP nodes for tracks, or log-age x EEP
    nodes for isochrones, within the sampling bounds in model_kwargs. Uniform
    prior endpoints are excluded so the unconstrained start remains finite.
    A range containing no interior native node uses its overlap midpoint.
    Isochrone mass bounds remain soft and only affect the score.

    With feh_init=None, use observed surface FeH (feh, feh_surface, then
    feh_photosphere), or zero if absent. This is only an approximate initial
    FeH starting value; it does not equate the initial and surface abundances
    in the inference model. An explicit feh_init overrides this choice.
    With feh_step=None this FeH is fixed and must lie inside the open prior
    bounds and the grid. feh_step=0.1 instead scans multiples of 0.1 dex within
    their overlap, adding the chosen FeH if supported. Interpolation evaluates
    intermediate values without rebuilding the NPZ. Unsupported automatic
    FeH guesses are skipped during a scan; explicit invalid feh_init raises.
    A range containing no step-aligned interior node uses its midpoint.
    distance is in kpc;
    when omitted, use the reciprocal of a positive observed parallax in mas.
    Distance is fixed only during this search. Rank candidates by the existing
    model's constrained joint log density, including all enabled likelihood
    and prior factors, and return the best with finite density and gradients.
    The returned dict contains only latent sample sites, including age in Gyr
    or logage for isochrones according to linear_age. No SVI is performed;
    this starting point does not guarantee discovery of all posterior modes.
    return_grid=True returns (initial, InitialGridSearch) so the same scores
    can be compared with posterior draws without repeating the search.
    """
    search = _search_grid(observations, physical_model, feh_init=feh_init,
                          distance=distance, model_kwargs=model_kwargs, feh_step=feh_step)
    model = _default_model(physical_model)

    def score(grid_model, point):
        return log_density(model, (observations, grid_model), search.model_kwargs, point)[0]

    scores = search.log_density_by_feh
    finite = np.flatnonzero(np.isfinite(scores.ravel()))
    ranked = finite[np.argsort(-scores.ravel()[finite], kind="stable")]
    value_and_grad = jax.jit(jax.value_and_grad(score, argnums=1))
    for index in ranked:
        k, i, j = np.unravel_index(index, scores.shape)
        name, value = search.axis_names[0], search.axes[0][i]
        if name == 'logage' and search.model_kwargs.get('linear_age', True):
            name, value = 'age', 10.**value / 1e9
        point = {name: jnp.asarray(value), 'eep': jnp.asarray(search.axes[1][j]),
                 'feh_init': jnp.asarray(search.feh_values[k]),
                 'distance': jnp.asarray(search.fixed_values['distance'])}
        value, gradient = value_and_grad(physical_model, point)
        if np.isfinite(value) and all(np.isfinite(v) for v in gradient.values()):
            initial = {key: float(value) for key, value in point.items()}
            return (initial, search) if return_grid else initial
    raise ValueError("grid initialization found no candidate with finite density and gradients; "
                     "check model bounds/data, change feh_init or distance via init_kwargs, "
                     "or supply init_strategy explicitly")


def _search_grid(observations, physical_model, *, feh_init=None, distance=None,
                 model_kwargs=None, feh_step=None):
    """Score the grid for initialization or diagnostics, without selecting a start."""
    model = _default_model(physical_model)
    options = dict(model_kwargs or {})
    bound = inspect.signature(model).bind(observations, physical_model, **options)
    bound.apply_defaults()
    settings = bound.arguments
    if physical_model.kind == "track":
        if not jax.config.x64_enabled:
            raise ValueError("Enable JAX x64 for track inference")
        axis_names, latent = ("initial_mass", "initial_feh", "eep"), "mass"
        limits = [(settings[lo], settings[hi]) for lo, hi in
                  (("massmin", "massmax"), ("fmin", "fmax"), ("eepmin", "eepmax"))]
        limits = [tuple(float(v) for v in _track_bounds(
            physical_model.grid.axis(name), lo, hi, name)[0])
            for name, (lo, hi) in zip(axis_names, limits)]
    else:
        axis_names, latent = ("age", "feh", "eep"), "logage"
        limits = [(settings[lo], settings[hi]) for lo, hi in
                  (("logamin", "logamax"), ("fmin", "fmax"), ("eepmin", "eepmax"))]
    axes = [physical_model.grid.axis(name) for name in axis_names]
    first = _initial_axis(axes[0], *limits[0], axis_names[0])
    eeps = _initial_axis(axes[2], *limits[2], "EEP")
    measured = dict(zip(observations.keys, observations.values))
    explicit_feh = feh_init is not None
    if not explicit_feh:
        feh_init = next((measured[key] for key in
                        ('feh', 'feh_surface', 'feh_photosphere') if key in measured), 0.)
    # Validate explicit/fixed FeH; scans may proceed without an automatic guess.
    _initial_axis(axes[1], *limits[1], "FeH")
    valid_feh = (not np.ndim(feh_init) and np.isfinite(feh_init)
                 and limits[1][0] < feh_init < limits[1][1]
                 and float(axes[1][0]) <= feh_init <= float(axes[1][-1]))
    if not valid_feh and (explicit_feh or feh_step is None):
        raise ValueError("feh_init must be finite, inside the open FeH prior bounds and within the grid; "
                         "choose a different feh_init for initialization")
    fehs = (np.array([feh_init], dtype=float) if feh_step is None else
            _feh_scan_axis(axes[1], *limits[1], feh_step, extra=feh_init if valid_feh else None))
    if distance is None:
        parallax = measured.get("parallax", 0.)
        if parallax <= 0:
            raise ValueError("grid search requires positive parallax; provide distance explicitly. "
                             "For NUTS initialization, alternatively supply init_strategy.")
        distance = 1. / parallax
    if np.ndim(distance) or not np.isfinite(distance) or distance <= 0:
        raise ValueError("initial distance must be a finite positive scalar in kpc")

    search_axes = (first, eeps)
    search_bounds = tuple((max(lo, float(axis[0])), min(hi, float(axis[-1])))
                          for axis, (lo, hi) in ((axes[0], limits[0]), (axes[2], limits[2])))
    first, eeps = (jnp.asarray(a.ravel()) for a in np.meshgrid(first, eeps, indexing="ij"))
    if physical_model.kind == "isochrone" and settings["linear_age"]:
        latent, first = "age", 10.**first / 1e9
    points = {latent: first, "eep": eeps,
              "feh_init": jnp.full_like(first, fehs[0]),
              "distance": jnp.full_like(first, distance)}

    def score(grid_model, point):
        return log_density(model, (observations, grid_model), options, point)[0]

    # Pass the large grid as a dynamic PyTree, not an embedded JIT constant.
    # Evaluate one FeH plane at a time, keeping intermediate model arrays bounded.
    evaluate = jax.jit(jax.vmap(score, in_axes=(None, 0)))
    scores = []
    for feh in fehs:
        points['feh_init'] = jnp.full_like(first, feh)
        scores.append(np.asarray(evaluate(physical_model, points)))
    shape = tuple(len(axis) for axis in search_axes)
    by_feh = np.stack(scores).reshape((len(fehs), *shape))
    supported = np.where(np.isfinite(by_feh), by_feh, -np.inf)
    profile = supported.max(axis=0)
    best_feh = np.where(np.isfinite(profile), fehs[supported.argmax(axis=0)], np.nan)
    fixed = dict(distance=float(distance))
    if feh_step is None:
        fixed['feh_init'] = float(fehs[0])
    return InitialGridSearch(
        axis_names=('mass' if physical_model.kind == 'track' else 'logage', 'eep'),
        axes=search_axes, bounds=search_bounds,
        log_density=profile, fixed_values=fixed,
        observations=observations, physical_model=physical_model, model_kwargs=options,
        feh_values=fehs, feh_step=None if feh_step is None else float(feh_step),
        log_density_by_feh=by_feh, best_feh=best_feh,
    )


def make_init_strategy(observations):
    """Retain the historical initialization rule, without fitting observations.

    Distance starts at 1/parallax (kpc for mas) for a positive parallax, otherwise
    8 kpc. Other observations are passed to init_to_value under their own names;
    only names matching latent sites set a start. In particular, surface ``feh``
    does not initialize ``feh_init``. Unspecified sites use NumPyro's fallback.
    With no parallax, supply an explicit init_strategy to the NUTS entry point.
    """
    if "parallax" not in observations.keys:
        raise ValueError(
            "default initialization requires parallax; supply init_strategy explicitly")
    parallax = np.asarray(observations.values)[
        np.asarray(observations.keys) == "parallax"]
    with np.errstate(divide="ignore"):
        init_dist = np.where(parallax > 0, 1. / parallax, 8).item()
    initdict = {"distance": init_dist}
    initdict.update((key, val) for key, val in zip(observations.keys, observations.values)
                    if key != "parallax")
    return init_to_value(values=initdict)


def run_nuts(model, rng_key, *, model_args=(), model_kwargs=None,
             num_warmup=5000, num_samples=5000, num_chains=4,
             target_accept_prob=0.98, dense_mass=True, max_tree_depth=10,
             init_strategy=None, chain_method="parallel", progress_bar=True,
             extra_fields=("potential_energy", "accept_prob",
                           "num_steps", "adapt_state.step_size"),
             init_params=None):
    """Run NUTS and return its MCMC object, preserving the chain dimension.

    Scientific options belong in model_kwargs; sampler options are explicit.
    Four chains run in parallel by default. NumPyro warns and falls back to
    sequential execution when there are too few devices. Configure CPU devices
    in the calling program before JAX initializes; this function does not change
    the device configuration. chain_method can also be sequential or vectorized.
    An omitted init_strategy uses NumPyro's default here; MistFit supplies a
    grid-search start by default, or make_init_strategy with grid_init=False.
    No samples or sampler state are written back to the model/fitter. Run
    metadata is attached to the returned MCMC for portable saving. By default,
    collect potential energy, acceptance probability, leapfrog count and step
    size, plus NumPyro's divergence flag. Explicit extra_fields=() keeps only
    that flag. These are stored diagnostics, not changes to the sampling target.
    """
    options = dict(model_kwargs or {})
    model_args = tuple(model_args)
    record = _new_run_record("nuts", model, model_args, options)
    record["random"] = {"key_data": np.asarray(random.key_data(rng_key)).tolist(),
                        "implementation": str(random.key_impl(rng_key))}
    record["sampling"] = dict(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                              target_accept_prob=target_accept_prob, dense_mass=dense_mass,
                              max_tree_depth=max_tree_depth, chain_method_requested=chain_method,
                              extra_fields=tuple(extra_fields))
    record["initialization"] = {"strategy": _json_value(init_strategy), "init_params": _json_value(init_params),
                                "default_if_omitted": "numpyro.infer.init_to_uniform"}
    kernel_options = dict(target_accept_prob=target_accept_prob, dense_mass=dense_mass,
                          max_tree_depth=max_tree_depth)
    if init_strategy is not None:
        kernel_options["init_strategy"] = init_strategy
    kernel = numpyro.infer.NUTS(model, **kernel_options)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                              num_chains=num_chains, chain_method=chain_method,
                              progress_bar=progress_bar)
    mcmc.run(rng_key, *model_args, extra_fields=extra_fields, init_params=init_params,
             **options)
    record["sampling"]["chain_method_used"] = mcmc.chain_method
    if isinstance(mcmc.last_state.z, dict):
        record["latent_names"] = list(mcmc.last_state.z)
    _finish_run_record(mcmc, record)
    return mcmc


def _gamma_quantile(probability, *, concentration, rate):
    """CPU prior callback; no differentiation or TFP is needed by MultiNest."""
    from scipy.special import gammaincinv

    dtype = np.dtype(jnp.result_type(probability, float))
    probability = np.asarray(probability, dtype=dtype)
    limits = np.finfo(dtype)
    probability = np.clip(probability, limits.tiny, 1. - limits.eps)
    return gammaincinv(concentration, probability) / rate


def _multinest_prior_overrides(model, model_args, options, site_overrides):
    """Supply Gamma quantiles through the runner's public override interface."""
    overrides = dict(site_overrides or {})
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, random.PRNGKey(0)))
    for name, site in trace.get_trace(*model_args, **options).items():
        if site["type"] != "sample" or site["is_observed"] or name in overrides:
            continue
        distribution = site["fn"]
        while isinstance(distribution, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
            distribution = distribution.base_dist
        if isinstance(distribution, dist.Gamma):
            overrides[name] = partial(_gamma_quantile,
                                       concentration=np.array(distribution.concentration, copy=True),
                                       rate=np.array(distribution.rate, copy=True))
    return overrides


def run_multinest(model, *, outputfiles_basename, model_args=(), model_kwargs=None,
                  site_overrides=None, **run_kwargs):
    """Run the optional MultiNestRunner with the same NumPyro probability model.

    Returns the runner, including its result and summary/deterministic helpers.
    run_kwargs are forwarded to runner.run (e.g. n_live_points and resume).
    outputfiles_basename is a filename prefix, e.g. "chains/star-", not a
    directory. Short relative prefixes avoid older MultiNest filename limits.
    Requires numpyro-inferutils plus PyMultiNest and its native MultiNest library;
    none are imported by the NUTS or prediction paths.
    Gamma sites (including the default distance prior) use SciPy's inverse CDF
    in the CPU prior callback, without a TensorFlow Probability dependency.
    Explicit site_overrides take precedence over these automatic transforms.
    """
    try:
        from numpyro_inferutils.experimental import MultiNestRunner
    except ImportError as exc:
        raise ImportError(
            "MultiNest requires numpyro-inferutils with experimental.MultiNestRunner support") from exc
    model_args, options = tuple(model_args), dict(model_kwargs or {})
    record = _new_run_record("multinest", model, model_args, options)
    site_overrides = _multinest_prior_overrides(model, model_args, options, site_overrides)
    runner = MultiNestRunner(model, model_args=model_args,
                             model_kwargs=options, site_overrides=site_overrides)
    # Keep the user's short relative prefix for the native call. Expanding it
    # to an absolute path can exceed MultiNest's 100-character filename buffer.
    outputfiles_basename = os.path.expanduser(str(outputfiles_basename))
    absolute_prefix = os.path.abspath(outputfiles_basename)
    if not outputfiles_basename or outputfiles_basename.endswith(os.sep):
        absolute_prefix = absolute_prefix.rstrip(os.sep) + os.sep
    Path(os.path.dirname(outputfiles_basename)).mkdir(
        parents=True, exist_ok=True)
    bound = inspect.signature(runner.run).bind(
        outputfiles_basename=outputfiles_basename, **run_kwargs)
    bound.apply_defaults()
    record["sampling"] = _json_value(dict(bound.arguments))
    # Record the runner's resolved defaults as well as the original options.
    record["sampling"]["resolved_n_live_points"] = (run_kwargs.get("n_live_points")
                                                    if run_kwargs.get("n_live_points") is not None else max(run_kwargs.get("use_num_live_points_factor", 25) * max(runner.ndim, 1), 100))
    record["sampling"]["resolved_sampling_efficiency"] = (run_kwargs.get("sampling_efficiency")
                                                          if run_kwargs.get("sampling_efficiency") is not None else (0.03 if run_kwargs.get("const_efficiency_mode", False) else 0.8))
    # PyMultiNest: -1 uses the clock.
    record["sampling"]["seed"] = run_kwargs.get("seed", -1)
    record["site_overrides"] = _json_value(site_overrides)
    record["native_output_prefix"] = absolute_prefix
    runner.run(outputfiles_basename=outputfiles_basename, **run_kwargs)
    # Python analysis and archiving can safely use full paths, and still work
    # if the caller subsequently changes its working directory.
    runner.outputfiles_basename = absolute_prefix
    record["native_files"] = {path.name: _file_hash(
        path) for path in _native_files(absolute_prefix)}
    record["evidence"] = {key: _json_value(
        runner.result.get(key)) for key in ("logZ", "logZerr")}
    _finish_run_record(runner, record)
    return runner
