"""Repository-only MIST track inference prototype; see benchmarks/README.md.

Values use multilinear interpolation. The age-prior Jacobian deliberately
uses separately interpolated centered differences of physical age, following
jaxstar's dmdeep convention, not autodiff of the interpolated log-age field.
No source downloads, isochrone inversion, or production MistFit changes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from jaxstar.grid import Field, RectilinearGrid
from validate_mist_bolometric_corrections import BANDS
from validate_mist_track_grid import DIMS


MAG_KEYS = ("umag", "bmag", "vmag", "rmag", "imag", "jmag", "hmag", "kmag",
            "gmag2", "bpmag2", "rpmag2", "gmag3", "bpmag3", "rpmag3")
SCALAR_KEYS = ("logage", "logTeff", "logg", "logL", "logR", "star_mass", "cmd_surface_feh")
LATENTS = ("mass", "feh_init", "eep", "distance")


def centered_age_derivative(age_gyr, eep):
    """dt/dEEP in Gyr/EEP on a uniform axis, with jaxstar endpoint copying.

    Only the last array dimension is differentiated. The full stored EEP
    range is used before any inference bounds are applied. No one-sided or
    autodiff derivative is substituted at either endpoint.
    """
    age_gyr, eep = np.asarray(age_gyr, dtype=np.float64), np.asarray(eep, dtype=np.float64)
    if eep.ndim != 1 or len(eep) < 3 or age_gyr.ndim == 0 or age_gyr.shape[-1] != len(eep):
        raise ValueError("At least three EEPs matching the final age dimension are required")
    spacing = np.diff(eep)
    if (not np.all(np.isfinite(eep)) or not np.all(spacing > 0)
            or not np.allclose(spacing, spacing[0], rtol=1e-12, atol=0)):
        raise ValueError("The jaxstar centered-difference convention requires uniform EEP spacing")
    if not np.all(np.isfinite(age_gyr) & (age_gyr > 0)) or not np.all(np.diff(age_gyr, axis=-1) > 0):
        raise ValueError("Native physical ages must be finite, positive, and increasing in EEP")
    middle = (age_gyr[..., 2:] - age_gyr[..., :-2]) / (2 * spacing[0])
    return np.concatenate((middle[..., :1], middle, middle[..., -1:]), axis=-1)


@jax.jit
def predict(grid, mass, feh_init, eep, distance):
    """Return MistFit-compatible observables; distance is kpc, age is Gyr."""
    fields = grid.interpolate(dict(zip(DIMS, (mass, feh_init, eep))))
    result = {"mass": mass, "age": 10 ** (fields["logage"] - 9), "logage": fields["logage"],
              "teff": 10 ** fields["logTeff"], "logt": fields["logTeff"],
              "logg": fields["logg"], "logl": fields["logL"], "radius": 10 ** fields["logR"],
              "star_mass": fields["star_mass"], "feh_photosphere": fields["cmd_surface_feh"],
              "feh": fields["cmd_surface_feh"], "dtdeep": fields["dtdeep"], "parallax": 1 / distance}
    apparent = fields["absolute_magnitudes"] + (5 * jnp.log10(distance) + 10)[..., None]
    result.update({name: apparent[..., i] for i, name in enumerate(MAG_KEYS)})
    return result


@jax.jit
def age_bounded_eep(age_curves, eep_axis, mass, feh_init, eep_bounds, logage_bounds):
    """Invert the same monotone age polyline only to bound the EEP support.

    This changes the sampler's support transform, not the chosen prior
    derivative. The conditional Uniform's width is canceled in logjac.
    """
    curve = age_curves.interpolate({"initial_mass": mass, "initial_feh": feh_init})["logage"]
    endpoints = jnp.interp(jnp.asarray(logage_bounds), curve, eep_axis)
    lo, hi = jnp.clip(endpoints, eep_bounds[0], eep_bounds[1])
    # If no age/EEP overlap exists, keep the transform finite. model()'s
    # age-support factor then rejects the point; never silently extrapolate.
    valid = lo < hi
    return jnp.where(valid, lo, eep_bounds[0]), jnp.where(valid, hi, eep_bounds[1]), valid


class MistTrackGrid:
    """Load the small validated track NPZ and add centered dt/dEEP fields."""

    def __init__(self, path):
        if not jax.config.x64_enabled:
            raise ValueError("Enable JAX x64 before loading the track inference grid")
        with np.load(path, allow_pickle=False) as data:
            if tuple(data["bands"]) != BANDS:
                raise ValueError("Unexpected photometric band order")
            self.axes = tuple(np.array(data[name], dtype=np.float64) for name in DIMS)
            shape = tuple(len(axis) for axis in self.axes)
            self.native = {name: np.array(data[name], dtype=np.float64)
                           for name in (*SCALAR_KEYS, "age_gyr", "absolute_magnitudes")}
        for name, values in self.native.items():
            expected = shape + ((len(BANDS),) if name == "absolute_magnitudes" else ())
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"Incomplete or nonfinite native field: {name}")
        if not np.allclose(10 ** (self.native["logage"] - 9), self.native["age_gyr"], rtol=2e-13, atol=0):
            raise ValueError("Native physical ages and log-age field disagree")
        self.native["dtdeep"] = centered_age_derivative(self.native["age_gyr"], self.axes[2])
        fields = {name: Field(self.native[name], DIMS) for name in (*SCALAR_KEYS, "dtdeep")}
        fields["absolute_magnitudes"] = Field(self.native["absolute_magnitudes"], DIMS,
                                               payload_dims=("band",))
        self.grid = RectilinearGrid(axes=dict(zip(DIMS, self.axes)), fields=fields, fill_value=np.nan)
        self.age_curves = RectilinearGrid(axes=dict(zip(DIMS[:2], self.axes[:2])),
            fields={"logage": Field(self.native["logage"], DIMS[:2], payload_dims=("eep",))},
            fill_value=np.nan)

    def values(self, mass, feh_init, eep, distance):
        return predict(self.grid, mass, feh_init, eep, distance)


class MistTrackFit:
    """Minimal four-coordinate NumPyro model with the existing Gaussian likelihood.

    Uniform initial-mass/initial-FeH priors and a flat physical-age (or
    log-age) target are restricted to the available grid and age window.
    The separately estimated Jacobian approximates that target as in jaxstar;
    exact transformed-density consistency with the age polyline is not claimed.
    No correction for varying mass coverage forces a flat age marginal
    (flat_age_marginal). Extinction, gyrochronology, and binaries are omitted.
    Coordinate bounds must lie wholly inside the grid.
    """

    def __init__(self, grid, obskeys, obsvals, obserrs, *, linear_age=True,
                 logage_bounds=(8., 10.14), mass_bounds=None, feh_bounds=None,
                 eep_bounds=None, dist_scale=1.35):
        self.mg = grid
        self.obskeys = tuple(obskeys)
        vals, errs = np.asarray(obsvals, dtype=float), np.asarray(obserrs, dtype=float)
        supported = {*MAG_KEYS, "teff", "logt", "logg", "logl", "radius", "star_mass",
                     "mass", "age", "logage", "feh", "feh_photosphere", "parallax"}
        if not self.obskeys or len(set(self.obskeys)) != len(self.obskeys) or not set(self.obskeys) <= supported:
            raise ValueError("Observations require unique supported names")
        if vals.shape != (len(self.obskeys),) or errs.shape != vals.shape:
            raise ValueError("Observation values and errors must match obskeys")
        if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(errs) & (errs > 0)):
            raise ValueError("Observations must be finite and uncertainties positive")
        self.obsvals, self.obserrs = jnp.asarray(vals), jnp.asarray(errs)
        self.bounds = []
        for axis, requested in zip(grid.axes, (mass_bounds, feh_bounds, eep_bounds)):
            lo, hi = (float(axis[0]), float(axis[-1])) if requested is None else requested
            if not (np.isfinite(lo) and np.isfinite(hi) and axis[0] <= lo < hi <= axis[-1]):
                raise ValueError("Inference bounds must lie inside the available track grid")
            self.bounds.append((float(lo), float(hi)))
        lo, hi = logage_bounds
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError("logage_bounds must be finite and increasing")
        self.logage_bounds = (float(lo), float(hi))
        self.linear_age = bool(linear_age)
        if not np.isfinite(dist_scale) or dist_scale <= 0:
            raise ValueError("dist_scale must be positive")
        self.dist_scale = float(dist_scale)

    def model(self, nodata=False):
        mass = numpyro.sample("mass", dist.Uniform(*self.bounds[0]))
        feh_init = numpyro.sample("feh_init", dist.Uniform(*self.bounds[1]))
        eep_lo, eep_hi, overlap = age_bounded_eep(self.mg.age_curves, self.mg.axes[2],
            mass, feh_init, self.bounds[2], self.logage_bounds)
        eep = numpyro.sample("eep", dist.Uniform(eep_lo, eep_hi))
        numpyro.deterministic("eep_lower", eep_lo)
        numpyro.deterministic("eep_upper", eep_hi)
        distance = numpyro.sample("distance", dist.Gamma(3, rate=1 / self.dist_scale))
        params = self.mg.values(mass, feh_init, eep, distance)
        for name, value in params.items():
            if name != "mass":
                numpyro.deterministic(name, value)

        # Same independent Normal likelihood, photospheric FeH, parallax and
        # apparent-magnitude conventions as MistFit.model.
        if not nodata:
            predicted = jnp.stack([params[name] for name in self.obskeys])
            numpyro.sample("obs", dist.Normal(predicted, self.obserrs), obs=self.obsvals)

        lo, hi = self.logage_bounds
        dt = params["dtdeep"]  # precomputed central differences in Gyr/EEP
        valid = (overlap & (params["logage"] >= lo) & (params["logage"] <= hi)
                 & jnp.isfinite(dt) & (dt > 0))
        logjac = jnp.log(jnp.where(valid, dt, 1.))
        if self.linear_age:
            logjac -= np.log(10 ** (hi - 9) - 10 ** (lo - 9))
        else:
            logjac -= jnp.log(params["age"] * np.log(10)) + np.log(hi - lo)
        # Cancel the auxiliary conditional Uniform(EEP) density, including its
        # mass/FeH-dependent width. Do not impose an extra age-span prior.
        logjac += jnp.log(eep_hi - eep_lo)
        numpyro.factor("logjac", jnp.where(valid, logjac, -jnp.inf))
