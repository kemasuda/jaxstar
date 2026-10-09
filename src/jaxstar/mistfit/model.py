"""Deterministic MIST predictions, independent of observations and priors."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxstar.grid import GridResult, RectilinearGrid
from ._data import _MistGridData
from ._track_interpolation import track_support
from .mistgrid.isochrone import load_isochrones
from .mistgrid.tracks import load_tracks


__all__ = ["MistModel"]

# These NPZ fields contain absolute magnitudes. Public unsuffixed names retain
# MistFit's apparent-magnitude convention; *_abs exposes the stored quantity.
_MAGNITUDES = (
    "jmag", "hmag", "kmag", "gmag2", "bpmag2", "rpmag2",
    "gmag3", "bpmag3", "rpmag3", "umag", "bmag", "vmag", "rmag", "imag",
)
_ALIASES = {
    "mass_init": "mass",
    "mass_current": "star_mass",
    "feh_surface": "feh_photosphere",
    "feh": "feh_photosphere",
}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class MistModel:
    """A reusable, immutable isochrone or evolutionary-track model.

    Normally construct with ``MistModel.from_isochrones(path)``. Direct
    ``MistModel(grid)`` accepts a RectilinearGrid for tests/custom grids; a
    private loader result also carries the kind and source provenance.
    Call with a parameter mapping containing ``logage`` (log10 years),
    ``feh_init`` (dex), and ``eep``. ``distance`` (kpc) is required only for
    apparent magnitudes, parallax (mas), or distance itself. Coordinates can
    be scalars or broadcastable arrays. All returned arrays share that shape.
    ``from_tracks(path)`` instead takes ``mass_init`` (solar masses),
    ``feh_init``, and ``eep``; logage is then an interpolated output. It reads
    a prebuilt track NPZ and requires JAX x64. Track Teff/radius/age
    are exponentiated AFTER interpolating their logarithms.

    ``keys`` is a name, an ordered sequence, or None for all available public
    outputs. The result is an immutable GridResult. Selection never changes
    model state; use a tuple/static argument when passing keys through JIT.
    Like SpecModel, the model is a PyTree with dynamic numerical grid leaves:
    prefer ``jax.jit(lambda model, p: model(p, keys=(...)))`` for large grids.

    ``mass_init``/``mass`` and ``mass_current``/``star_mass`` are in solar
    masses. ``feh_surface``/``feh``/``feh_photosphere`` is surface FeH in dex.
    ``age`` is Gyr, radius is solar radii, teff is K, and logg is log10 cgs.
    Unsuffixed photometric keys (e.g. kmag) are apparent magnitudes; *_abs
    keys are absolute magnitudes. No extinction is added. ``rho`` uses
    current mass / radius**3 in solar density units; ``rho_initial`` exposes
    the initial-mass expression used by the historical MistFit constraint.

    Values are linearly interpolated. ``dmdeep`` is the separately stored,
    centered-difference initial-mass derivative (solar masses per EEP), NOT
    autodiff of the mass interpolant. The model imposes no mass/age prior.
    Tracks expose the analogous centered ``dtdeep`` in Gyr per EEP, computed
    from native physical ages on loading, before applying inference bounds.

    Both modes use RectilinearGrid's inclusive endpoints and out-of-domain
    -inf fill. Output NaNs become -inf; other infinities are left unchanged.
    ``legacy_interpolation=True`` uses first-node spacing and disables input
    sanitization, preserving the old interior arithmetic for comparisons.
    It requires regular axes, can yield nonfinite gradients for nonfinite
    coordinates, and does not promise complete historical/trajectory parity.
    Set this immutable option at construction; it is static within JIT.
    Parallax and apparent magnitudes share one reciprocal of distance.
    """

    data: _MistGridData
    legacy_interpolation: bool

    def __init__(self, grid, *, legacy_interpolation=False):
        if isinstance(grid, RectilinearGrid):
            data = _MistGridData(grid)
        elif isinstance(grid, _MistGridData):
            data = grid
        else:
            raise TypeError("grid must be a RectilinearGrid or MIST loader result")
        grid = data.grid
        if not isinstance(legacy_interpolation, bool):
            raise TypeError("legacy_interpolation must be a bool")
        if data.kind == "track" and legacy_interpolation:
            raise ValueError("legacy_interpolation is only supported for isochrones")
        if legacy_interpolation and any(kind != "regular" for kind in grid.axis_kinds):
            raise ValueError("legacy_interpolation requires regular axes with at least two nodes")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "legacy_interpolation", legacy_interpolation)

    @classmethod
    def from_isochrones(cls, path=None, *, legacy_interpolation=False):
        """Load all NPZ fields and construct an isochrone prediction model.

        An explicit path must exist. If omitted, use the environment override
        or cached grid and generate on first use. Subsequent calls do no I/O.
        ``legacy_interpolation`` has the same meaning as in the constructor.
        """
        if not isinstance(legacy_interpolation, bool):
            raise TypeError("legacy_interpolation must be a bool")
        return cls(load_isochrones(path), legacy_interpolation=legacy_interpolation)

    @classmethod
    def from_tracks(cls, path):
        """Load a track NPZ with precomputed UBVRIplus photometry.

        An explicit path and JAX x64 are required. No automatic download or
        grid generation is performed. Subsequent calls do no I/O. Native
        scalar fields and all 14 bands are retained; *_abs denotes absolute
        magnitudes. The raw age_gyr field is linearly interpolated physical
        age; public age instead comes from interpolated logage, as in the
        validated track prototype.
        Partially populated grids retain per-field support: a prediction is
        -inf if any contributing corner lacks its required fields. Missing
        photometry does not invalidate available physical outputs.
        Native primary EEP metadata, when present, is retained for plotting.
        """
        return cls(load_tracks(path))

    @property
    def grid(self):
        """Numerical storage; masked track nodes contain finite placeholders.

        Use model(...) for predictions and support checks. Raw photometric
        fields are absolute magnitudes.
        """
        return self.data.grid

    @property
    def kind(self):
        """The source parameterization: 'isochrone' or 'track'."""
        return self.data.kind

    @property
    def primary_eeps(self):
        """Native primary EEP labels for track plots; empty when unavailable.

        These are the union of labels in the source tracks' metadata and can
        extend beyond a cropped grid. They do not change interpolation/priors.
        """
        return self.data.primary_eeps

    @property
    def input_keys(self):
        """Required intrinsic coordinates; distance is conditional on outputs."""
        return (("logage", "feh_init", "eep") if self.kind == "isochrone"
                else ("mass_init", "feh_init", "eep"))

    @jax.jit
    def eep_given_mass(self, logage, feh_init, mass_init):
        """Invert the initial-mass curve for scalar logage, FeH and mass.

        logage is log10(age/yr), feh_init is initial FeH in dex, and mass_init
        is initial mass in solar units. Use jax.vmap for batches. The model's
        interpolation option also applies to the mass curve. This preserves
        the historical local-neighbor lookup: finite increasing brackets are
        needed, no mass-range/monotonicity policy is added, and inverse upper
        endpoints or invalid cells may produce NaNs/nonfinite gradients.
        """
        if self.kind != "isochrone":
            raise ValueError("eep_given_mass requires an isochrone model")
        if any(jnp.ndim(value) != 0 for value in (logage, feh_init, mass_init)):
            raise ValueError("logage, feh_init, and mass_init must be scalars; use jax.vmap for batches")
        eep = self.grid.axis("eep")
        mass_curve = self._interpolate(
            {"age": logage, "feh": feh_init, "eep": eep}, ("mass",),
        )["mass"]
        return _interp_mass_to_eep(mass_init, mass_curve, eep)

    def _output_specs(self):
        specs = {name: ("field", (name,)) for name in self.grid.field_names}
        for name in _MAGNITUDES:
            if name in self.grid.field_names:
                specs[name] = ("apparent", (name,))
                specs[name + "_abs"] = ("field", (name,))
        for name, field in _ALIASES.items():
            if field in self.grid.field_names:
                specs[name] = ("field", (field,))
        for name, mass in (("rho", "star_mass"), ("rho_initial", "mass")):
            if mass in self.grid.field_names and "radius" in self.grid.field_names:
                specs[name] = ("density", (mass, "radius"))
        for name in (*self.input_keys, "distance"):
            specs[name] = ("coordinate", ())
        specs["age"] = ("age", ())
        specs["parallax"] = ("parallax", ())
        if self.kind == "track":
            specs.update({
                "mass": ("initial_mass", ()),
                "logt": ("field", ("logTeff",)),
                "logl": ("field", ("logL",)),
                "teff": ("exp10", ("logTeff",)),
                "radius": ("exp10", ("logR",)),
                "age": ("track_age", ("logage",)),
                "rho": ("track_density", ("star_mass", "logR")),
                "rho_initial": ("track_initial_density", ("logR",)),
                "feh_photosphere": ("field", ("cmd_surface_feh",)),
                "feh_surface": ("field", ("cmd_surface_feh",)),
                "feh": ("field", ("cmd_surface_feh",)),
            })
        return specs

    @property
    def output_keys(self):
        """All available public output names, independent of prior selections."""
        return tuple(self._output_specs())

    def _interpolate(self, coordinates, keys):
        return self.grid.interpolate(
            coordinates, keys=keys, legacy_interpolation=self.legacy_interpolation,
        )

    def __call__(self, params, *, keys=None):
        if not isinstance(params, Mapping):
            raise TypeError(f"params must be a mapping with {', '.join(self.input_keys)}")
        specs = self._output_specs()
        selected = tuple(specs) if keys is None else ((keys,) if isinstance(keys, str) else tuple(keys))
        if not selected:
            raise ValueError("at least one output key must be selected")
        if any(not isinstance(key, str) or not key for key in selected):
            raise ValueError("output keys must be nonempty strings")
        if len(set(selected)) != len(selected):
            raise ValueError("output keys must be unique")
        for key in selected:
            if key not in specs:
                raise KeyError(f"unknown MistModel output: {key}")
        required = list(self.input_keys)
        needs_parallax = any(specs[key][0] in ("apparent", "parallax") for key in selected)
        if needs_parallax or "distance" in selected:
            required.append("distance")
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"missing MistModel parameters: {', '.join(missing)}")
        values = dict(zip(required, jnp.broadcast_arrays(*(jnp.asarray(params[key]) for key in required))))
        field_keys = tuple(dict.fromkeys(field for key in selected for field in specs[key][1]))
        coordinates = ({"age": values["logage"], "feh": values["feh_init"], "eep": values["eep"]}
                       if self.kind == "isochrone" else
                       {"initial_mass": values["mass_init"], "initial_feh": values["feh_init"],
                        "eep": values["eep"]})
        raw = self._interpolate(coordinates, field_keys) if field_keys else GridResult((), ())
        support = (track_support(self.data, coordinates, field_keys)
                   if field_keys and self.data.validity is not None else None)
        parallax = 1. / values["distance"] if needs_parallax else None
        outputs = []
        for key in selected:
            operation, fields = specs[key]
            if operation == "coordinate":
                value = values[key]
            elif operation == "age":
                value = 10**values["logage"] / 1e9
            elif operation == "parallax":
                value = parallax
            elif operation == "density":
                # Match historical NaN replacement before derived arithmetic.
                mass, radius = (jnp.where(jnp.isnan(raw[f]), -jnp.inf, raw[f]) for f in fields)
                value = mass / radius**3
            elif operation == "apparent":
                value = raw[fields[0]] - 5 * jnp.log10(parallax) + 10
            elif operation == "initial_mass":
                value = values["mass_init"]
            elif operation == "exp10":
                value = 10 ** raw[fields[0]]
            elif operation == "track_age":
                value = 10 ** (raw["logage"] - 9)
            elif operation == "track_density":
                value = raw["star_mass"] / (10 ** raw["logR"])**3
            elif operation == "track_initial_density":
                value = values["mass_init"] / (10 ** raw["logR"])**3
            else:
                value = raw[fields[0]]
            if support is not None and fields:
                valid = support[fields[0]]
                for field in fields[1:]:
                    valid = valid & support[field]
                value = jnp.where(valid, value, -jnp.inf)
            outputs.append(jnp.where(jnp.isnan(value), -jnp.inf, value))
        return GridResult(selected, tuple(outputs))

    def tree_flatten(self):
        return (self.data,), self.legacy_interpolation

    @classmethod
    def tree_unflatten(cls, legacy_interpolation, children):
        model = object.__new__(cls)
        object.__setattr__(model, "data", children[0])
        object.__setattr__(model, "legacy_interpolation", legacy_interpolation)
        return model


def _interp_mass_to_eep(x, xp, fp):
    """Historical local linear inverse, including its NaN-aware neighbor choice."""
    idx_min = jnp.nanargmin(jnp.abs(xp - x))
    idx_add = jnp.where(x < xp[idx_min], 0, 1)
    idx_low, idx_upp = idx_min - 1 + idx_add, idx_min + idx_add
    weight_low = (xp[idx_upp] - x) / (xp[idx_upp] - xp[idx_low])
    weight_upp = 1. - weight_low
    return weight_low * fp[idx_low] + weight_upp * fp[idx_upp]
