"""Frozen grid adapters before their removal from production (2026-10-06).

Keep their interpolation, inverse lookup and mutable-key behavior unchanged.
Only imports/path handling may be adjusted; production must not depend on this
reference. Tests and benchmarks use it to check the old numerical conventions.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.ndimage import map_coordinates as mapc
from functools import partial
from jaxstar.grid import Field, RectilinearGrid
from jaxstar.mistfit.mistgrid.paths import resolve_mistgrid_path

def check_mistgrid_path():
    """Reference runs only read an existing grid; never download one."""
    path = resolve_mistgrid_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def interp(x, xp, fp):
    """1D interpolation

        Args:
            x: x-coordinates at which interpolated values are evaluated
            xp: x-coordinates of the data
            fp: y-coordinates of the data

        Returns:
            array: linearly interpolated values

    """
    idx_min = jnp.nanargmin(jnp.abs(xp - x))
    idx_add = jnp.where(x < xp[idx_min], 0, 1)
    idx_low, idx_upp = idx_min - 1 + idx_add, idx_min + idx_add
    weight_low = (xp[idx_upp] - x) / (xp[idx_upp] - xp[idx_low])
    weight_upp = 1. - weight_low
    return weight_low * fp[idx_low] + weight_upp * fp[idx_upp]


def _iso_grid(data):
    """Load every stored field into the immutable numerical grid once."""
    axis_keys = {"age": "logagrid", "feh": "fgrid", "eep": "eepgrid"}
    return RectilinearGrid(
        axes={name: data[key] for name, key in axis_keys.items()},
        fields={
            name: Field(data[name], dims=("age", "feh", "eep"))
            for name in data.files if name not in axis_keys.values()
        },
    )


@partial(jit, static_argnames=("keys",))
def _named_grid_values(grid, age, feh, eep, *, keys):
    """Compile the immutable numerical grid rather than a mutable MistGridIso."""
    return grid.interpolate({"age": age, "feh": feh, "eep": eep}, keys=keys)


@jit
def _named_eep_given_mass(grid, age, feh, mass):
    if any(jnp.ndim(value) != 0 for value in (age, feh, mass)):
        raise ValueError("age, feh, and mass must be scalars; use jax.vmap for batches")
    eep = grid.axis("eep")
    mass_curve = grid.interpolate(
        {"age": age, "feh": feh, "eep": eep}, keys="mass"
    )["mass"]
    return interp(mass, mass_curve, eep)


@partial(jit, static_argnames=("keys",))
def _named_values_given_mass(grid, age, feh, mass, *, keys):
    eep = _named_eep_given_mass(grid, age, feh, mass)
    return _named_grid_values(grid, age, feh, eep, keys=keys)


class MistGridIso:
    """class to store model grid """

    def __init__(self, path=None):
        """initialization

            Args:
                path: path to an existing npz grid file. If omitted, use the
                    environment override or user cache, downloading on first use.

        """
        if path is None:
            path = check_mistgrid_path()
        self.dgrid = np.load(path)
        self.a0, self.da = self.dgrid['logagrid'][0], np.diff(self.dgrid['logagrid'])[
            0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[
            0]
        self.eep0, self.deep = self.dgrid['eepgrid'][0], np.diff(self.dgrid['eepgrid'])[
            0]
        self.amin, self.amax = np.min(
            self.dgrid['logagrid']), np.max(self.dgrid['logagrid'])
        self.fmin, self.fmax = np.min(
            self.dgrid['fgrid']), np.max(self.dgrid['fgrid'])
        self.eepmin, self.eepmax = np.min(
            self.dgrid['eepgrid']), np.max(self.dgrid['eepgrid'])
        self.keys = None

    def set_keys(self, keys):
        """set keys for the stellar parameters to be evaluated

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
                list: interpolated values for the parameters in self.keys

        """
        aidx = (age - self.a0) / self.da
        fidx = (feh - self.f0) / self.df
        eepidx = (eep - self.eep0) / self.deep
        idxs = [aidx, fidx, eepidx]
        return [mapc(self.dgrid[key], idxs, order=1, cval=-jnp.inf) for key in self.keys]

    @partial(jit, static_argnums=(0,))
    def eep_given_mass(self, age, feh, mass):
        """compute EEP for given age, feh, mass

            Args:
                age: log10(stellar age in yr)
                feh: metallicity (dex)
                mass: stellar mass (solar unit)

            Returns:
                tuple: equivalent evolutionary point, age index, feh index

        """
        aidx = (age - self.a0) / self.da
        fidx = (feh - self.f0) / self.df
        eepidx_grid = (self.dgrid['eepgrid'] - self.eep0) / self.deep
        idxs_grid = [aidx, fidx, eepidx_grid]

        mass_grid = mapc(self.dgrid['mass'], idxs_grid, order=1, cval=-jnp.inf)
        eep = interp(mass, mass_grid, jnp.array(
            self.dgrid['eepgrid']))  # jnp.interp fails
        return eep, aidx, fidx

    @partial(jit, static_argnums=(0,))
    def values_given_mass(self, age, feh, mass):
        """ compute stellar parameters for given age, feh, and mass

            Args:
                age: log10(stellar age in yr)
                feh: metallicity (dex)
                mass: stellar mass (solar unit)

            Returns:
                list: interpolated values for the parameters in self.keys

        """
        eep, aidx, fidx = self.eep_given_mass(age, feh, mass)
        eepidx = (eep - self.eep0) / self.deep
        idxs = [aidx, fidx, eepidx]

        return [mapc(self.dgrid[key], idxs, order=1, cval=-jnp.inf) for key in self.keys]


class NamedMistGridIso:
    """MIST interpolation with named outputs and per-call selection.

    All fields are loaded during initialization. The three NPZ entries
    logagrid, fgrid, and eepgrid supply the axes; all remaining entries are
    scalar fields with dimensions (age, feh, eep). Forward interpolation includes
    axis endpoints and returns -inf for out-of-range coordinates. Nonfinite
    field handling follows RectilinearGrid. Mass-to-EEP lookup uses the existing
    interp() helper on a mass curve generated by the named grid.
    """

    def __init__(self, path=None):
        if path is None:
            path = check_mistgrid_path()
        with np.load(path) as data:
            self._grid = _iso_grid(data)

    @property
    def grid(self):
        """The immutable, all-field RectilinearGrid for use with MistModel."""
        return self._grid

    def values(self, age, feh, eep, *, keys=None):
        """Return a GridResult for all fields or the requested ordered keys.

        age is log10(age in years). Coordinates may be scalars or broadcastable
        arrays. keys accepts a name, a sequence of names, or None for all fields;
        the selection is static within a JIT trace.
        """
        if keys is not None and not isinstance(keys, str):
            keys = tuple(keys)
        return _named_grid_values(self._grid, age, feh, eep, keys=keys)

    def eep_given_mass(self, age, feh, mass):
        """Return EEP alone for scalar log10(age in years), feh, and mass.

        mass is the initial stellar mass in solar units. Use jax.vmap for
        multiple queries. The existing interp() neighbor selection is retained:
        a finite, increasing local mass bracket is needed. Invalid regions,
        flat/nonmonotonic curves and endpoints are not guaranteed; no mass range
        check is added. Gradients can be nonfinite with invalid field values.

        On isochrones, near-flat initial-mass curves amplify roundoff into large
        EEP/output differences, especially in float32. Addressing this numerical
        sensitivity is deferred until an evolutionary-track grid is considered.
        """
        return _named_eep_given_mass(self._grid, age, feh, mass)

    def values_given_mass(self, age, feh, mass, *, keys=None):
        """Return a GridResult after scalar mass-to-EEP lookup.

        keys follows values(); mass is always used for inverse lookup even
        when it is not selected. The restrictions of eep_given_mass() apply.
        """
        if keys is not None and not isinstance(keys, str):
            keys = tuple(keys)
        return _named_values_given_mass(self._grid, age, feh, mass, keys=keys)
