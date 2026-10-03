"""Deterministic component physics on an offline log-uniform spectral library."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._data import _SpectralLibrary
from .broadening import default_broadening
from .sampling import _C_KMS, is_log_uniform


def _require(condition, message):
    """Fail clearly for concrete inputs and for data-dependent compiled checks."""
    if not isinstance(condition, jax.core.Tracer):
        if not bool(condition):
            raise ValueError(message)
        return

    def fail(valid):
        def raise_error(valid):
            if not bool(valid):
                raise ValueError(message)
        # Passing the condition also avoids false failures if an outer vmap
        # transforms cond into selection and evaluates both branches.
        jax.debug.callback(raise_error, valid)
        return jnp.int32(0)

    jax.lax.cond(condition, lambda _: jnp.int32(0), fail, operand=condition)


def _region_parameter(value, name, count):
    value = jnp.asarray(value)
    if value.ndim == 0:
        return jnp.broadcast_to(value, (count,))
    if value.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape ({count},), got {value.shape}")
    return value


def _doppler_factor(rv):
    """Frozen relativistic convention: positive effective RV moves lines redward."""
    return jnp.sqrt((1 + rv / _C_KMS) / (1 - rv / _C_KMS))


def _sample(wavelength, source_wave, flux):
    _require(jnp.all((wavelength[:, :1] >= source_wave[:, :1])
                     & (wavelength[:, -1:] <= source_wave[:, -1:])),
             "insufficient model wavelength coverage for broadening/RV/sampling; "
             "prepare wider spectral regions or reduce the requested range")
    # Explicit NaN sentinels are a second line of defense; never edge-fill.
    return jax.vmap(lambda out, wave, row: jnp.interp(out, wave, row, left=jnp.nan, right=jnp.nan))(
        wavelength, source_wave, flux)


def _sample_with_rv(wavelength, source_wave, flux, rv):
    """Apply the relativistic shift in centered interpolation coordinates.

    Multiplying absolute float32 wavelengths by a factor close to one loses
    small RV shifts. Relative to a reference wavelength L, interpolate at
    (wavelength - L) - L*(D-1) on D*(source_wave - L). This is exactly the
    same Doppler convention, with D-1 rationalized to avoid cancellation.
    """
    beta = rv / _C_KMS
    factor = _doppler_factor(rv)
    delta = 2 * beta / ((1 - beta) * (factor + 1))
    reference = source_wave[:, source_wave.shape[-1] // 2:source_wave.shape[-1] // 2 + 1]
    return _sample((wavelength - reference) - reference * delta[:, None],
                   (source_wave - reference) * factor[:, None], flux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class SpecModel:
    """Single-component physical spectrum, independent of observation/fit state.

    ``intrinsic(params, wavelength)``, ``broadened(...)`` and ``full(...)``
    respectively evaluate atmosphere only, atmosphere plus combined broadening,
    and broadening plus relativistic RV. Calling the model is exactly ``full``.
    Wavelengths are in the prepared library's Angstrom/medium convention, with
    shape (region, pixel); 1D input/output is allowed for one region only.

    Parameters are nested ordinary dict/PyTree data: one entry in ``components``
    contains scalar ``atmosphere`` coordinates, ``broadening`` with vsini/vmacro/
    u1/u2, and effective ``rv`` in km/s. ``instrument.resolving_power`` specifies
    Gaussian IP resolving power. Post-atmosphere parameters are scalar or one
    per region. Stages only require the parameters they physically use.

    ``vmax`` controls finite kernel support, as in frozen jaxspec. Prepared
    coverage must include that support and the requested RV. No model regridding
    or automatic padding is performed. A custom broadening callable follows
    ``default_broadening``'s array contract and returns its own safe wave/flux.
    To keep large arrays dynamic, pass the model as a PyTree argument to JIT.
    """

    spectra: _SpectralLibrary
    velocity_grid: object
    vmax: float
    broadening_operator: object

    def __init__(self, spectra, *, vmax=50., broadening_operator=None):
        if not isinstance(spectra, _SpectralLibrary):
            raise TypeError("spectra must be a spectral-library loader/preparer result")
        if not is_log_uniform(spectra):
            raise ValueError("SpecModel requires log-uniform fitting wavelengths; use "
                             "prepare_*(sampling='log', velocity_step=...) or "
                             "resample_spectral_grid before constructing the model")
        if not np.isscalar(vmax) or not np.isfinite(vmax) or vmax <= 0:
            raise ValueError("vmax must be positive finite km/s")
        operator = default_broadening if broadening_operator is None else broadening_operator
        if not callable(operator):
            raise TypeError("broadening_operator must be callable")
        wave = np.asarray(spectra.wavelength, dtype=np.float64)
        # Endpoint mean is stable even with quantized float32 wavelengths.
        dv = _C_KMS * (np.log(wave[:, -1]) - np.log(wave[:, 0])) / (wave.shape[-1] - 1)
        half = int(np.round(vmax / dv[0]))
        if half < 1:
            raise ValueError("model velocity sampling is too coarse for vmax; prepare a finer grid or increase vmax")
        if operator is default_broadening and wave.shape[-1] <= 2 * half + 1:
            raise ValueError("prepared wavelength coverage is too short for broadening support; prepare wider regions")
        velocity = jnp.asarray(dv[:, None] * np.arange(-half, half + 1), dtype=spectra.wavelength.dtype)
        for name, value in (("spectra", spectra), ("velocity_grid", velocity),
                            ("vmax", float(vmax)), ("broadening_operator", operator)):
            object.__setattr__(self, name, value)

    def _atmosphere(self, params):
        if not isinstance(params, Mapping) or "components" not in params:
            raise ValueError("params must contain a components tuple/list")
        components = params["components"]
        if not isinstance(components, (tuple, list)) or len(components) != 1:
            raise ValueError("Milestone 3a requires exactly one component")
        component = components[0]
        if not isinstance(component, Mapping) or "atmosphere" not in component:
            raise ValueError("the component must contain an atmosphere dict")
        atmosphere = component["atmosphere"]
        axes = self.spectra.grid.axis_names
        if not isinstance(atmosphere, Mapping) or set(atmosphere) != set(axes):
            raise ValueError(f"atmosphere must supply exactly the named axes {axes}")
        coordinates, valid = {}, jnp.asarray(True)
        for name in axes:
            value = jnp.asarray(atmosphere[name])
            if value.ndim != 0:
                raise ValueError(f"atmosphere parameter {name} must be scalar per component")
            axis = self.spectra.grid.axis(name)
            valid &= jnp.isfinite(value) & (value >= axis[0]) & (value <= axis[-1])
            coordinates[name] = value
        _require(valid, "atmosphere coordinates must be finite and inside the prepared grid")
        flux = self.spectra.grid.interpolate(coordinates)["flux"]
        _require(jnp.all(jnp.isfinite(flux)), "atmosphere interpolation produced nonfinite flux")
        return component, flux

    def _wavelength(self, wavelength):
        wavelength = jnp.asarray(wavelength)
        single = wavelength.ndim == 1 and self.spectra.wavelength.shape[0] == 1
        if single:
            wavelength = wavelength[None, :]
        if (wavelength.ndim != 2 or wavelength.shape[0] != self.spectra.wavelength.shape[0]
                or wavelength.shape[1] == 0):
            raise ValueError("wavelength must have shape (n_region, n_pixel); 1D is allowed for one region")
        _require(jnp.all(jnp.isfinite(wavelength) & (wavelength > 0))
                 & jnp.all(jnp.diff(wavelength, axis=-1) > 0),
                 "requested wavelengths must be positive finite and strictly increasing per region")
        return wavelength, single

    def _broaden(self, component, params, flux):
        count = self.spectra.wavelength.shape[0]
        if not isinstance(component.get("broadening"), Mapping):
            raise ValueError("broadened/full require component.broadening with vsini/vmacro/u1/u2")
        names = ("vsini", "vmacro", "u1", "u2")
        if set(component["broadening"]) != set(names):
            raise ValueError("broadening must supply exactly vsini, vmacro, u1, u2")
        broad = {name: _region_parameter(component["broadening"][name], name, count) for name in names}
        if not isinstance(params.get("instrument"), Mapping) or "resolving_power" not in params["instrument"]:
            raise ValueError("broadened/full require instrument.resolving_power")
        resolution = _region_parameter(params["instrument"]["resolving_power"], "resolving_power", count)
        _require(jnp.all(jnp.stack([jnp.isfinite(value) for value in broad.values()]))
                 & jnp.all(broad["vsini"] >= 0) & jnp.all(broad["vmacro"] >= 0)
                 & jnp.all(1 - broad["u1"] / 3 - broad["u2"] / 6 > 0)
                 & jnp.all((resolution > 0) & ~jnp.isnan(resolution)),
                 "broadening requires finite nonnegative speeds, finite limb darkening with "
                 "positive disk intensity, and positive resolving_power (infinity disables Gaussian IP)")
        if self.broadening_operator is default_broadening:
            _require(jnp.all(broad["vsini"] <= self.velocity_grid[:, -1]),
                     "vsini exceeds broadening support; increase SpecModel vmax")
        wave, values = self.broadening_operator(self.spectra.wavelength, flux,
            broadening=broad, resolving_power=resolution, velocity_grid=self.velocity_grid)
        wave, values = jnp.asarray(wave), jnp.asarray(values)
        if wave.ndim != 2 or wave.shape != values.shape or wave.shape[0] != count or wave.shape[-1] < 2:
            raise ValueError("broadening operator must return matching (n_region, n_pixel>=2) wavelength/flux arrays")
        _require(jnp.all(jnp.isfinite(wave) & (wave > 0)) & jnp.all(jnp.diff(wave, axis=-1) > 0)
                 & jnp.all(jnp.isfinite(values)), "broadening operator returned invalid wavelength or flux")
        return wave, values

    def intrinsic(self, params, wavelength):
        """Atmosphere interpolation and requested-wavelength sampling only."""
        wavelength, single = self._wavelength(wavelength)
        _, flux = self._atmosphere(params)
        result = _sample(wavelength, self.spectra.wavelength, flux)
        return result[0] if single else result

    def broadened(self, params, wavelength):
        """Combined stellar/Gaussian broadening and sampling; no RV."""
        wavelength, single = self._wavelength(wavelength)
        component, flux = self._atmosphere(params)
        wave, flux = self._broaden(component, params, flux)
        result = _sample(wavelength, wave, flux)
        return result[0] if single else result

    def full(self, params, wavelength):
        """Combined broadening, effective relativistic RV and sampling."""
        wavelength, single = self._wavelength(wavelength)
        component, flux = self._atmosphere(params)
        wave, flux = self._broaden(component, params, flux)
        if "rv" not in component:
            raise ValueError("full requires component.rv (effective line-of-sight km/s)")
        rv = _region_parameter(component["rv"], "rv", wave.shape[0])
        _require(jnp.all(jnp.isfinite(rv) & (jnp.abs(rv) < _C_KMS)), "rv must be finite with abs(rv) < c")
        result = _sample_with_rv(wavelength, wave, flux, rv)
        return result[0] if single else result

    __call__ = full

    def tree_flatten(self):
        return (self.spectra, self.velocity_grid), (self.vmax, self.broadening_operator)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        result = object.__new__(cls)
        for name, value in zip(("spectra", "velocity_grid", "vmax", "broadening_operator"), (*children, *metadata)):
            object.__setattr__(result, name, value)
        return result
