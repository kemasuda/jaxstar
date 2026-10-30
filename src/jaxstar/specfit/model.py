"""Deterministic component physics on an offline log-uniform spectral library."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._data import _SpectralLibrary
from .broadening import _default_kernels, default_broadening, double_gaussian_fourier
from .ip_calibration import _IP_PARAMETERS, _fixed_ip_values
from .sampling import _C_KMS, is_log_uniform


_STELLAR_KEYS = frozenset(("vsini", "vmacro", "q1", "q2", "u1", "u2", "rv"))
_SHARED_KEYS = frozenset(("resolving_power", "flux_weights", "dilution"))
# Native sites from the built-in single-star model that do not enter the
# physical spectrum. Keep this explicit so misspelled physical keys fail.
_INFERENCE_KEYS = frozenset((
    "sigma_constant", "sigma_continuum", "jitter", "gp_amplitude", "gp_scale",
    "mismatch_fraction", "mismatch_scale", "continuum_coefficients", "model_flux",
))


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


def _limb_darkening_from_q(q1, q2):
    """Shared physical conversion, independent of NumPyro and fit state."""
    q1, q2 = jnp.asarray(q1), jnp.asarray(q2)
    _require(jnp.all(jnp.isfinite(q1) & (q1 >= 0) & (q1 <= 1))
             & jnp.all(jnp.isfinite(q2) & (q2 >= 0) & (q2 <= 1)),
             "q1 and q2 must be finite in [0, 1]")
    u1 = 2 * jnp.sqrt(q1) * q2
    return u1, jnp.sqrt(q1) - u1


def _parameter_array(value, name, shapes):
    """Check the numerical input shape without requiring a physical stage."""
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric value") from exc
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric value")
    if array.shape not in shapes:
        raise ValueError(f"{name} must have shape in {shapes}, got {array.shape}")
    return array


def _check_parameter_keys(values, allowed, context):
    if not isinstance(values, Mapping):
        raise TypeError(f"{context} must be a mapping")
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unexpected keys in {context}: {sorted(unknown, key=repr)}")


def _region_parameter(value, name, count):
    value = jnp.asarray(value)
    if value.ndim == 0:
        return jnp.broadcast_to(value, (count,))
    if value.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape ({count},), got {value.shape}")
    return value


def _kernel_fwhm(velocity, kernel):
    """Measure one bounded half-maximum interval per region, without a fit."""
    peak = jnp.max(kernel, axis=-1, keepdims=True)
    _require(jnp.all(jnp.isfinite(kernel)) & jnp.all(peak > 0),
             "kernel FWHM requires finite kernels with positive peaks")
    half = peak / 2
    above = kernel >= half
    _require(jnp.all(~above[:, 0] & ~above[:, -1]),
             "kernel half-maximum crossings are outside the velocity grid; "
             "increase SpecModel vmax and check support convergence")
    rising = ~above[:, :-1] & above[:, 1:]
    falling = above[:, :-1] & ~above[:, 1:]
    _require(jnp.all(jnp.sum(rising, axis=-1) == 1)
             & jnp.all(jnp.sum(falling, axis=-1) == 1),
             "kernel FWHM is ambiguous: multiple intervals at half maximum")
    rows = jnp.arange(kernel.shape[0])

    def crossing(transitions):
        index = jnp.argmax(transitions, axis=-1)
        y0, y1 = kernel[rows, index], kernel[rows, index + 1]
        x0, x1 = velocity[rows, index], velocity[rows, index + 1]
        return x0 + (half[:, 0] - y0) * (x1 - x0) / (y1 - y0)

    return crossing(falling) - crossing(rising)


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


class SpectralDecomposition(NamedTuple):
    """JAX PyTree of component spectra and continuum-light accounting.

    Spectra always retain the region axis, including for 1D wavelength input.
    ``components`` and ``weighted_components`` have shape (N, region, pixel);
    ``total`` has shape (region, pixel). Weights/fractions have shape (N, region)
    and dilution has shape (region,). No observational or fitted-continuum state.
    """

    total: object
    components: object
    flux_weights: object
    stellar_fractions: object
    light_fractions: object
    dilution: object
    weighted_components: object


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class SpecModel:
    """Deterministic stellar composition, independent of observation/fit state.

    ``intrinsic(params, wavelength)``, ``broadened(...)`` and ``full(...)``
    respectively evaluate atmosphere only, atmosphere plus combined broadening,
    and broadening plus relativistic RV. Calling the model is exactly ``full``.
    Wavelengths are in the prepared library's Angstrom/medium convention, with
    shape (region, pixel); 1D input/output is allowed for one region only.

    Parameters are nested ordinary dict/PyTree data: each entry in ``components``
    contains scalar ``atmosphere`` coordinates, ``broadening`` with vsini/vmacro/
    u1/u2, and effective ``rv`` in km/s. ``instrument.resolving_power`` specifies
    Gaussian IP resolving power. Post-atmosphere parameters are scalar or one
    per region. Stages only require the parameters they physically use.

    Alternatively, ``fixed_ip`` is a mapping containing mu1_kms, mu2_kms,
    sigma1_kms, sigma2_kms and fraction2 (scalars or one value per region).
    It specifies (1-fraction2)*G(mu1, sigma1) + fraction2*G(mu2, sigma2), with
    unit-area Gaussians and positive shifts moving lines redward. Values are
    copied and the instrumental Fourier factor is computed at construction.
    Omit instrument parameters in this mode. No extra Gaussian R broadening
    is applied. A custom broadening_operator and fixed_ip are mutually exclusive.
    Each nonzero-weight Gaussian's abs(mu)+5*sigma must fit the velocity support;
    choose still wider support as needed for rotation/macroturbulence tails.

    All components use the same library. ``flux_weights`` are finite nonnegative
    relative stellar continuum fluxes, shape (N,) or (N, region), with a positive
    sum in every region. A tuple/list may mix scalar and per-region entries.
    They need not sum to one. ``dilution`` is scalar or per region, in [0, 1),
    and is the featureless fraction of TOTAL continuum light. Each stage returns
    ``d + (1-d) * sum(a_i*f_i) / sum(a_i)`` for normalized component spectra.
    One component defaults to weight 1; multiple components require weights.
    Dilution defaults to zero. ``decompose`` exposes the individual spectra and
    fractions. Component count is static under JIT; numerical values are dynamic.

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
    _fixed_ip: object
    _ip_fourier: object

    @classmethod
    def from_grid(cls, gridpath, wavelength, *, regions=None, library=None,
                  pattern="*.npz", wav_margin=3., pixels=None, velocity_step=None,
                  vmax=50., wavelength_medium="unknown", flux_kind="as_stored",
                  broadening_operator=None, fixed_ip=None):
        """Select spectral regions from a common NPZ or legacy NPZ directory.

        ``wavelength`` (Angstrom; 1D or region-by-pixel) is used only to select
        regions covering each full row plus ``wav_margin`` on both sides.
        ``regions`` labels the output rows; it does not select stored labels.
        Zero or multiple covering candidates fail before loading flux.

        For a common-format NPZ file, metadata identifies the library, wavelength
        medium and flux kind. Saved samples are retained unless ``pixels`` or
        ``velocity_step`` (km/s) requests log resampling. Without resampling,
        saved wavelengths must already be log-uniform. Explicit ``library``,
        non-'unknown' medium and non-'as_stored' flux kind must match metadata.
        Omitted ``regions`` retains the selected saved labels.

        For a directory, ``select_grid_paths`` selects legacy files matching
        ``pattern``. ``library`` chooses 'coelho' (default), 'bosz' or 'tlusty'.
        Specify exactly one of ``pixels`` or ``velocity_step``. Medium and flux
        kind are declarations passed to the legacy loader. Its shared-axis,
        dtype and pixel-count requirements still apply.

        Selection/resampling uses each selected region's full wavelength span,
        independently of detector pixel counts. The observation is not stored.
        Neither wavelength conversion nor flux normalization is performed.
        Setup happens once, outside JIT; no raw files or downloads are used.
        ``fixed_ip`` is passed to SpecModel in the selected region row order.
        """
        from .libraries import select_grid_paths, load_coelho, load_bosz, load_tlusty
        from .observation import _identifiers
        from .sampling import _sampling_mode, resample_spectral_grid
        from .storage import load_spectral_grid

        if wavelength is None:
            raise ValueError("wavelength is required to select model regions")
        path = Path(gridpath).expanduser()
        regions = None if regions is None else tuple(regions)
        resample = pixels is not None or velocity_step is not None
        if path.is_file():
            if resample:
                _sampling_mode("log", pixels, velocity_step)
            spectra = load_spectral_grid(path, wavelength=wavelength,
                                         wav_margin=wav_margin, regions=regions)
            for name, value, unspecified in (
                ("library", library, None),
                ("wavelength_medium", wavelength_medium, "unknown"),
                ("flux_kind", flux_kind, "as_stored"),
            ):
                stored = getattr(spectra, name)
                if value != unspecified and value != stored:
                    raise ValueError(f"{name}={value!r} conflicts with saved {name}={stored!r}")
        elif path.is_dir():
            loaders = {"coelho": load_coelho, "bosz": load_bosz, "tlusty": load_tlusty}
            library = "coelho" if library is None else library
            if library not in loaders:
                raise ValueError("library must be 'coelho', 'bosz' or 'tlusty'")
            _sampling_mode("log", pixels, velocity_step)
            paths = select_grid_paths(path, wavelength, wav_margin=wav_margin,
                                      pattern=pattern, regions=regions)
            regions = _identifiers(regions, "regions", len(paths))
            spectra = loaders[library](paths, regions=regions,
                                       wavelength_medium=wavelength_medium, flux_kind=flux_kind)
        else:
            raise FileNotFoundError(f"Spectral grid path does not exist: {path}")
        if resample:
            spectra = resample_spectral_grid(spectra, sampling="log", pixels=pixels,
                                             velocity_step=velocity_step)
        return cls(spectra, vmax=vmax, broadening_operator=broadening_operator, fixed_ip=fixed_ip)

    def __init__(self, spectra, *, vmax=50., broadening_operator=None, fixed_ip=None):
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
        if fixed_ip is not None and operator is not default_broadening:
            raise ValueError("fixed_ip cannot be combined with a custom broadening_operator")
        wave = np.asarray(spectra.wavelength, dtype=np.float64)
        # Endpoint mean is stable even with quantized float32 wavelengths.
        dv = _C_KMS * (np.log(wave[:, -1]) - np.log(wave[:, 0])) / (wave.shape[-1] - 1)
        half = int(np.round(vmax / dv[0]))
        if half < 1:
            raise ValueError("model velocity sampling is too coarse for vmax; prepare a finer grid or increase vmax")
        if operator is default_broadening and wave.shape[-1] <= 2 * half + 1:
            raise ValueError("prepared wavelength coverage is too short for broadening support; prepare wider regions")
        velocity = jnp.asarray(dv[:, None] * np.arange(-half, half + 1), dtype=spectra.wavelength.dtype)
        ip_values, ip_fourier = None, None
        if fixed_ip is not None:
            values = _fixed_ip_values(fixed_ip, len(wave))
            # Guard against wrapping even the IP itself across the FFT boundary.
            # Stellar broadening can require still more support; see the docs.
            for index, active in ((1, values["fraction2"] < 1), (2, values["fraction2"] > 0)):
                extent = np.abs(values[f"mu{index}_kms"]) + 5 * values[f"sigma{index}_kms"]
                if np.any(active & (extent > dv * half)):
                    raise ValueError("fixed_ip requires abs(mu) + 5*sigma within broadening support; "
                                     "increase SpecModel vmax")
            ip_values = tuple(jnp.asarray(values[name], dtype=velocity.dtype) for name in _IP_PARAMETERS)
            ip_fourier = jax.vmap(double_gaussian_fourier)(velocity, *ip_values)
        for name, value in (("spectra", spectra), ("velocity_grid", velocity),
                            ("vmax", float(vmax)), ("broadening_operator", operator),
                            ("_fixed_ip", ip_values), ("_ip_fourier", ip_fourier)):
            object.__setattr__(self, name, value)

    @property
    def fixed_ip(self):
        """Fixed IP values in region order, or None for resolving-power mode.

        Returns a fresh mapping of immutable JAX arrays so editing the mapping
        cannot invalidate the precomputed Fourier factor. Rebuild to change IP.
        """
        return None if self._fixed_ip is None else dict(zip(_IP_PARAMETERS, self._fixed_ip))

    def make_params(self, values):
        """Convert flat physical values or a completed point to nested params.

        For one star, pass a flat mapping of the grid's atmosphere coordinates,
        vsini/vmacro, q1/q2 or u1/u2, rv, and resolving_power. For SB-N, pass
        {"components": (star1, star2, ...), ...} with flat stellar mappings;
        resolving_power, flux_weights and dilution belong only in the outer
        mapping. All stars use this model's atmosphere grid. The result is the
        existing component/instrument dict accepted by every physical method.

        Atmosphere coordinates are scalar per star. Broadening, RV, resolving
        power and dilution are scalar or (n_region,). Weights are (N,) or
        (N, n_region); a tuple/list may mix scalar and per-region entries.
        Omit resolving_power with fixed_ip. A single star defaults to weight
        one and dilution zero at evaluation; composing multiple stars requires
        explicit weights. No epoch or posterior-draw axis is inferred.

        Both limb-darkening pairs must be complete when present. If both are
        supplied, u1/u2 must match the q1/q2 conversion (rtol=1e-6, atol=1e-7);
        the converted values are used. Remove the redundant pair before editing
        limb darkening in a completed point.

        The built-in inference sites sigma_constant, sigma_continuum, jitter,
        gp_amplitude, gp_scale, mismatch_fraction, mismatch_scale,
        continuum_coefficients and model_flux are ignored. Other unknown keys
        raise. Custom site names must be mapped explicitly to physical names.

        This checks keys, numeric shapes, limb-darkening consistency and IP
        mode, but does not fill missing values or run interpolation, broadening,
        a likelihood or a probabilistic model. Partial inputs are allowed:
        each physical method checks the values it needs, so kernel FWHM needs
        no atmosphere or RV. Physical range/coverage checks also remain in
        those methods. Values are copied into new dicts with JAX-array leaves;
        inputs are not mutated and JIT/autodiff are supported.
        """
        axes = self.spectra.grid.axis_names
        reserved = _STELLAR_KEYS | _SHARED_KEYS | _INFERENCE_KEYS | {"components"}
        if set(axes) & reserved:
            raise ValueError("make_params requires atmosphere axis names distinct "
                             "from stellar, shared and inference parameter names")
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        if "components" in values:
            _check_parameter_keys(values, _SHARED_KEYS | _INFERENCE_KEYS | {"components"},
                                  "outer parameters")
            stars = values["components"]
            if not isinstance(stars, (tuple, list)) or not stars:
                raise ValueError("components must be a nonempty tuple/list of flat mappings")
        else:
            _check_parameter_keys(values, set(axes) | _STELLAR_KEYS | _SHARED_KEYS | _INFERENCE_KEYS,
                                  "single-star parameters")
            stars = ({name: value for name, value in values.items() if name not in _SHARED_KEYS},)
        components = tuple(self._make_component(star) for star in stars)
        shapes = ((), (len(self.spectra.regions),))
        instrument = {}
        if "resolving_power" in values:
            if self._fixed_ip is not None:
                raise ValueError("omit resolving_power when SpecModel has fixed_ip")
            instrument["resolving_power"] = _parameter_array(
                values["resolving_power"], "resolving_power", shapes)
        params = {"components": components, "instrument": instrument}
        if "dilution" in values:
            params["dilution"] = _parameter_array(values["dilution"], "dilution", shapes)
        if "flux_weights" in values:
            weights, count = values["flux_weights"], len(components)
            if isinstance(weights, (tuple, list)):
                if len(weights) != count:
                    raise ValueError(f"flux_weights must have one entry per component ({count})")
                weights = tuple(_parameter_array(value, "flux_weights entry", shapes) for value in weights)
            else:
                weights = _parameter_array(weights, "flux_weights", ((count,), (count, shapes[1][0])))
            params["flux_weights"] = weights
        return params

    def _make_component(self, values):
        axes = self.spectra.grid.axis_names
        _check_parameter_keys(values, set(axes) | _STELLAR_KEYS | _INFERENCE_KEYS,
                              "flat component")
        atmosphere = {name: _parameter_array(values[name], name, ((),))
                      for name in axes if name in values}
        shapes = ((), (len(self.spectra.regions),))
        stellar = {name: _parameter_array(values[name], name, shapes)
                   for name in sorted(_STELLAR_KEYS) if name in values}
        for pair in (("q1", "q2"), ("u1", "u2")):
            if any(name in stellar for name in pair) and not all(name in stellar for name in pair):
                raise ValueError(f"supply both {pair[0]} and {pair[1]}")
        if "q1" in stellar:
            u1, u2 = _limb_darkening_from_q(stellar["q1"], stellar["q2"])
            if "u1" in stellar:
                _require(jnp.allclose(stellar["u1"], u1, rtol=1e-6, atol=1e-7)
                         & jnp.allclose(stellar["u2"], u2, rtol=1e-6, atol=1e-7),
                         "u1/u2 are inconsistent with q1/q2; supply one pair or matching values")
            stellar.update(u1=u1, u2=u2)
        broadening = {name: stellar[name] for name in ("vsini", "vmacro", "u1", "u2") if name in stellar}
        component = {}
        if atmosphere:
            component["atmosphere"] = atmosphere
        if broadening:
            component["broadening"] = broadening
        if "rv" in stellar:
            component["rv"] = stellar["rv"]
        return component

    def _components(self, params):
        if not isinstance(params, Mapping) or "components" not in params:
            raise ValueError("params must contain a components tuple/list")
        components = params["components"]
        if not isinstance(components, (tuple, list)) or not components:
            raise ValueError("components must be a nonempty tuple/list")
        return components

    def _atmosphere(self, component):
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
        return flux

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

    def _broadening_parameters(self, component, params):
        count = self.spectra.wavelength.shape[0]
        if not isinstance(component, Mapping) or not isinstance(component.get("broadening"), Mapping):
            raise ValueError("broadening requires component.broadening with vsini/vmacro/u1/u2")
        names = ("vsini", "vmacro", "u1", "u2")
        if set(component["broadening"]) != set(names):
            raise ValueError("broadening must supply exactly vsini, vmacro, u1, u2")
        broad = {name: _region_parameter(component["broadening"][name], name, count) for name in names}
        if self._fixed_ip is None:
            if not isinstance(params.get("instrument"), Mapping) or "resolving_power" not in params["instrument"]:
                raise ValueError("broadening requires instrument.resolving_power")
            resolution = _region_parameter(params["instrument"]["resolving_power"], "resolving_power", count)
        else:
            instrument = params.get("instrument", {})
            if not isinstance(instrument, Mapping) or instrument:
                raise ValueError("omit instrument parameters when SpecModel has fixed_ip")
            resolution = jnp.full((count,), jnp.inf)
        _require(jnp.all(jnp.stack([jnp.isfinite(value) for value in broad.values()]))
                 & jnp.all(broad["vsini"] >= 0) & jnp.all(broad["vmacro"] >= 0)
                 & jnp.all(1 - broad["u1"] / 3 - broad["u2"] / 6 > 0)
                 & jnp.all((resolution > 0) & ~jnp.isnan(resolution)),
                 "broadening requires finite nonnegative speeds, finite limb darkening with "
                 "positive disk intensity, and positive resolving_power (infinity disables Gaussian IP)")
        if self.broadening_operator is default_broadening:
            _require(jnp.all(broad["vsini"] <= self.velocity_grid[:, -1]),
                     "vsini exceeds broadening support; increase SpecModel vmax")
        return broad, resolution

    def _broaden(self, component, params, flux):
        count = self.spectra.wavelength.shape[0]
        broad, resolution = self._broadening_parameters(component, params)
        ip_options = {} if self._fixed_ip is None else {"ip_fourier": self._ip_fourier}
        wave, values = self.broadening_operator(self.spectra.wavelength, flux,
            broadening=broad, resolving_power=resolution, velocity_grid=self.velocity_grid, **ip_options)
        wave, values = jnp.asarray(wave), jnp.asarray(values)
        if wave.ndim != 2 or wave.shape != values.shape or wave.shape[0] != count or wave.shape[-1] < 2:
            raise ValueError("broadening operator must return matching (n_region, n_pixel>=2) wavelength/flux arrays")
        _require(jnp.all(jnp.isfinite(wave) & (wave > 0)) & jnp.all(jnp.diff(wave, axis=-1) > 0)
                 & jnp.all(jnp.isfinite(values)), "broadening operator returned invalid wavelength or flux")
        return wave, values

    def _component_spectrum(self, component, params, wavelength, stage):
        """The common physical path, evaluated once per component."""
        flux = self._atmosphere(component)
        wave = self.spectra.wavelength
        if stage != "intrinsic":
            wave, flux = self._broaden(component, params, flux)
        if stage == "full":
            if "rv" not in component:
                raise ValueError("full requires component.rv (effective line-of-sight km/s)")
            rv = _region_parameter(component["rv"], "rv", wave.shape[0])
            _require(jnp.all(jnp.isfinite(rv) & (jnp.abs(rv) < _C_KMS)), "rv must be finite with abs(rv) < c")
            return _sample_with_rv(wavelength, wave, flux, rv)
        return _sample(wavelength, wave, flux)

    def _fractions(self, params, count, dtype):
        regions = self.spectra.wavelength.shape[0]
        if "flux_weights" not in params:
            if count != 1:
                raise ValueError("multiple components require explicit flux_weights")
            weights = jnp.ones((1, regions), dtype=dtype)
        else:
            value = params["flux_weights"]
            if isinstance(value, (tuple, list)):
                if len(value) != count:
                    raise ValueError(f"flux_weights must have one entry per component ({count})")
                weights = jnp.stack([_region_parameter(item, "flux_weights entry", regions) for item in value])
            else:
                weights = jnp.asarray(value)
                if weights.shape == (count,):
                    weights = jnp.broadcast_to(weights[:, None], (count, regions))
                elif weights.shape != (count, regions):
                    raise ValueError(f"flux_weights must have shape ({count},) or ({count}, {regions})")
        sums = jnp.sum(weights, axis=0)
        _require(jnp.all(jnp.isfinite(weights) & (weights >= 0))
                 & jnp.all(jnp.isfinite(sums) & (sums > 0)),
                 "flux_weights must be finite nonnegative with positive finite sum in every region")
        dilution = _region_parameter(params.get("dilution", jnp.asarray(0, dtype=dtype)), "dilution", regions)
        _require(jnp.all(jnp.isfinite(dilution) & (dilution >= 0) & (dilution < 1)),
                 "dilution must be finite with 0 <= dilution < 1")
        fractions = weights / sums[None, :]
        return weights, fractions, (1 - dilution)[None, :] * fractions, dilution

    def _evaluate(self, params, wavelength, stage):
        wavelength, single = self._wavelength(wavelength)
        components = self._components(params)
        # Preserve the ordinary M3a path without composition allocations/checks.
        if len(components) == 1 and "flux_weights" not in params and "dilution" not in params:
            result = self._component_spectrum(components[0], params, wavelength, stage)
        else:
            _, _, fractions, dilution = self._fractions(params, len(components), self.spectra.wavelength.dtype)
            values = jnp.stack([self._component_spectrum(component, params, wavelength, stage)
                                for component in components])
            result = dilution[:, None] + jnp.sum(fractions[:, :, None] * values, axis=0)
        return result[0] if single else result

    def intrinsic(self, params, wavelength):
        """Compose atmosphere spectra sampled at wavelength; no broadening/RV."""
        return self._evaluate(params, wavelength, "intrinsic")

    def broadened(self, params, wavelength):
        """Compose combined stellar/instrument-broadened spectra; no RV."""
        return self._evaluate(params, wavelength, "broadened")

    def broadening_fwhm(self, params):
        """Return combined-kernel FWHM in km/s, shape (component, region).

        Use the model's velocity grid and the same kernels as spectral
        broadening, including rotation, radial-tangential macroturbulence and
        either the resolving-power IP or fixed_ip. No Gaussian fit or
        Gaussian-equivalent width is used. Only component.broadening and,
        without fixed_ip, instrument.resolving_power are required. Atmosphere,
        RV, flux weights and dilution do not enter these per-component widths.

        Half-maximum crossings are linearly interpolated on the sampled
        kernels. Exactly disabled broadening returns zero. For nonzero widths,
        resolve the kernel with sufficiently fine model velocity sampling and
        check convergence with vmax; broadening of intrinsic lines is excluded.
        Disconnected intervals at half maximum or unbracketed crossings raise
        an error. Asymmetric kernels with one half-maximum interval are valid.
        Custom broadening operators do not expose kernels and are unsupported.
        The method supports JIT and retains both output axes for a single star
        or region. It does not change any fit mask or choose a dilation factor.
        """
        if self.broadening_operator is not default_broadening:
            raise NotImplementedError(
                "broadening_fwhm requires the built-in broadening operator; "
                "custom broadening operators do not expose kernels")
        widths = []
        for component in self._components(params):
            broad, resolution = self._broadening_parameters(component, params)
            kernel = _default_kernels(broadening=broad, resolving_power=resolution,
                                      velocity_grid=self.velocity_grid, ip_fourier=self._ip_fourier)
            width = _kernel_fwhm(self.velocity_grid, kernel)
            if self._fixed_ip is None:
                disabled = (broad["vsini"] == 0) & (broad["vmacro"] == 0) & jnp.isinf(resolution)
                width = jnp.where(disabled, 0., width)
            widths.append(width)
        return jnp.stack(widths)

    def full(self, params, wavelength):
        """Compose combined-broadened, relativistic-RV-shifted spectra."""
        return self._evaluate(params, wavelength, "full")

    __call__ = full

    def decompose(self, params, wavelength, *, stage="full"):
        """Return component spectra and light accounting, retaining region axes.

        ``stage`` is one of intrinsic/broadened/full, and is static under JIT.
        ``total = dilution[:, None] + weighted_components.sum(axis=0)``.
        This shares the same per-component physical implementation as the stage
        methods; no broadening, interpolation or RV physics is duplicated.
        """
        if stage not in ("intrinsic", "broadened", "full"):
            raise ValueError("stage must be intrinsic, broadened or full")
        wavelength, _ = self._wavelength(wavelength)
        components = self._components(params)
        weights, stellar, light, dilution = self._fractions(params, len(components), self.spectra.wavelength.dtype)
        values = jnp.stack([self._component_spectrum(component, params, wavelength, stage)
                            for component in components])
        weighted = light[:, :, None] * values
        total = dilution[:, None] + jnp.sum(weighted, axis=0)
        return SpectralDecomposition(total, values, weights, stellar, light, dilution, weighted)

    def tree_flatten(self):
        return (self.spectra, self.velocity_grid, self._fixed_ip, self._ip_fourier), (self.vmax, self.broadening_operator)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        result = object.__new__(cls)
        for name, value in zip(("spectra", "velocity_grid", "_fixed_ip", "_ip_fourier",
                                "vmax", "broadening_operator"), (*children, *metadata)):
            object.__setattr__(result, name, value)
        return result
