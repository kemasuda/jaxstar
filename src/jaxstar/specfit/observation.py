"""Immutable measured spectral arrays, independent of model/fit parameters."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


def _array(value):
    # JAX arrays are already immutable. Copy host inputs so later caller edits
    # cannot change an observation, without applying JAX dtype canonicalization.
    if isinstance(value, jax.Array):
        return value
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _numeric_array(value, name):
    array = _array(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    return array


def _exclusion_mask(value, wavelength):
    if value is None:
        if isinstance(wavelength, jax.Array):
            return jnp.zeros_like(wavelength, dtype=bool)
        return _array(np.zeros(wavelength.shape, dtype=bool))
    mask = _array(value)
    if mask.shape != wavelength.shape:
        raise ValueError("mask shape must match wavelength, flux and uncertainty")
    if mask.dtype.kind != "b":
        values = np.asarray(mask)
        if mask.dtype.kind not in "fiu" or not np.all((values == 0) | (values == 1)):
            raise ValueError("mask must be Boolean or contain only numeric 0/1 values")
        mask = mask.astype(bool)
        if isinstance(mask, np.ndarray):
            mask.setflags(write=False)
    return mask


def _identifiers(value, name, count):
    if value is None:
        return None
    labels = np.asarray(value, dtype=object)
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError(f"{name} must be a 1D sequence with one identifier per region ({count})")
    result = []
    for label in labels:
        if isinstance(label, str):
            result.append(label)
        elif isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_)):
            result.append(int(label))
        else:
            raise ValueError(f"{name} identifiers must be strings or integers")
    return tuple(result)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Observation:
    """Measured wavelength, flux, uncertainty and data-level exclusion mask.

    Arrays have identical, nonempty (pixel,) or (region, pixel) shapes. Wavelength
    is finite, positive and strictly increasing within each region, even at
    masked pixels. Its unit/medium must match the model when used for evaluation.
    Flux and uncertainty must be finite, and uncertainty positive, wherever
    ``mask`` is False. Masked flux/uncertainty values are retained as supplied.

    ``mask=True`` excludes a pixel; None supplies an all-False Boolean mask.
    Numeric 0/1 masks are accepted without inversion. This mask is observation
    data, separate from any later adjustable fitting mask.

    NumPy/list inputs become independent read-only NumPy arrays, preserving
    their dtypes. JAX arrays retain their dtype/device. Construction validates
    concrete data before JIT; the four arrays are dynamic PyTree leaves.
    JAX transformations follow the caller's precision configuration; this class
    never enables x64. Region/order tuples and a nonempty exposure string are
    optional static labels. Region identifiers need not be order identifiers.

    Resolving power and other physical/instrument parameters belong to the
    forward model, not this container. No evaluation, fitting or file I/O.
    """

    wavelength: object
    flux: object
    uncertainty: object
    mask: object = None
    region: tuple[str | int, ...] | None = field(default=None, kw_only=True)
    order: tuple[str | int, ...] | None = field(default=None, kw_only=True)
    exposure: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        wavelength = _numeric_array(self.wavelength, "wavelength")
        flux = _numeric_array(self.flux, "flux")
        uncertainty = _numeric_array(self.uncertainty, "uncertainty")
        if wavelength.ndim not in (1, 2) or any(size == 0 for size in wavelength.shape):
            raise ValueError("wavelength must have nonempty (n_pixel,) or (n_region, n_pixel) shape")
        if flux.shape != wavelength.shape or uncertainty.shape != wavelength.shape:
            raise ValueError("wavelength, flux and uncertainty must have identical shapes")
        mask = _exclusion_mask(self.mask, wavelength)
        wave = np.asarray(wavelength)
        if not np.all(np.isfinite(wave) & (wave > 0)):
            raise ValueError("wavelength must contain finite positive values, including masked pixels")
        if np.any(wave[..., 1:] <= wave[..., :-1]):
            raise ValueError("wavelength must be strictly increasing within each region")
        usable = ~np.asarray(mask)
        if not np.all(np.isfinite(np.asarray(flux)[usable])):
            raise ValueError("flux must be finite on usable (unmasked) pixels")
        error = np.asarray(uncertainty)[usable]
        if not np.all(np.isfinite(error) & (error > 0)):
            raise ValueError("uncertainty must be finite and positive on usable (unmasked) pixels")
        count = 1 if wavelength.ndim == 1 else wavelength.shape[0]
        region = _identifiers(self.region, "region", count)
        order = _identifiers(self.order, "order", count)
        if self.exposure is not None and (not isinstance(self.exposure, str) or not self.exposure):
            raise ValueError("exposure must be a nonempty string or None")
        for name, value in (("wavelength", wavelength), ("flux", flux),
                            ("uncertainty", uncertainty), ("mask", mask),
                            ("region", region), ("order", order)):
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return self.wavelength.shape

    @property
    def ndim(self):
        return self.wavelength.ndim

    @property
    def n_regions(self):
        return 1 if self.ndim == 1 else self.shape[0]

    @property
    def n_pixels(self):
        return self.shape[-1]

    @property
    def valid(self):
        return ~self.mask

    def tree_flatten(self):
        return (self.wavelength, self.flux, self.uncertainty, self.mask), (self.region, self.order, self.exposure)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        # JIT/vmap may supply tracers or abstract placeholders; validate only at
        # public construction, not while JAX reconstructs the numerical PyTree.
        result = object.__new__(cls)
        for name, value in zip(("wavelength", "flux", "uncertainty", "mask", "region", "order", "exposure"),
                               (*children, *metadata)):
            object.__setattr__(result, name, value)
        return result
