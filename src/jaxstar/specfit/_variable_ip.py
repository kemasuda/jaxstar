"""Precomputed local response for a wavelength-dependent two-Gaussian IP."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .sampling import _C_KMS


class _VariableIP(NamedTuple):
    """Banded response on the library's fixed observer-frame log-lambda grid.

    Each input point has its own calibrated, unit-area kernel. Gather those
    contributions by output point, then normalize by the response to a flat
    continuum. This keeps the point-sampled spectral model continuum-preserving
    even when the IP varies; it is not detector-bin flux integration.

    Only numerical arrays are retained, so ordinary NamedTuple PyTree handling
    keeps the response dynamic under JIT. The user callable is never traced.
    """

    weights: object
    indices: object
    padding: object

    @property
    def half(self):
        return self.weights.shape[-1] // 2

    @classmethod
    def prepare(cls, values, velocity_step, *, dtype):
        """Sample each Gaussian out to at least five sigma, once on the host."""
        values = {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}
        count, pixels = values["fraction2"].shape
        extent = np.zeros((count, pixels))
        for i, active in ((1, values["fraction2"] < 1), (2, values["fraction2"] > 0)):
            radius = np.abs(values[f"mu{i}_kms"]) + 5 * values[f"sigma{i}_kms"]
            extent = np.maximum(extent, np.where(active, radius, 0.))
        half = max(1, int(np.ceil(np.max(extent / velocity_step[:, None]))))
        if pixels <= 2 * half + 1:
            raise ValueError("insufficient model wavelength coverage for fixed_ip; "
                             "prepare wider spectral regions")
        offsets = np.arange(-half, half + 1)
        velocity = velocity_step[:, None, None] * offsets
        kernels = np.zeros((count, pixels, len(offsets)))
        for i, fraction in ((1, 1 - values["fraction2"]), (2, values["fraction2"])):
            sigma = values[f"sigma{i}_kms"][..., None]
            z = (velocity - values[f"mu{i}_kms"][..., None]) / sigma
            kernels += fraction[..., None] / sigma * np.exp(-.5 * z**2)
        area = kernels.sum(axis=-1, keepdims=True)
        if np.any(~np.isfinite(area) | (area <= 0)):
            raise ValueError("fixed_ip is unresolved on the model grid; use finer velocity sampling")
        kernels /= area

        # Output i gathers light from input j = i - offset. In particular,
        # positive mu moves light redward; the calibration is evaluated at j.
        indices = np.arange(half, pixels - half)[:, None] - offsets
        weights = kernels[:, indices, np.arange(len(offsets))]
        continuum = weights.sum(axis=-1, keepdims=True)
        if np.any(continuum <= 0):
            raise ValueError("fixed_ip leaves output wavelengths without support")
        weights /= continuum
        # One extra model step covers the final interpolation's two endpoints.
        padding = np.exp((half + 1) * velocity_step / _C_KMS)
        return cls(jnp.asarray(weights, dtype=dtype), jnp.asarray(indices),
                   jnp.asarray(padding, dtype=dtype))

    def apply(self, flux):
        """Weighted local sums, without a dense pixel-by-pixel matrix."""
        return jnp.sum(self.weights * flux[:, self.indices], axis=-1)
