"""Multiplicative Chebyshev continua and small-system Gaussian marginalization.

These are data-space nuisance operations after the final SpecModel spectrum;
neither SpecModel nor Observation owns continuum parameters. No positivity
constraint is applied: coefficients are intentionally unconstrained Gaussian.
"""

from numbers import Integral
from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from .model import _require
from .observation import Observation


class ContinuumPrior(NamedTuple):
    """Diagonal Gaussian prior; arrays have shape (coefficient,) or (region, coefficient)."""

    mean: object
    scale: object

    @property
    def covariance(self):
        return self.scale[..., :, None] ** 2 * jnp.eye(self.scale.shape[-1], dtype=self.scale.dtype)


class ContinuumPosterior(NamedTuple):
    """Conditional coefficient mean and covariance, with an optional region axis."""

    mean: object
    covariance: object


def _degree(degree):
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0:
        raise ValueError("degree must be a nonnegative static integer")
    return int(degree)


def _real_array(value, name):
    array = jnp.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    if array.dtype.kind in "iu":
        array = array.astype(jnp.result_type(array, 0.))
    return array


def chebyshev_basis(wavelength, degree=4):
    """Return T0..Tdegree with shape wavelength.shape + (degree + 1,).

    Each nonempty 1D spectrum or row of a 2D spectrum uses its own endpoints:
    x = 2*(wavelength - wavelength[..., :1])/(last - first) - 1.
    Wavelengths must be finite, positive and strictly increasing. For the
    degenerate one-pixel region x=0. Degree is static under JIT; numerical
    wavelength inputs remain differentiable. No global precision toggle.
    """
    degree = _degree(degree)
    wave = _real_array(wavelength, "wavelength")
    if wave.ndim not in (1, 2) or any(size == 0 for size in wave.shape):
        raise ValueError("wavelength must have nonempty (pixel,) or (region, pixel) shape")
    _require(jnp.all(jnp.isfinite(wave) & (wave > 0))
             & jnp.all(wave[..., 1:] > wave[..., :-1]),
             "wavelength must be finite, positive and strictly increasing within each region")
    if wave.shape[-1] == 1:
        x = jnp.zeros_like(wave)
    else:
        x = 2 * (wave - wave[..., :1]) / (wave[..., -1:] - wave[..., :1]) - 1
    terms = [jnp.ones_like(x)]
    if degree:
        terms.append(x)
    for k in range(2, degree + 1):
        terms.append(2 * x * terms[k - 1] - terms[k - 2])
    return jnp.stack(terms, axis=-1)


def _basis(basis):
    basis = _real_array(basis, "basis")
    if basis.ndim not in (2, 3) or any(size == 0 for size in basis.shape):
        raise ValueError("basis must have nonempty (pixel, coefficient) or (region, pixel, coefficient) shape")
    _require(jnp.all(jnp.isfinite(basis)), "basis must be finite")
    return basis


def evaluate_continuum(basis, coefficients):
    """Evaluate C=B*a; require exactly (coefficient,) or (region, coefficient)."""
    basis = _basis(basis)
    coefficients = _real_array(coefficients, "coefficients")
    expected = basis.shape[:-2] + (basis.shape[-1],)
    if coefficients.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}, got {coefficients.shape}")
    return jnp.sum(basis * coefficients[..., None, :], axis=-1)


def continuum_design_matrix(model_flux, basis):
    """Return X[i,k]=model_flux[i]*basis[i,k], preserving independent regions."""
    basis = _basis(basis)
    flux = _real_array(model_flux, "model_flux")
    if flux.shape != basis.shape[:-1]:
        raise ValueError("model_flux shape must match the pixel/region axes of basis")
    return flux[..., None] * basis


def apply_continuum(model_flux, basis, coefficients):
    """Multiply the final physical spectrum by an explicit continuum, without clipping."""
    continuum = evaluate_continuum(basis, coefficients)
    flux = _real_array(model_flux, "model_flux")
    if flux.shape != continuum.shape:
        raise ValueError("model_flux shape must match continuum shape")
    return flux * continuum


def _scale(value, name, count):
    value = _real_array(value, name)
    if value.ndim == 0:
        value = jnp.broadcast_to(value, (count,))
    elif value.shape != (count,):
        raise ValueError(f"{name} must be scalar or shape ({count},), got {value.shape}")
    _require(jnp.all(jnp.isfinite(value) & (value > 0)), f"{name} must be finite and positive")
    return value


def continuum_prior(*, sigma_constant, sigma_continuum, degree=4, n_regions=None):
    """Construct mu=[1,0,...], diagonal scales=[sigma_constant,sigma_continuum,...].

    Both scales must be supplied and finite positive; each is scalar or one per
    region. With n_regions=None return 1D coefficient arrays (one region).
    Explicit n_regions returns (n_regions, degree+1), including n_regions=1.
    No scientific prior-scale default or degree-dependent shrinkage is imposed.
    """
    degree = _degree(degree)
    if n_regions is not None and (isinstance(n_regions, bool)
            or not isinstance(n_regions, Integral) or n_regions < 1):
        raise ValueError("n_regions must be a positive static integer or None")
    count = 1 if n_regions is None else int(n_regions)
    constant = _scale(sigma_constant, "sigma_constant", count)
    nonconstant = _scale(sigma_continuum, "sigma_continuum", count)
    scales = jnp.concatenate((constant[:, None], jnp.broadcast_to(nonconstant[:, None], (count, degree))), axis=-1)
    mean = jnp.zeros_like(scales).at[:, 0].set(1)
    return ContinuumPrior(mean[0], scales[0]) if n_regions is None else ContinuumPrior(mean, scales)


def _region_jitter(value, count):
    value = _real_array(value, "jitter")
    if value.ndim == 0:
        value = jnp.broadcast_to(value, (count,))
    elif value.shape != (count,):
        raise ValueError(f"jitter must be scalar or shape ({count},), got {value.shape}")
    _require(jnp.all(jnp.isfinite(value) & (value >= 0)), "jitter must be finite and nonnegative")
    return value


def _conditional_system(observation, model_flux, sigma_constant, sigma_continuum, degree, basis, jitter):
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")
    degree = _degree(degree)
    flux = _real_array(model_flux, "model_flux")
    if flux.shape != observation.shape:
        raise ValueError("model_flux shape must match Observation.shape")
    basis = chebyshev_basis(observation.wavelength, degree) if basis is None else _basis(basis)
    if basis.shape != observation.shape + (degree + 1,):
        raise ValueError("basis shape must be Observation.shape + (degree + 1,)")
    # Normal JAX promotion, including weak Python scalar prior scales. Do not
    # promote float32 observations merely because the scales are Python floats.
    jitter = _region_jitter(jitter, observation.n_regions)
    dtype = jnp.result_type(flux, basis, observation.flux, observation.uncertainty,
                            _real_array(sigma_constant, "sigma_constant"),
                            _real_array(sigma_continuum, "sigma_continuum"), jitter)
    prior = continuum_prior(sigma_constant=jnp.asarray(sigma_constant, dtype=dtype),
                            sigma_continuum=jnp.asarray(sigma_continuum, dtype=dtype),
                            degree=degree, n_regions=None if observation.ndim == 1 else observation.n_regions)
    usable = jnp.asarray(observation.valid)
    # Replace excluded inputs BEFORE division, squaring, logs or products.
    # This prevents NaNs in inactive branches from poisoning reverse-mode AD.
    y = jnp.where(usable, jnp.asarray(observation.flux, dtype=dtype), 0)
    error = jnp.where(usable, jnp.asarray(observation.uncertainty, dtype=dtype), 1)
    jitter = jitter.astype(dtype)
    per_pixel_jitter = jitter[0] if observation.ndim == 1 else jitter[:, None]
    # Excluded errors stay exactly one even with nonzero regional jitter, so
    # they contribute no Gaussian determinant term. hypot is the stable
    # evaluation of sqrt(sigma_obs**2 + jitter**2), including at jitter=0.
    error = jnp.hypot(error, jnp.where(usable, per_pixel_jitter, 0))
    flux = jnp.where(usable, flux.astype(dtype), 0)
    _require(jnp.all(jnp.isfinite(flux)), "model_flux must be finite on usable pixels")
    design = continuum_design_matrix(flux, basis.astype(dtype))
    residual = (y - jnp.sum(design * prior.mean[..., None, :], axis=-1)) / error
    scaled = design * prior.scale[..., None, :] / error[..., None]
    identity = jnp.eye(degree + 1, dtype=dtype)
    system = identity + jnp.einsum("...ik,...ij->...kj", scaled, scaled)
    cholesky = jnp.linalg.cholesky(system)
    rhs = jnp.einsum("...ik,...i->...k", scaled, residual)
    delta = cho_solve((cholesky, True), rhs[..., None])[..., 0]
    return prior, scaled, residual, cholesky, delta, error, usable


def marginalized_continuum_log_likelihood(observation, model_flux, *, sigma_constant,
                                         sigma_continuum, degree=4, basis=None, jitter=0.0):
    """Fully normalized Gaussian density, summed over independent regions.

    Integrates a~N(mu,diag(scale**2)) in y~N(X*a,D), with diagonal
    D[i,i]=uncertainty[i]**2+jitter[region]**2. Jitter is additive absolute
    noise in the same flux units as uncertainty, scalar or (n_region,),
    finite nonnegative. It has no prior here and is not observation/model state.
    With Z=D**(-1/2)*X*diag(scale), r=D**(-1/2)*(y-X*mu),
    H=I+Z.T*Z and delta=H**(-1)*Z.T*r, the residual quadratic is evaluated
    stably as ||r-Z*delta||**2+||delta||**2, avoiding subtractive Woodbury
    cancellation. logdet = 2*sum(log(error)) + 2*sum(log(diag(chol(H)))).
    The standardized system contains the complete prior determinant contribution.
    Only usable pixels contribute to N*log(2*pi). Fully masked regions contribute
    zero. Both whitening and the Gaussian determinant use the effective error
    sqrt(uncertainty**2+jitter**2). No pixel covariance or numerical ridge.

    Optional basis permits setup-time caching; its last dimension must match
    the static degree. Invalid concrete scales raise ValueError; invalid traced
    scales raise a runtime JAX error reporting the same validation message.
    """
    _, scaled, residual, cholesky, delta, error, usable = _conditional_system(
        observation, model_flux, sigma_constant, sigma_continuum, degree, basis, jitter)
    fitted_residual = residual - jnp.einsum("...ik,...k->...i", scaled, delta)
    quadratic = jnp.sum(fitted_residual ** 2, axis=-1) + jnp.sum(delta ** 2, axis=-1)
    logdet = 2 * (jnp.sum(jnp.log(error), axis=-1)
                  + jnp.sum(jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1))
    count = jnp.sum(usable, axis=-1)
    return jnp.sum(-.5 * (quadratic + logdet + count * jnp.log(2 * jnp.pi)))


def continuum_posterior(observation, model_flux, *, sigma_constant,
                        sigma_continuum, degree=4, basis=None, jitter=0.0):
    """Return conditional coefficient mean and covariance via Cholesky solves.

    Shapes are (degree+1,), (degree+1,degree+1) for 1D data; a leading
    independent region axis is retained for 2D data. Evaluate the recovered
    continuum with evaluate_continuum(basis, posterior.mean), or its spectrum
    with apply_continuum(model_flux, basis, posterior.mean).
    Uses the same uncertainty**2+jitter[region]**2 diagonal covariance as the
    marginalized likelihood, with scalar or per-region additive absolute jitter.
    """
    prior, _, _, cholesky, delta, _, _ = _conditional_system(
        observation, model_flux, sigma_constant, sigma_continuum, degree, basis, jitter)
    identity = jnp.broadcast_to(jnp.eye(prior.scale.shape[-1], dtype=prior.scale.dtype), cholesky.shape)
    standardized_covariance = cho_solve((cholesky, True), identity)
    covariance = prior.scale[..., :, None] * standardized_covariance * prior.scale[..., None, :]
    return ContinuumPosterior(prior.mean + prior.scale * delta, covariance)
