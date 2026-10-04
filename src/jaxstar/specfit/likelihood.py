"""Normalized spectral densities and conditional means at fixed parameters.

GP regions are independent. Exact masking is prepared once outside JIT; large
numerical arrays remain dynamic. tinygp is imported only by GP operations.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np

from .continuum import (
    ContinuumPosterior, _basis, _conditional_system, _degree, _real_array,
    _region_jitter, chebyshev_basis, continuum_prior,
)
from .model import _require
from .observation import Observation


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _GPObservation:
    observation: Observation
    indices: tuple[tuple[int, ...], ...]

    def tree_flatten(self):
        return (self.observation,), self.indices

    @classmethod
    def tree_unflatten(cls, indices, children):
        return cls(children[0], indices)


def prepare_gp_observation(observation):
    """Prepare exact fixed-mask conditioning indices before JIT/NUTS setup.

    Returns a PyTree accepted in place of Observation by the GP functions.
    Observation arrays stay dynamic; each region's usable integer indices are
    static metadata, allowing different usable counts without traced Boolean
    subsetting. Rebuild this object whenever the mask changes. Observation
    itself is unchanged. No covariance or GP parameters are cached here.

    Eager calls and JIT closures over a concrete Observation may pass the
    Observation directly. To pass observation data as a dynamic JIT argument,
    prepare it outside the transformed function first.
    """
    if isinstance(observation, _GPObservation):
        return observation
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation or prepared GP observation")
    if isinstance(observation.mask, jax.core.Tracer):
        raise ValueError("GP masks must be fixed before tracing; call "
                         "prepare_gp_observation(observation) outside JIT/NUTS")
    mask = np.atleast_2d(np.asarray(observation.mask))
    indices = tuple(tuple(np.flatnonzero(~row).tolist()) for row in mask)
    return _GPObservation(observation, indices)


def _resolve_gp_solver(gp_solver):
    if gp_solver not in ("auto", "quasisep", "direct"):
        raise ValueError("gp_solver must be 'auto', 'quasisep' or 'direct'")
    if gp_solver == "auto":
        platform = jax.default_backend()
        if platform == "cpu":
            return "quasisep"
        if platform in ("gpu", "cuda", "rocm"):
            return "direct"
        raise ValueError("auto GP solver supports CPU/GPU; select an explicit gp_solver")
    return gp_solver


def _positive_scalar(value, name):
    value = _real_array(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar shared across regions")
    _require(jnp.isfinite(value) & (value > 0), f"{name} must be finite and positive")
    return value


def _gp_setup(observation, model_flux, gp_amplitude, gp_scale, jitter, gp_solver,
              *, dtype_values=()):
    prepared = prepare_gp_observation(observation)
    obs = prepared.observation
    flux = _real_array(model_flux, "model_flux")
    if flux.shape != obs.shape:
        raise ValueError("model_flux shape must match Observation.shape")
    amplitude = _positive_scalar(gp_amplitude, "gp_amplitude")
    scale = _positive_scalar(gp_scale, "gp_scale")
    jitter = _region_jitter(jitter, obs.n_regions)
    dtype = jnp.result_type(flux, obs.wavelength, obs.flux, obs.uncertainty,
                            amplitude, scale, jitter, *dtype_values)
    # tinygp 0.3's NumPy sqrt(3) promotes some state-space arrays to float64
    # in x64 mode, giving an invalid mixed-dtype lax.scan for float32 inputs.
    # Keep precision under caller control rather than changing global config.
    if dtype == jnp.float32 and jax.config.x64_enabled:
        raise ValueError("tinygp 0.3 float32 calculations require x64 disabled; "
                         "use float64 inputs or disable x64 in the caller")
    return (prepared, jnp.atleast_2d(flux.astype(dtype)), amplitude.astype(dtype),
            scale.astype(dtype), jitter.astype(dtype), dtype, _resolve_gp_solver(gp_solver))


def _region_gp(obs, region, indices, amplitude, scale, jitter, dtype, solver):
    # Optional dependency: iid users need not import/install tinygp.
    try:
        import tinygp
    except ImportError as exc:
        raise ImportError("GP functions require tinygp; install jaxstar[spectral-inference]") from exc
    index = jnp.asarray(indices, dtype=jnp.int32)
    wave = jnp.atleast_2d(jnp.asarray(obs.wavelength, dtype=dtype))
    origin = wave[region, 0]
    x = wave[region, index] - origin
    # Gather usable values before any subtraction, squaring or GP operation.
    y = jnp.atleast_2d(jnp.asarray(obs.flux, dtype=dtype))[region, index]
    error = jnp.atleast_2d(jnp.asarray(obs.uncertainty, dtype=dtype))[region, index]
    kernel = tinygp.kernels.quasisep.Matern32(sigma=amplitude, scale=scale)
    kwargs = ({"solver": tinygp.solvers.QuasisepSolver, "assume_sorted": True}
              if solver == "quasisep" else {"solver": tinygp.solvers.DirectSolver})
    gp = tinygp.GaussianProcess(kernel, x, diag=error**2 + jitter**2, mean=0., **kwargs)
    return gp, y, index, origin


class _GPContinuumSystem(NamedTuple):
    gp: object
    scaled: object
    residual: object
    cholesky: object
    delta: object


def _gp_continuum_systems(observation, model_flux, gp_amplitude, gp_scale,
                          sigma_constant, sigma_continuum, degree, basis, jitter, gp_solver):
    prepared = prepare_gp_observation(observation)
    obs = prepared.observation
    degree = _degree(degree)
    basis = chebyshev_basis(obs.wavelength, degree) if basis is None else _basis(basis)
    if basis.shape != obs.shape + (degree + 1,):
        raise ValueError("basis shape must be Observation.shape + (degree + 1,)")
    constant = _real_array(sigma_constant, "sigma_constant")
    continuum = _real_array(sigma_continuum, "sigma_continuum")
    prepared, flux, amplitude, scale, jitters, dtype, solver = _gp_setup(
        prepared, model_flux, gp_amplitude, gp_scale, jitter, gp_solver,
        dtype_values=(basis, constant, continuum))
    prior = continuum_prior(sigma_constant=constant.astype(dtype),
                            sigma_continuum=continuum.astype(dtype), degree=degree,
                            n_regions=obs.n_regions)
    basis = basis.astype(dtype).reshape((obs.n_regions, obs.n_pixels, degree + 1))
    systems = []
    identity = jnp.eye(degree + 1, dtype=dtype)
    for region, indices in enumerate(prepared.indices):
        if not indices:
            systems.append(None)
            continue
        gp, y, index, _ = _region_gp(obs, region, indices, amplitude, scale,
                                    jitters[region], dtype, solver)
        f = flux[region, index]
        _require(jnp.all(jnp.isfinite(f)), "model_flux must be finite on usable pixels")
        design = f[:, None] * basis[region, index]
        residual = y - design @ prior.mean[region]
        columns = design * prior.scale[region]
        # One solver whitening operation for residual and all prior-scaled
        # design columns. Only H has a dense continuum-space Cholesky here.
        rhs = jnp.concatenate((residual[:, None], columns), axis=1)
        transformed = gp.solver.solve_triangular(rhs)
        r, z = transformed[:, 0], transformed[:, 1:]
        chol = jnp.linalg.cholesky(identity + z.T @ z)
        delta = cho_solve((chol, True), z.T @ r)
        systems.append(_GPContinuumSystem(gp, z, r, chol, delta))
    return obs, prior, systems


def gp_marginalized_continuum_log_likelihood(
        observation, model_flux, *, gp_amplitude, gp_scale, sigma_constant,
        sigma_continuum, degree=4, basis=None, jitter=0.0, gp_solver="auto"):
    """Normalized Matérn-3/2 density with Gaussian continuum integrated out.

    k(d)=gp_amplitude**2*(1+sqrt(3)*d/gp_scale)*exp(-sqrt(3)*d/gp_scale).
    Amplitude is a scalar standard deviation in observation flux units; scale
    is a scalar correlation length in the observation wavelength unit. Jitter
    is additive absolute flux noise, scalar or (n_region,). Amplitude/scale
    must be finite positive, jitter finite nonnegative. No priors are added.

    Independent regions use C=K+diag(uncertainty**2+jitter**2). Whiten r=y-X*mu
    and A=X*S with the tinygp solver, then solve H=I+Z.T@Z in coefficient space.
    Evaluate the quadratic as ||r_white-Z*delta||**2+||delta||**2 for stability;
    include log|C|, log|H| and every usable pixel's Gaussian normalization.
    Masked pixels are absent from C, not given zero weight or huge noise.
    Fully masked regions contribute exactly zero.

    gp_solver is static: auto selects Quasisep on JAX's default CPU backend,
    Direct on GPU. This is a workload-based benchmark heuristic; an explicit
    override is allowed. Solver choice does not change the statistical model.
    Direct can require much more memory. See prepare_gp_observation for JIT.
    """
    _, prior, systems = _gp_continuum_systems(
        observation, model_flux, gp_amplitude, gp_scale, sigma_constant,
        sigma_continuum, degree, basis, jitter, gp_solver)
    total = jnp.zeros((), dtype=prior.scale.dtype)
    for system in systems:
        if system is None:
            continue
        fitted = system.residual - system.scaled @ system.delta
        quadratic = jnp.sum(fitted**2) + jnp.sum(system.delta**2)
        # tinygp normalization is (log|C| + N*log(2*pi))/2.
        total -= (.5 * quadratic + system.gp.solver.normalization()
                  + jnp.log(jnp.diag(system.cholesky)).sum())
    return total


def gp_continuum_posterior(
        observation, model_flux, *, gp_amplitude, gp_scale, sigma_constant,
        sigma_continuum, degree=4, basis=None, jitter=0.0, gp_solver="auto"):
    """Conditional continuum coefficients under the same GP covariance.

    Returns ContinuumPosterior(mean, covariance) with the same shapes and prior
    convention as continuum_posterior. Uses the likelihood's whitening and
    small Cholesky system. A fully masked region returns its continuum prior.
    At fixed parameters, use apply_continuum with this mean, then condition the
    GP on residuals relative to that modified spectrum with gp_conditional_mean.
    """
    obs, prior, systems = _gp_continuum_systems(
        observation, model_flux, gp_amplitude, gp_scale, sigma_constant,
        sigma_continuum, degree, basis, jitter, gp_solver)
    means, covariances = [], []
    identity = jnp.eye(degree + 1, dtype=prior.scale.dtype)
    for region, system in enumerate(systems):
        scale = prior.scale[region]
        if system is None:
            means.append(prior.mean[region])
            covariances.append(scale[:, None]**2 * identity)
        else:
            means.append(prior.mean[region] + scale * system.delta)
            covariance = cho_solve((system.cholesky, True), identity)
            covariances.append(scale[:, None] * covariance * scale[None, :])
    mean, covariance = jnp.stack(means), jnp.stack(covariances)
    if obs.ndim == 1:
        mean, covariance = mean[0], covariance[0]
    return ContinuumPosterior(mean, covariance)


def gp_conditional_mean(observation, model_flux, *, gp_amplitude, gp_scale,
                        jitter=0.0, wavelength=None, gp_solver="auto"):
    """Latent residual GP mean at one explicit nonlinear/continuum point.

    model_flux is the deterministic data-space mean on the full observation
    grid, normally apply_continuum(physical_flux, basis, posterior.mean).
    Condition only on usable residuals (observation.flux-model_flux). Predict
    at all observation wavelengths by default, including excluded pixels.
    Optional wavelength has shape (test_pixel,) for 1D data or
    (n_region, test_pixel) for 2D data; values are finite positive and increasing.
    A fully masked region returns zero. No cross-region covariance is formed.

    Together with gp_continuum_posterior, this is the correct posterior-mean
    decomposition of the linear-Gaussian continuum + GP at fixed parameters.
    It returns no predictive variance or continuum/GP uncertainty propagation,
    and performs no averaging over nonlinear posterior samples.
    """
    prepared, flux, amplitude, scale, jitters, dtype, solver = _gp_setup(
        observation, model_flux, gp_amplitude, gp_scale, jitter, gp_solver)
    obs = prepared.observation
    wave = _real_array(obs.wavelength if wavelength is None else wavelength, "wavelength")
    if wave.ndim != obs.ndim or wave.shape[:-1] != obs.shape[:-1] or wave.shape[-1] == 0:
        raise ValueError("prediction wavelength must preserve Observation's region shape")
    _require(jnp.all(jnp.isfinite(wave) & (wave > 0))
             & jnp.all(wave[..., 1:] > wave[..., :-1]),
             "prediction wavelength must be finite, positive and strictly increasing")
    wave = jnp.atleast_2d(wave.astype(dtype))
    predictions = []
    for region, indices in enumerate(prepared.indices):
        if not indices:
            predictions.append(jnp.zeros_like(wave[region]))
            continue
        gp, y, index, origin = _region_gp(obs, region, indices, amplitude, scale,
                                         jitters[region], dtype, solver)
        f = flux[region, index]
        _require(jnp.all(jnp.isfinite(f)), "model_flux must be finite on usable pixels")
        residual = y - f
        whitened = gp.solver.solve_triangular(residual)
        alpha = gp.solver.solve_triangular(whitened, transpose=True)
        # The released kernel matmul computes K_test,valid * C^-1 * residual
        # without constructing a conditional test covariance just for its mean.
        predictions.append(gp.kernel.matmul(wave[region] - origin, gp.X, alpha))
    prediction = jnp.stack(predictions)
    return prediction[0] if obs.ndim == 1 else prediction
