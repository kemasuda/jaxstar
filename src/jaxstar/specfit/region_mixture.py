"""One marginalized good/bad label per region, with white noise in both.

Both components share the physical spectrum and analytically integrate the
same Gaussian prior on a multiplicative Chebyshev baseline. Their diagonal
residual variances differ only by an extra global mismatch_scale squared.
"""
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from jax.scipy.linalg import cho_solve

from .configured_model import ConfiguredModel
from .continuum import (_conditional_system, ContinuumPosterior, chebyshev_basis,
                        evaluate_continuum)
from .likelihood import _mixture_parameters
from .numpyro_model import _draw, make_model_single, model_single


class RegionMixtureStatistics(NamedTuple):
    component_log_likelihood: object  # (region, 2), good / bad
    log_likelihood: object            # (region,), labels marginalized
    region_mismatch_probability: object


def _white_statistics(obs, flux, *, jitter, sigma_constant, sigma_continuum,
                      degree, basis, posterior=False):
    prior, z, r, chol, delta, error, valid = _conditional_system(
        obs, flux, sigma_constant, sigma_continuum, degree, basis, jitter)
    residual = r - jnp.einsum('...ik,...k->...i', z, delta)
    quadratic = jnp.sum(residual**2, axis=-1) + jnp.sum(delta**2, axis=-1)
    logdet = 2*(jnp.log(error).sum(axis=-1) + jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1))
    logp = -.5*(quadratic + logdet + jnp.sum(valid, axis=-1)*jnp.log(2*jnp.pi))
    post = None
    if posterior:
        identity = jnp.broadcast_to(jnp.eye(degree+1, dtype=chol.dtype), chol.shape)
        covariance = cho_solve((chol, True), identity)
        post = ContinuumPosterior(prior.mean + prior.scale*delta,
            prior.scale[..., :, None]*covariance*prior.scale[..., None, :])
    return jnp.atleast_1d(logp), post


def _components(observation, model_flux, *, mismatch_fraction, mismatch_scale,
                jitter=0., sigma_constant=.1, sigma_continuum=.05,
                degree=1, basis=None, posterior=False):
    fraction, scale = _mixture_parameters(mismatch_fraction, mismatch_scale)
    options = dict(sigma_constant=sigma_constant, sigma_continuum=sigma_continuum,
                   degree=degree, basis=basis, posterior=posterior)
    good, good_post = _white_statistics(observation, model_flux, jitter=jitter, **options)
    bad, bad_post = _white_statistics(observation, model_flux, jitter=jnp.hypot(jitter, scale), **options)
    components = jnp.stack((good, bad), axis=-1)
    log_good, log_bad = jnp.log1p(-fraction)+good, jnp.log(fraction)+bad
    total = jnp.logaddexp(log_good, log_bad)
    p_bad = jnp.exp(log_bad-total)
    # Entirely masked regions have no evidence, including at pure-component limits.
    empty = jnp.atleast_1d(jnp.sum(observation.valid, axis=-1) == 0)
    total = jnp.where(empty, 0., total)
    p_bad = jnp.where(empty, fraction, p_bad)
    return RegionMixtureStatistics(components, total, p_bad), good_post, bad_post


def region_mixture_statistics(observation, model_flux, **kwargs):
    """Normalized per-region densities and conditional bad probabilities.

    The mixture is taken AFTER accumulating pixels and integrating the baseline
    separately in each component. Fixed fractions 0/1 are supported. Fully
    masked regions contribute zero and return the prior bad fraction.
    """
    return _components(observation, model_flux, **kwargs)[0]


def model_region_mixture(observation, specmodel, *, mismatch_fraction,
                         mismatch_scale, prior_only=False, save_model_flux=False,
                         **single_options):
    """Reuse single-star priors/forward model; integrate one label per region."""
    with handlers.trace() as trace:
        params = model_single(observation, specmodel, prior_only=True,
                              use_gp=False, likelihood='gaussian', **single_options)
    fraction, scale = _mixture_parameters(_draw('mismatch_fraction', mismatch_fraction),
                                         _draw('mismatch_scale', mismatch_scale))
    if prior_only:
        return params
    flux = specmodel(params, observation.wavelength)
    if save_model_flux:
        numpyro.deterministic('model_flux', flux)
    statistics = region_mixture_statistics(observation, flux,
        mismatch_fraction=fraction, mismatch_scale=scale,
        **{name: trace[name]['value'] for name in ('jitter', 'sigma_constant', 'sigma_continuum')},
        degree=single_options.get('degree', 1), basis=single_options.get('basis'))
    numpyro.factor('spectrum', statistics.log_likelihood.sum())
    return params


def reconstruct_region_mixture(obs, specmodel, params, trace, settings):
    """Reconstruct the baseline averaged over the two possible region labels."""
    physical = specmodel(params, obs.wavelength)
    noise = {name: trace[name]['value'] for name in (
        'jitter', 'sigma_constant', 'sigma_continuum', 'mismatch_fraction', 'mismatch_scale')}
    basis = settings.get('basis')
    if basis is None:
        basis = chebyshev_basis(obs.wavelength, settings['degree'])
    stats, good, bad = _components(obs, physical, **noise,
                                   degree=settings['degree'], basis=basis, posterior=True)
    probability = stats.region_mismatch_probability
    p = probability[0] if obs.ndim == 1 else probability
    mean = (1-p[..., None])*good.mean + p[..., None]*bad.mean
    dg, db = good.mean-mean, bad.mean-mean
    covariance = ((1-p[..., None, None])*(good.covariance + dg[..., :, None]*dg[..., None, :])
                  + p[..., None, None]*(bad.covariance + db[..., :, None]*db[..., None, :]))
    continuum = evaluate_continuum(basis, mean)
    flux = physical*continuum
    return dict(params=params, physical_flux=physical, continuum=continuum,
        model_flux=flux, mean_flux=flux, **noise,
        continuum_posterior=ContinuumPosterior(mean, covariance),
        good_model_flux=physical*evaluate_continuum(basis, good.mean),
        bad_model_flux=physical*evaluate_continuum(basis, bad.mean),
        region_mismatch_probability=probability,
        component_log_likelihood=stats.component_log_likelihood,
        region_log_likelihood=stats.log_likelihood,
        mismatch_probability=jnp.where(obs.valid, p[..., None], jnp.nan))


def make_model_region_mixture(sf, *, mismatch_fraction=None, mismatch_scale=None,
                              degree=1, use_gp=False, vsini=None, **single_options):
    """Configure a white/white region mixture, compatible with SVI and NUTS.

    All single-star physical settings apply. Atmosphere may be fixed or free.
    vsini=None keeps the default Uniform(0, vsini_max); a number fixes rotation,
    and a scalar NumPyro distribution supplies a shared rotation prior. Its
    support must stay within the physical kernel coverage.
    Defaults: independent depth_scale~LogNormal(0,.3) per region, global
    mismatch_fraction~Beta(1,9), mismatch_scale~HalfNormal(.1). Supply fixed
    numbers or different distributions to override. RV bounds are per region;
    vsini/vmacro are shared unless explicitly configured otherwise. Baseline
    coefficients and region labels are marginalized, never sampled. Both
    components use uncertainty²+jitter²; bad adds mismatch_scale². No GP and
    no pixel guard. No masking or rejection is performed by this factory.
    """
    fixed = dict(single_options.get('fixed') or {})
    if vsini is not None:
        if 'vsini' in fixed:
            raise ValueError("supply vsini or fixed['vsini'], not both")
        if isinstance(vsini, dist.Distribution):
            if vsini.batch_shape or vsini.event_shape:
                raise ValueError('vsini prior must be scalar/shared across regions')
        else:
            fixed['vsini'] = vsini
            single_options['fixed'] = fixed
    if use_gp is not False:
        raise ValueError('region mixture currently uses white noise in both components; use_gp=False')
    for key in ('likelihood', 'mask_guard', 'gp_amplitude_bounds', 'gp_scale_bounds', 'gp_solver'):
        if key in single_options:
            raise ValueError(f'{key} does not apply to the white/white region mixture')
    if sf.mask_guard is not None:
        raise ValueError('pixel mask_guard is not defined for whole-region labels')
    single_options.setdefault('depth_scale', dist.LogNormal(0., .3).expand([sf.observation.n_regions]))
    single = make_model_single(sf, use_gp=False, degree=degree, **single_options)
    settings = dict(single.settings)
    if isinstance(vsini, dist.Distribution):
        # Preserve the usual priors for the remaining broadening parameters.
        fixed = settings['fixed']
        broadening = dict(vsini=vsini,
            q1=fixed.get('q1', dist.Uniform(0., 1.)),
            q2=fixed.get('q2', dist.Uniform(0., 1.)))
        if not settings['use_empirical_vmacro']:
            broadening['vmacro'] = fixed.get('vmacro', dist.Uniform(*settings['vmacro_bounds']))
        settings['broadening'] = broadening
    for key in ('use_gp', 'likelihood', 'mismatch_fraction', 'mismatch_scale', 'mask_guard',
                'prior_only', 'save_model_flux'):
        settings.pop(key, None)
    settings.update(mismatch_fraction=dist.Beta(1., 9.) if mismatch_fraction is None else mismatch_fraction,
                    mismatch_scale=dist.HalfNormal(.1) if mismatch_scale is None else mismatch_scale)
    model = ConfiguredModel(model_region_mixture, settings=settings,
        reconstruct_fn=reconstruct_region_mixture, observation=sf.observation,
        specmodel=sf.specmodel, site_annotations=single.site_annotations)
    model.prior_trace()
    return model


def run_region_mixture_prefit(sf, *, rng_key, model_kwargs=None, p_initial=None,
                              step_size=.01, num_steps=2000, progress_bar=True):
    """Return (configured model, physical-coordinate MAP point); never mask data."""
    from numpyro.distributions.transforms import biject_to
    for name, value in (('model_kwargs', model_kwargs), ('p_initial', p_initial)):
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f'{name} must be a mapping')
    model = make_model_region_mixture(sf, **dict(model_kwargs or {}))
    init_key, fit_key = jax.random.split(rng_key)
    supplied = {name: jnp.asarray(value) for name, value in (p_initial or {}).items()}
    trace = model.prior_trace(rng_key=init_key, condition=supplied)
    defaults = {}
    for name, value in (('depth_scale', 1.), ('jitter', .002), ('mismatch_scale', .1), ('mismatch_fraction', .1)):
        site = trace.get(name)
        if site is not None and site['type'] == 'sample' and not site['is_observed']:
            candidate = jnp.full_like(site['value'], value)
            if np.all(site['fn'].support.check(candidate)):
                defaults[name] = candidate
    trace = model.prior_trace(rng_key=init_key, condition={**defaults, **supplied})
    initial = {}
    for name, site in trace.items():
        if site['type'] != 'sample' or site['is_observed']:
            continue
        value, prior = jnp.asarray(site['value']), site['fn']
        if prior.support.is_discrete:
            raise ValueError(f'prefit requires a continuous prior for {name!r}')
        if not np.all(np.isfinite(value)) or not np.all(prior.support.check(value)) or not np.all(np.isfinite(biject_to(prior.support).inv(value))):
            raise ValueError(f'initial value for {name!r} must be finite and strictly inside its prior support')
        initial[name] = value
    point = sf.run_svi(model=model, rng_key=fit_key, p_initial=initial,
                       step_size=step_size, num_steps=num_steps, progress_bar=progress_bar)
    return model, point
