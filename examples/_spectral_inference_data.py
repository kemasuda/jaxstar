"""Repository-only known-truth fixtures for the spectral inference example.

These line responses are synthetic, not a substitute for a stellar library.
The actual grid interpolation, combined broadening, RV, pixel sampling and
continuum likelihood are used without a surrogate forward model.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import (Observation, SpecModel, chebyshev_basis,
                            evaluate_continuum, single_star_params)
from jaxstar.specfit._data import _SpectralLibrary


def synthetic_case(*, regions=3, pixels=384, model_pixels=1201, dtype=np.float64,
                   seed=20261004):
    if not 1 <= regions <= 3 or pixels < 16 or model_pixels < 601:
        raise ValueError("use 1..3 regions, >=16 observed pixels and >=601 model pixels")
    c = 299792.458
    centers = np.array([5000., 6000., 16000.])[:regions]
    velocity = np.linspace(-300., 300., model_pixels)
    wave = (centers[:, None] * np.exp(velocity[None, :] / c)).astype(dtype)
    axes = {"teff": np.array([5000., 6500.], dtype=dtype),
            "logg": np.array([3.5, 4.7], dtype=dtype),
            "mh": np.array([-.6, .4], dtype=dtype),
            "alpha": np.array([0., .4], dtype=dtype)}
    # Four independent line-depth patterns identify the atmosphere coordinates.
    line = np.arange(12)
    responses = np.stack([np.cos(line * 1.7), np.sin(line * 1.2),
                          np.ones(12), np.cos(line * 2.4 + .5)])
    nodes = np.stack(np.meshgrid(*axes.values(), indexing="ij"), axis=-1)
    scales = np.array([1500., 1.2, 1., .4])
    relative = (nodes - np.array([5750., 4.1, -.1, .2])) / scales
    depths = .24 + .10 * np.einsum("...a,al->...l", relative, responses)
    line_centers = np.linspace(-112., 112., 12)
    widths = 1.2 + .6 * (line % 4)
    profiles = np.exp(-.5 * ((velocity[:, None] - line_centers) / widths)**2)
    flux = 1 - np.einsum("...l,pl->...p", depths, profiles)
    # Change line depths between orders, preserving atmosphere axes.
    rows = [1 - (1 - flux) * (1 + .12 * r) for r in range(regions)]
    payload = np.stack(rows, axis=-2).astype(dtype)
    grid = RectilinearGrid(axes=axes, fields={"flux": Field(payload, tuple(axes),
                          payload_dims=("region", "pixel"))})
    library = _SpectralLibrary(grid, wave, tuple(f"synthetic-{r}" for r in range(regions)),
                               "synthetic-line-library", "vacuum", "normalized")
    specmodel = SpecModel(library, vmax=40.)
    # Different count AND nonuniform, offset pixel coordinates; never grid indices.
    observed_velocity = np.linspace(-136.37, 136.19, pixels)
    observed_velocity += .09 * np.sin(np.linspace(0, 7, pixels))
    wav_obs = (centers[:, None] * np.exp((observed_velocity[None, :]
                                      + .073 * np.arange(regions)[:, None]) / c)).astype(dtype)
    # Float32 quantization can accidentally land a few offset pixels on a model
    # sample. Move only those by one representable wavelength step, offline.
    coincident = np.any(wav_obs[..., None] == wave[:, None, :], axis=-1)
    wav_obs[coincident] = np.nextafter(wav_obs[coincident], np.dtype(dtype).type(np.inf))
    separation = np.min(np.abs(wav_obs[..., None] - wave[:, None, :]), axis=-1)
    assert np.all(separation > 0), "observed pixels must differ from model samples"
    truth = {"teff": 5770., "logg": 4.2, "mh": -.12, "alpha": .12,
             "vsini": 7.5, "vmacro": 3.2, "q1": .36, "q2": .3, "rv": 12.4,
             "resolving_power": np.array([68000., 71000., 73000.])[:regions],
             "jitter": .004, "sigma_constant": .1}
    reference_sigma_continuum = .04  # analysis hyperprior center, not injected noise/physics
    basis = chebyshev_basis(wav_obs, degree=4)
    params = point_params(truth, tuple(axes))
    physical = np.asarray(specmodel(params, wav_obs))
    coefficients = np.array([[1.02, .035, -.020, 0., 0.],
                             [.98, -.020, .025, 0., 0.],
                             [1.01, .015, .012, 0., 0.]])[:regions].astype(dtype)
    continuum = np.asarray(evaluate_continuum(basis, coefficients))
    error = np.full_like(physical, .004)
    noise = np.random.default_rng(seed).normal(size=physical.shape) * np.hypot(error, truth["jitter"])
    measured = (physical * continuum + noise).astype(dtype)
    mask = np.zeros(measured.shape, dtype=bool)
    mask[:, pixels // 5:pixels // 5 + max(2, pixels // 32)] = True
    mask[:, 3 * pixels // 4:3 * pixels // 4 + max(2, pixels // 40)] = True
    measured[mask], error[mask] = np.nan, np.inf
    obs = Observation(wav_obs, measured, error, mask, region=library.regions)
    # This is an example's explicit prior choice, not package defaults.
    priors = {
        "atmosphere": {name: dist.Uniform(float(values[0]), float(values[-1]))
                       for name, values in axes.items()},
        "broadening": {"vsini": dist.Uniform(2., 13.),
                       "vmacro": dist.TruncatedNormal(3., 1., low=.5, high=6.),
                       "q1": truth["q1"], "q2": truth["q2"]},
        "rv": dist.Uniform(8., 17.),
        "resolving_power": dist.TruncatedNormal(jnp.asarray(truth["resolving_power"]),
                                                3000., low=55000., high=85000.),
        "sigma_constant": truth["sigma_constant"],
        "sigma_continuum": dist.LogNormal(np.log(.04), .7),
        "jitter": dist.HalfNormal(.01), "basis": basis, "degree": 4,
    }
    initial = {"teff": 5650., "logg": 4.05, "mh": -.05, "alpha": .18,
               "vsini": 6.5, "vmacro": 3., "rv": 11.7,
               "resolving_power": truth["resolving_power"],
               "jitter": .006, "sigma_continuum": .03}
    return SimpleNamespace(observation=obs, specmodel=specmodel, basis=basis,
                           truth=truth, initial=initial, priors=priors,
                           reference_sigma_continuum=reference_sigma_continuum,
                           physical=physical, continuum=continuum, coefficients=coefficients,
                           seed=seed, minimum_grid_separation=float(separation.min()))


def point_params(point, axes):
    return single_star_params({name: point[name] for name in axes},
        **{name: point[name] for name in ("vsini", "vmacro", "q1", "q2", "rv", "resolving_power")})
