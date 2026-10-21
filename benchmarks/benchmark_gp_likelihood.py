"""Compare Matérn-3/2 solvers including marginalized continuum and gradients.

Run with PYTHONPATH=src; set JAX_PLATFORMS before starting Python. Uses a
synthetic differentiable line profile, not a stellar grid or inference run.
The installed tinygp is never modified. Pin development dependencies outside
this script and record the tinygp commit with --tinygp-revision.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import tinygp

from _benchmark_utils import emit_result, environment, measure
from jaxstar.specfit import (
    Observation, chebyshev_basis, gp_marginalized_continuum_log_likelihood,
    prepare_gp_observation,
)
from jaxstar.specfit.likelihood import _supports_parallel_gp


def line_model(theta, coordinate):
    depth, center, log_width = theta[:3]
    return 1 - depth * jnp.exp(-.5 * ((coordinate - center) / jnp.exp(log_width))**2)


def problem(pixels, regions, dtype):
    rng = np.random.default_rng(42)
    # Positive, unequal wavelength steps; keep actual gaps after masking.
    x = np.cumsum(rng.uniform(.03, .05, size=(regions, pixels)), axis=1)
    wave = (5000. + 1000. * np.arange(regions)[:, None] + x).astype(dtype)
    coordinate = wave - wave.mean(axis=1, keepdims=True)
    theta = jnp.asarray([.3, .05, np.log(.2), np.log(.025), np.log(.55),
                         np.log(.009), np.log(.1), np.log(.03)], dtype=dtype)
    flux = np.asarray(line_model(theta, coordinate))
    continuum = 1.02 + .02 * coordinate / np.max(np.abs(coordinate), axis=1, keepdims=True)
    y = (flux * continuum + .012 * np.sin(3 * coordinate)
         + rng.normal(0., .01, coordinate.shape)).astype(dtype)
    error = rng.uniform(.018, .022, coordinate.shape).astype(dtype)
    mask = rng.uniform(size=coordinate.shape) < .07
    mask[:, pixels // 3:pixels // 3 + max(1, pixels // 25)] = True
    mask[:, 0] = mask[:, -1] = False
    y[mask], error[mask] = np.nan, np.inf
    return Observation(wave, y, error, mask), jnp.asarray(coordinate), theta


def run(args):
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    supported = _supports_parallel_gp()
    if args.require_parallel and not supported:
        raise ValueError("installed tinygp lacks the QuasisepSolver parallel argument")
    observation, coordinate, theta = problem(args.pixels, args.regions, np.dtype(args.dtype))
    prepared = prepare_gp_observation(observation)
    basis = chebyshev_basis(observation.wavelength, args.degree)
    inputs = jax.device_put((theta, prepared, coordinate, basis))
    jax.block_until_ready(inputs)
    report = {
        "benchmark": "matern32-marginalized-continuum-value-and-grad",
        "environment": environment(jax.devices()[0]),
        "tinygp_version": tinygp.__version__, "tinygp_revision": args.tinygp_revision,
        "parallel_supported": supported, "dtype": args.dtype,
        "pixels": args.pixels, "regions": args.regions, "degree": args.degree,
        "usable_pixels": [len(i) for i in prepared.indices], "rounds": args.rounds,
        "gradient_parameters": ["line_depth", "line_center", "log_line_width",
                                "log_gp_amplitude", "log_gp_scale", "log_jitter",
                                "log_sigma_constant", "log_sigma_continuum"],
        "memory_note": "Compiler buffer estimates; not measured peak device memory.",
        "results": {},
    }
    reference = None
    tolerance = 2e-3 if args.dtype == "float32" else 1e-8
    for name, solver in (("sequential", "quasisep"),
                         ("direct", "direct"),
                         ("parallel", "quasisep_parallel")):
        if name == "parallel" and not supported:
            report["results"][name] = {"skipped": "tinygp does not support parallel Quasisep"}
            continue

        def objective(theta, data, coordinate, basis):
            amplitude, scale, jitter, constant, continuum = jnp.exp(theta[3:])
            return gp_marginalized_continuum_log_likelihood(
                data, line_model(theta, coordinate), basis=basis, degree=args.degree,
                gp_amplitude=amplitude, gp_scale=scale, jitter=jitter,
                sigma_constant=constant, sigma_continuum=continuum,
                gp_solver=solver)

        timing, memory, result = measure(jax.value_and_grad(objective), inputs,
                                          rounds=args.rounds, warmup=3)
        value, gradient = map(np.asarray, result)
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise ValueError(f"{name}: nonfinite value or gradient")
        if reference is None:
            reference = value, gradient
        np.testing.assert_allclose(value, reference[0], rtol=tolerance, atol=tolerance,
                                   err_msg=f"{name}: log likelihood")
        np.testing.assert_allclose(gradient, reference[1], rtol=tolerance, atol=tolerance,
                                   err_msg=f"{name}: gradient")
        report["results"][name] = {
            "timing": timing, "compiler_memory": memory,
            "log_likelihood": float(value), "gradient": gradient.tolist(),
            "value_abs_difference": float(np.abs(value - reference[0])),
            "gradient_max_abs_difference": float(np.max(np.abs(gradient - reference[1]))),
        }
    emit_result(report, args.json_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=2048)
    parser.add_argument("--regions", type=int, default=2)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--require-parallel", action="store_true")
    parser.add_argument("--tinygp-revision", help="Exact source commit of a development build")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.pixels < 8 or args.regions < 1 or args.degree < 0 or args.rounds < 1:
        parser.error("pixels >= 8, regions >= 1, degree >= 0 and rounds >= 1 are required")
    run(args)
