"""Small synchronized continuum sanity check, with the physical model for scale.

PYTHONPATH=src python benchmarks/benchmark_continuum.py --require-cpu
Repeat with --jitter 0.01 for additive flux-unit noise, including its derivative.
Use JAX_PLATFORMS=cpu on a GPU host to select CPU, or --require-gpu to verify GPU.
Synthetic inputs only; no fitting, data downloads or benchmark infrastructure.
"""

import argparse
import re

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.specfit import Observation, chebyshev_basis, marginalized_continuum_log_likelihood
from benchmark_specmodel import make_inputs
from _benchmark_utils import emit_result, environment, measure


def benchmark(regions, pixels, dtype, rounds, jitter=0.0):
    model, params, wavelength = make_inputs(regions, max(4000, pixels+400), pixels, dtype)
    device = jax.devices()[0]
    model, params, wavelength = jax.device_put((model, params, wavelength), device)
    physical = model(params, wavelength)
    basis = chebyshev_basis(wavelength)
    measured = physical*(1.02 + .01*basis[..., 1])
    obs = Observation(wavelength, measured, jnp.full_like(measured, .01))
    obs, basis, physical = jax.device_put((obs, basis, physical), device)
    s0, sc = jnp.asarray(.1, dtype=dtype), jnp.asarray(.03, dtype=dtype)
    jitter = jnp.asarray(jitter, dtype=dtype)
    jax.block_until_ready((model, params, obs, basis, physical))
    likelihood = lambda o, f, b, a, c, j: marginalized_continuum_log_likelihood(
        o, f, basis=b, sigma_constant=a, sigma_continuum=c, jitter=j)
    joint = lambda m, p, o, b, a, c, j: likelihood(o, m(p, o.wavelength), b, a, c, j)
    cases = (
        ("continuum", likelihood, (obs, physical, basis, s0, sc, jitter)),
        ("continuum_value_and_grad", jax.value_and_grad(likelihood, argnums=(1, 3, 4, 5)), (obs, physical, basis, s0, sc, jitter)),
        ("specmodel_forward", lambda m, p, w: m(p, w), (model, params, wavelength)),
        ("joint_value_and_grad", jax.value_and_grad(joint, argnums=(1, 4, 5, 6)), (model, params, obs, basis, s0, sc, jitter)),
    )
    timing, memory = {}, {}
    for name, function, args in cases:
        timing[name], memory[name], result = measure(function, args, rounds=rounds)
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(result))
    hlo = jax.jit(likelihood).lower(obs, physical, basis, s0, sc, jitter).compiler_ir(dialect="hlo").as_hlo_text()
    shapes = {tuple(int(n) for n in match.split(",") if n)
              for match in re.findall(r"(?:f32|f64|s32|pred)\[([0-9,]*)\]", hlo)}
    assert not any(shape.count(pixels) > 1 for shape in shapes), "unexpected pixel covariance in HLO"
    assert (regions, 5, 5) in shapes
    return {
        "benchmark": "continuum", "environment": environment(device),
        "problem": {"regions": regions, "pixels_per_region": pixels, "degree": 4,
                    "dtype": str(dtype), "jitter_flux_units": float(jitter)},
        "timing": timing, "memory": memory,
        "validation": {"finite_values_and_gradients": True, "pixel_covariance_in_hlo": False,
                       "coefficient_system_shape": [regions, 5, 5],
                       "largest_hlo_array_elements": int(max(np.prod(shape) for shape in shapes))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--pixels", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.regions < 1 or args.pixels < 6 or args.rounds < 1:
        parser.error("regions/rounds must be positive and pixels >= 6")
    if not np.isfinite(args.jitter) or args.jitter < 0:
        parser.error("jitter must be finite and nonnegative")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    backend = jax.default_backend()
    if (args.require_cpu and backend != "cpu") or (args.require_gpu and backend != "gpu"):
        parser.error(f"required backend unavailable: {backend}")
    emit_result(benchmark(args.regions, args.pixels, np.dtype(args.dtype), args.rounds, args.jitter), args.json_out)


if __name__ == "__main__":
    main()
