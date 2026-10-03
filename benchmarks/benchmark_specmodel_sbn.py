"""Synchronized deterministic SB-N scaling; no data, inference or specialization.

Defaults: N=1,2,3; ten regions, 4000 model / 2000 output pixels, float32.
N=1 exactly reuses benchmark_specmodel's inputs and complete parameter objective.
"""

import argparse
import copy

import jax
import jax.numpy as jnp
import numpy as np

from benchmark_specmodel import make_inputs
from _benchmark_utils import emit_result, environment, measure


def component_inputs(count, regions, pixels, output_pixels, dtype):
    model, params, wavelength = make_inputs(regions, pixels, output_pixels, dtype)
    if count == 1:
        return model, params, wavelength
    components = []
    for i in range(count):
        component = copy.deepcopy(params["components"][0])
        component["atmosphere"] = {"teff": 5750. + 20 * i, "logg": 4.2 + .02 * i,
                                    "feh": -.15 - .02 * i, "alpha": .1 + .03 * i}
        component["broadening"] = {"vsini": 6.3 + .7 * i, "vmacro": 3.1 + .25 * i,
                                    "u1": .5 - .02 * i, "u2": .2 + .01 * i}
        component["rv"] = component["rv"] + 2.17 * i
        components.append(component)
    params.update(components=tuple(components), flux_weights=np.array([1 / (i + 1) for i in range(count)]))
    # Dilution is omitted (zero). Unequal weights are dynamic AD inputs; every
    # stellar atmosphere, broadening and RV differs to prevent duplicate work.
    return model, jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), params), wavelength


def benchmark(counts, regions, pixels, output_pixels, dtype, rounds, warmup):
    device = jax.devices()[0]
    cases = []
    for count in counts:
        inputs = component_inputs(count, regions, pixels, output_pixels, dtype)
        model, params, wavelength = jax.device_put(inputs, device)
        jax.block_until_ready((model, params, wavelength))
        forward = lambda m, p, w: m(p, w)
        objective = lambda m, p, w: jnp.mean(m(p, w)**2 * jnp.linspace(.5, 1.5, w.shape[-1]))
        timings, memory, results = {}, {}, {}
        for name, function in (("forward", forward), ("value_and_grad", jax.value_and_grad(objective, argnums=1))):
            timings[name], memory[name], results[name] = measure(
                function, (model, params, wavelength), rounds=rounds, warmup=warmup)
        flux = np.asarray(results["forward"])
        value, gradient = results["value_and_grad"]
        assert flux.shape == (regions, output_pixels) and flux.dtype == dtype
        assert np.all(np.isfinite(flux)) and np.isfinite(value)
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
        cases.append({
            "N": count, "dtype": np.dtype(dtype).name,
            "parameters": jax.tree.map(lambda x: np.asarray(x).tolist(), params),
            "timing": timings, "memory": memory,
            "validation": {"objective": float(value), "finite_forward": True,
                           "finite_gradients": True, "gradient_leaves": len(jax.tree.leaves(gradient))},
        })
    return {
        "schema_version": 1, "benchmark": "specmodel_sbn", "environment": environment(device),
        "problem": {"regions": regions, "model_pixels_per_region": pixels,
                    "output_pixels_per_region": output_pixels, "grid_axes": 4, "nodes_per_axis": 2,
                    "velocity_step_kms": 1., "vmax_kms": model.vmax,
                    "model_leaf_bytes": sum(x.nbytes for x in jax.tree.leaves(model)),
                    "model_argument": "dynamic PyTree", "dilution": 0.},
        "timing": {"rounds": rounds, "warmup_calls": warmup}, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--model-pixels", type=int, default=4000)
    parser.add_argument("--output-pixels", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    required = parser.add_mutually_exclusive_group()
    required.add_argument("--require-cpu", action="store_true")
    required.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if any(n < 1 or n > 8 for n in args.components):
        parser.error("this benchmark's distinct atmosphere recipe supports 1 <= N <= 8; the model has no such limit")
    if min(args.regions, args.rounds, args.warmup) < 1 or args.model_pixels < 401 or args.output_pixels < 2:
        parser.error("regions/rounds/warmup must be positive, model-pixels >= 401, output-pixels >= 2")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    backend = jax.default_backend()
    if (args.require_cpu and backend != "cpu") or (args.require_gpu and backend != "gpu"):
        parser.error(f"required backend unavailable: active backend is {backend}")
    report = benchmark(args.components, args.regions, args.model_pixels, args.output_pixels,
                       np.dtype(args.dtype), args.rounds, args.warmup)
    emit_result(report, args.json_out)


if __name__ == "__main__":
    main()
