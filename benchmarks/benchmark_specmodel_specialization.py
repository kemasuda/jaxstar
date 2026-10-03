"""Compare dynamic-PyTree and closure-captured SpecModel execution only.

Uses benchmark_specmodel's identical standard synthetic workload and objective.
Model/library capture is the only difference; parameters and requested
wavelengths remain dynamic in both modes. No public API or physics changes.
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from benchmark_specmodel import make_inputs
from _benchmark_utils import emit_result, environment, measure


def compare_outputs(dynamic, specialized, dtype):
    """Check values and every parameter gradient after all timing has finished."""
    rtol, atol = (5e-5, 2e-6) if np.dtype(dtype) == np.dtype("float32") else (1e-10, 1e-12)

    def compare(left, right):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError("Dynamic/specialized output shape or dtype mismatch")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise FloatingPointError("Nonfinite dynamic/specialized output")
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)
        return float(np.max(np.abs(left - right)))

    flux_difference = compare(dynamic["forward"], specialized["forward"])
    dynamic_value, dynamic_gradient = dynamic["value_and_grad"]
    specialized_value, specialized_gradient = specialized["value_and_grad"]
    value_difference = compare(dynamic_value, specialized_value)
    if jax.tree_util.tree_structure(dynamic_gradient) != jax.tree_util.tree_structure(specialized_gradient):
        raise AssertionError("Dynamic/specialized gradient PyTree mismatch")
    gradient_differences = {}
    left_leaves, _ = jax.tree_util.tree_flatten_with_path(dynamic_gradient)
    right_leaves = jax.tree_util.tree_leaves(specialized_gradient)
    for (path, left), right in zip(left_leaves, right_leaves, strict=True):
        gradient_differences[jax.tree_util.keystr(path)] = compare(left, right)
    return {
        "finite_forward_and_gradients": True,
        "equal_within_tolerance": True,
        "rtol": rtol, "atol": atol,
        "forward_max_abs_difference": flux_difference,
        "objective_abs_difference": value_difference,
        "gradient_max_abs_difference": max(gradient_differences.values()),
        "gradient_max_abs_difference_per_leaf": gradient_differences,
        "gradient_leaf_count": len(left_leaves),
        "dynamic_objective": float(dynamic_value),
        "specialized_objective": float(specialized_value),
    }


def benchmark(regions=10, pixels=4000, output_pixels=2000, rounds=20,
              dtype=np.float32, warmup=3, device=None):
    device = jax.devices()[0] if device is None else device
    with jax.default_device(device):
        model, params, wavelength = make_inputs(regions, pixels, output_pixels, dtype)
        model, params, wavelength = jax.device_put((model, params, wavelength), device)
        jax.block_until_ready((model, params, wavelength))

        # These are the same forward and scalar objective as benchmark_specmodel.
        dynamic_forward = lambda m, p, w: m(p, w)
        dynamic_objective = lambda m, p, w: jnp.mean(m(p, w)**2 * jnp.linspace(.5, 1.5, w.shape[-1]))
        specialized_forward = lambda p, w: dynamic_forward(model, p, w)
        specialized_objective = lambda p, w: dynamic_objective(model, p, w)
        cases = (
            ("dynamic_model", dynamic_forward, jax.value_and_grad(dynamic_objective, argnums=1),
             (model, params, wavelength), ["model", "params", "wavelength"], []),
            ("specialized_model", specialized_forward, jax.value_and_grad(specialized_objective, argnums=0),
             (params, wavelength), ["params", "wavelength"], ["model"]),
        )
        modes, results = {}, {}
        for name, forward, value_and_grad, args, dynamic_names, captured_names in cases:
            mode = {"dynamic_arguments": dynamic_names, "captured": captured_names, "memory": {}}
            results[name] = {}
            for stage, function in (("forward", forward), ("value_and_grad", value_and_grad)):
                mode[stage], mode["memory"][stage], results[name][stage] = measure(
                    function, args, rounds=rounds, warmup=warmup)
            modes[name] = mode

    validation = compare_outputs(results["dynamic_model"], results["specialized_model"], dtype)
    for result in results.values():
        flux = result["forward"]
        if flux.shape != (regions, output_pixels) or flux.dtype != np.dtype(dtype):
            raise AssertionError("Benchmark forward shape/dtype differs from requested workload")
        if jax.tree_util.tree_structure(result["value_and_grad"][1]) != jax.tree_util.tree_structure(params):
            raise AssertionError("Benchmark gradient does not match the parameter PyTree")
    comparison = {}
    for stage in ("forward", "value_and_grad"):
        dynamic_timing = modes["dynamic_model"][stage]
        specialized_timing = modes["specialized_model"][stage]
        dynamic_warm = dynamic_timing["warm_median_seconds"]
        specialized_warm = specialized_timing["warm_median_seconds"]
        comparison[f"{stage}_speedup_specialized_over_dynamic"] = dynamic_warm / specialized_warm
        comparison[f"{stage}_warm_improvement_percent"] = 100 * (1 - specialized_warm / dynamic_warm)
        comparison[f"{stage}_compile_ratio"] = specialized_timing["compile_seconds"] / dynamic_timing["compile_seconds"]
    component = params["components"][0]
    env = environment(device)
    env["python_executable"] = sys.executable
    env["runtime_environment"] = {
        name: os.environ.get(name) for name in (
            "JAX_PLATFORMS", "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR",
            "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")}
    return {
        "schema_version": 1,
        "benchmark": "specmodel_specialization",
        "environment": env,
        "problem": {
            "dtype": np.dtype(dtype).name, "batch": 1, "regions": regions,
            "model_pixels_per_region": pixels, "output_pixels_per_region": output_pixels,
            "grid_axes": len(model.spectra.grid.axis_names), "nodes_per_axis": 2,
            "velocity_step_kms": 1.0, "vmax_kms": model.vmax,
            "model_leaf_bytes": sum(x.nbytes for x in jax.tree_util.tree_leaves(model)),
            "model_leaf_count": len(jax.tree_util.tree_leaves(model)),
            "wavelength_shape": list(wavelength.shape), "wavelength_dynamic_in_both_modes": True,
            "objective": "mean(model(params, wavelength)**2 * linspace(0.5, 1.5, wavelength.shape[-1]))",
        },
        "parameters": {
            "atmosphere": {key: float(value) for key, value in component["atmosphere"].items()},
            "vsini_kms": float(component["broadening"]["vsini"]),
            "vmacro_kms": float(component["broadening"]["vmacro"]),
            "u1": float(component["broadening"]["u1"]),
            "u2": float(component["broadening"]["u2"]),
            "rv_kms": np.asarray(component["rv"]).tolist(),
            "resolving_power": np.asarray(params["instrument"]["resolving_power"]).tolist(),
        },
        "timing": {"rounds": rounds, "warmup_calls": warmup,
                   "mode_order": [case[0] for case in cases]},
        "modes": modes,
        "comparison": comparison,
        "validation": validation,
        "notes": [
            "The same model, parameter and wavelength arrays are constructed/placed once and reused in both modes.",
            "Only model/library is captured; requested wavelengths remain dynamic. No fully captured wavelength experiment is run.",
            "Speedup is dynamic/specialized warm median; >1 is faster. Compile ratio is specialized/dynamic lowering+XLA time.",
            "Every timed forward/objective/gradient result is synchronized; setup, transfers and compilation are excluded from warm timing.",
            "Memory is the existing public compiled-executable analysis, not peak runtime memory; unavailable values are null.",
            "Closure capture does not prescribe how this JAX/compiler version represents constants internally.",
            "This is the existing nonlinear synthetic M3a objective, not the legacy iid likelihood. No architecture decision follows automatically.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--model-pixels", type=int, default=4000)
    parser.add_argument("--output-pixels", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--require-gpu", action="store_true")
    backend.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    if min(args.regions, args.rounds, args.warmup) < 1 or args.model_pixels < 401 or args.output_pixels < 2:
        parser.error("regions/rounds/warmup must be positive, model-pixels >= 401, output-pixels >= 2")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    device = jax.devices()[0]
    if args.require_gpu and device.platform not in {"gpu", "cuda", "rocm"}:
        raise RuntimeError(f"--require-gpu requested, but selected device is {device}")
    if args.require_cpu and device.platform != "cpu":
        raise RuntimeError(f"--require-cpu requested, but selected device is {device}")
    report = benchmark(args.regions, args.model_pixels, args.output_pixels, args.rounds,
                       np.dtype(args.dtype), args.warmup, device)
    emit_result(report, args.json_out)


if __name__ == "__main__":
    main()
