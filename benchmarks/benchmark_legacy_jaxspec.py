#!/usr/bin/env python3
"""Rerun frozen jaxspec's real-IRD full_iid_value_and_grad, without GP imports.

Setup helpers and all spectral physics execute from the read-only sibling.
Only the iid objective/reporting live here; no jaxstar physical model is used.
The default grids are the historical Linux/A100 grids, not Mac sample grids.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np

from _benchmark_utils import emit_result, environment, measure


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_GRID_DIR = Path("/home/masuda/specgrid_irdh_coelho")
HISTORICAL_ORDERS = "8,9,10,11,12,13,14,15,16,17"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaxspec-root", type=Path, default=REPO_ROOT.parent / "jaxspec")
    parser.add_argument("--grid-dir", type=Path, default=HISTORICAL_GRID_DIR)
    parser.add_argument("--data-csv", type=Path,
                        help="Default: <jaxspec-root>/data/IRDA00042313_H.csv")
    parser.add_argument("--orders", default=HISTORICAL_ORDERS,
                        help="Comma-separated observed orders (default: 8–17).")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--wavelength-scale", type=float, default=10.)
    parser.add_argument("--wav-margin", type=float, default=3.)
    parser.add_argument("--vmax", type=float, default=50.)
    parser.add_argument("--vsini", type=float, default=5.)
    parser.add_argument("--zeta", type=float, default=2.)
    parser.add_argument("--wavres", type=float, default=70000.)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=30)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--require-gpu", action="store_true")
    backend.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup non-negative")
    for name in ("wavelength_scale", "vmax", "wavres"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("wav_margin", "vsini", "zeta"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    args.jaxspec_root = args.jaxspec_root.expanduser().resolve()
    args.grid_dir = args.grid_dir.expanduser().resolve()
    args.data_csv = (args.data_csv or args.jaxspec_root / "data/IRDA00042313_H.csv").expanduser().resolve()
    return args


def load_frozen_sources(root):
    """Execute unchanged sources without __init__, inference, or bytecode writes.

    The frozen pipeline's top level only imports standard-library modules and
    NumPy. Its GP imports are inside main(), which is never called here.
    Computational modules use the same isolated-package loading strategy as
    that pipeline, but compile source bytes directly to avoid sibling caches.
    This is import isolation, not a numerical compatibility shim.
    """
    hashes = {}

    def load(name, relative, package=""):
        path = root / relative
        source = path.read_bytes()
        hashes[relative] = hashlib.sha256(source).hexdigest()
        module = types.ModuleType(name)
        module.__file__, module.__package__ = str(path), package
        sys.modules[name] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        return module

    common = load("_jaxstar_legacy_pipeline", "benchmarks/benchmark_pipeline.py")
    namespace = "_jaxstar_legacy_jaxspec_source"
    package = types.ModuleType(namespace)
    package.__path__ = []
    sys.modules[namespace] = package
    modules = {}
    for name in ("kernels", "utils", "specgrid", "specmodel"):
        modules[name] = load(f"{namespace}.{name}", f"src/jaxspec/{name}.py", namespace)
    # Record the source of the objective, without importing its GP entry point.
    relative = "benchmarks/benchmark_best_gp_backend.py"
    hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return common, types.SimpleNamespace(**modules), hashes


def prepare_workload(args, device, common, modules):
    """Use the historical CSV, matching, grid casting and parameter helpers."""
    dtype = np.dtype(args.dtype)
    orders = common.parse_orders(args.orders)
    observations, orders, _ = common.load_observed_csv(
        args.data_csv, orders, len(orders), args.wavelength_scale)
    paths = common.resolve_grid_files(args.grid_dir, observations[0], len(orders), args.wav_margin)
    grid, model_name, _ = common.load_real_grid(paths, modules.specgrid)
    if model_name != "coelho":
        raise ValueError("This controlled legacy benchmark requires Coelho grids")
    common.cast_float_arrays(grid, dtype)
    wav, observed, error, mask = observations
    model = modules.specmodel.SpecModel(
        grid, np.asarray(wav, dtype=dtype), np.asarray(observed, dtype=dtype),
        np.asarray(error, dtype=dtype), np.asarray(mask, dtype=bool),
        vmax=args.vmax, gpu=device.platform in {"gpu", "cuda", "rocm"})
    common.cast_float_arrays(model, dtype)
    if model.varr.shape[1] < 3:
        raise ValueError("The frozen velocity kernel has fewer than 3 samples")
    host_parameters = common.make_parameters(grid, model.Norder, args, dtype)
    parameters = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), host_parameters)
    observed_device = jax.device_put(model.flux_obs, device)
    error_device = jax.device_put(model.error_obs, device)
    valid_device = jax.device_put(~model.mask_obs.astype(bool), device)
    jax.block_until_ready((parameters, observed_device, error_device, valid_device))

    # This is exactly the non-GP objective in benchmark_best_gp_backend.py.
    def iid_nll(flux_model):
        with jax.named_scope("iid_gaussian_nll"):
            residual = (observed_device - flux_model) / error_device
            return .5 * jnp.sum(jnp.where(valid_device, residual * residual, 0.))

    def full_iid_nll(current_parameters):
        return iid_nll(model.fluxmodel_multiorder(current_parameters))

    return types.SimpleNamespace(
        grid=grid, model=model, paths=paths, orders=orders,
        host_parameters=host_parameters, parameters=parameters,
        objective=full_iid_nll,
        value_and_grad=jax.value_and_grad(full_iid_nll))


def validate_result(result, parameters, reference, dtype):
    value, gradients = result
    if jax.tree_util.tree_structure(gradients) != jax.tree_util.tree_structure(parameters):
        raise AssertionError("Gradient PyTree differs from the full legacy parameter dict")
    for name, gradient in gradients.items():
        if gradient.shape != parameters[name].shape:
            raise AssertionError(f"Gradient shape mismatch for {name}")
    if not np.isfinite(np.asarray(value)).all() or not all(
        np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gradients)):
        raise FloatingPointError("Nonfinite legacy iid objective/gradients")
    tolerance = 1e-10 if dtype == "float64" else 2e-5
    np.testing.assert_allclose(np.asarray(value), reference, rtol=tolerance, atol=tolerance)


def main(argv=None):
    args = parse_args(argv)
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    device = jax.devices()[0]
    gpu_selected = device.platform in {"gpu", "cuda", "rocm"}
    if args.require_gpu and not gpu_selected:
        raise RuntimeError(f"--require-gpu requested, but selected device is {device}")
    if args.require_cpu and device.platform != "cpu":
        raise RuntimeError(f"--require-cpu requested, but selected device is {device}")
    if not args.grid_dir.is_dir():
        raise FileNotFoundError(
            f"Historical grid directory not found: {args.grid_dir}. "
            "Supply --grid-dir explicitly; no sample or synthetic fallback is used.")
    with jax.default_device(device):
        common, modules, hashes = load_frozen_sources(args.jaxspec_root)
        workload = prepare_workload(args, device, common, modules)
        model = workload.model
        parameters = workload.parameters

        # Independent host reduction and full AD sanity check precede timing.
        flux = jax.block_until_ready(model.fluxmodel_multiorder(parameters))
        flux_host = np.asarray(flux)
        if flux_host.shape != model.flux_obs.shape or not np.isfinite(flux_host).all():
            raise FloatingPointError("Nonfinite or incorrectly shaped legacy model")
        residual = (model.flux_obs - flux_host) / model.error_obs
        reference = float(.5 * np.sum(np.where(~model.mask_obs.astype(bool), residual**2, 0.)))
        preflight = jax.block_until_ready(jax.jit(workload.value_and_grad)(parameters))
        validate_result(preflight, parameters, reference, args.dtype)

        # The historical benchmark clears in-memory JIT caches for a cold row.
        jax.clear_caches()
        gc.collect()
        timing, memory, result = measure(
            workload.value_and_grad, (parameters,), rounds=args.repeat,
            warmup=args.warmup, time_first=True)
        validate_result(result, parameters, reference, args.dtype)

    parameter_report = {
        name: {"value": np.asarray(value).tolist(), "shape": list(value.shape),
               "dtype": str(value.dtype)}
        for name, value in sorted(workload.host_parameters.items())}
    file_shapes = []
    axis_bounds = []
    for path in workload.paths:
        with np.load(path) as grid:
            file_shapes.append(list(grid["flux"].shape))
            axis_bounds.append({
                name: [float(grid[name][0]), float(grid[name][-1])]
                for name in ("tgrid", "ggrid", "fgrid", "agrid") if name in grid})
    valid_counts = np.sum(~model.mask_obs.astype(bool), axis=1).astype(int).tolist()
    env = environment(device)
    env["backend"] = device.platform
    env["python_executable"] = sys.executable
    env["logical_cpu_count"] = os.cpu_count()
    env["runtime_environment"] = {
        name: os.environ.get(name) for name in (
            "JAX_PLATFORMS", "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "LD_LIBRARY_PATH")}
    report = {
        "schema_version": 1,
        "benchmark": "legacy_jaxspec_full_iid",
        "stage": "full_iid_value_and_grad",
        "environment": env,
        "source": {"jaxspec_root": str(args.jaxspec_root), "sha256": hashes,
                   "compatibility_shims": []},
        "problem": {
            "dtype": args.dtype, "orders": workload.orders, "n_orders": int(model.Norder),
            "data_csv": str(args.data_csv),
            "data_csv_sha256": hashlib.sha256(args.data_csv.read_bytes()).hexdigest(),
            "grid_dir": str(args.grid_dir), "grid_files": [str(p) for p in workload.paths],
            "grid_shapes": file_shapes, "combined_grid_shape": list(workload.grid.fluxgrid.shape),
            "grid_dtype": str(workload.grid.fluxgrid.dtype), "atmosphere_bounds": axis_bounds,
            "native_model_pixels_per_order": int(workload.grid.wavgrid.shape[1]),
            "log_model_pixels_per_order": int(model.wavgrid.shape[1]),
            "log_wavelength_step_per_order": model.dlogwav.tolist(),
            "velocity_step_kms_per_order": (model.dlogwav * modules.utils.c_in_kms).tolist(),
            "velocity_kernel_shape": list(model.varr.shape), "kernel_Nt": 500,
            "model_output_shape": list(flux_host.shape),
            "observed_pixels_per_order": [int(row.size) for row in model.wav_obs],
            "observed_pixels_total": int(model.wav_obs.size),
            "unmasked_pixels_per_order": valid_counts, "unmasked_pixels_total": sum(valid_counts),
            "wavelength_scale": args.wavelength_scale, "matching_margin_angstrom": args.wav_margin,
            "vmax_kms": args.vmax,
        },
        "parameters": parameter_report,
        "differentiation": {
            "parameter_names": sorted(parameters), "leaf_count": len(parameters),
            "scalar_entry_count": sum(int(p.size) for p in parameters.values()),
            "argnums": 0,
            "objective": "0.5 * sum(where(~mask_obs, ((flux_obs - flux_model) / error_obs)**2, 0))",
            "normalization_terms": "omitted, as in historical benchmark; no jitter or GP",
        },
        "timing": {"rounds": args.repeat, "warmup_calls": args.warmup,
                   "additional_first_execution_calls": 1, "value_and_grad": timing},
        "memory": {"value_and_grad": memory},
        "validation": {
            "objective": float(np.asarray(result[0])), "finite_gradients": True,
            "finite_model": True, "iid_objective_from_host_forward": reference,
            "iid_host_vs_value_and_grad_abs": abs(reference - float(np.asarray(result[0]))),
            "gradient_shapes_match_parameters": True,
            "tinygp_imported": any(n == "tinygp" or n.startswith("tinygp.") for n in sys.modules),
        },
        "historical_reference": {
            "a100_jax_version": "0.4.33", "warm_median_seconds": .001685,
            "iid_objective": None,
            "note": "Linux/A100 iid objective is not available in the frozen local artifacts. "
                    "The saved Mac objective uses a different grid and is not an A100 oracle.",
        },
        "notes": [
            "Frozen SpecGrid.values and SpecModel.fluxmodel_multiorder are unchanged.",
            "Large NumPy grid/model arrays remain captured by frozen static-self JIT. "
            "Their first-execution transfer is excluded from warm timing.",
            "CSV loading, grid matching/casting, input transfer, validation and compile are excluded from warm timing.",
            "Every timed result synchronizes the objective and all gradient leaves.",
            "No GP, inference, new jaxstar physics, or synthetic fallback is used.",
            "This real-data iid workload differs from the synthetic M3a benchmark; do not infer implementation speedup yet.",
        ],
    }
    emit_result(report, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
