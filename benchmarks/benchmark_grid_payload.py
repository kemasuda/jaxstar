"""Compile and time generic trailing-payload interpolation and coordinate AD.

Run from the repository root, for example:

    PYTHONPATH=src python benchmarks/benchmark_grid_payload.py
    PYTHONPATH=src python benchmarks/benchmark_grid_payload.py --dtype float64
    PYTHONPATH=src python benchmarks/benchmark_grid_payload.py --coordinate-dtype float64

Uses the default JAX device, so the same harness can run on CPU or GPU. Set
JAX_PLATFORMS=cpu to select CPU explicitly. Grid arrays are dynamic PyTree
arguments. Host setup/transfers and compilation are excluded from warm timings;
all results are synchronized. These measurements are not CI thresholds.
"""

import argparse
import json
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from jaxstar.grid import Field, RectilinearGrid


def block_until_ready(result):
    for leaf in jax.tree_util.tree_leaves(result):
        leaf.block_until_ready()


def make_grid(axis_count, nodes, regions, pixels, field_dtype, coordinate_dtype):
    """Small regular/nonuniform axes and contiguous, varying trailing blocks."""
    names = tuple(f"axis_{index}" for index in range(axis_count))
    axes = {}
    for index, name in enumerate(names):
        values = np.linspace(0, 1, nodes, dtype=coordinate_dtype)
        if index % 2:
            values = values**2
        axes[name] = values
    shape = (nodes,) * axis_count + (regions, pixels)
    values = np.random.default_rng(10 + axis_count).uniform(0.5, 1.5, shape).astype(field_dtype)
    return RectilinearGrid(axes=axes, fields={
        "block": Field(values, dims=names, payload_dims=("region", "pixel")),
    })


def forward(grid, points):
    coordinates = {name: points[..., index] for index, name in enumerate(grid.axis_names)}
    return grid.interpolate(coordinates, keys="block")["block"]


def objective(grid, points):
    """A scalar reduction through all payload elements; differentiate queries."""
    return jnp.mean((forward(grid, points) - 1.0)**2)


def measure(function, args, rounds):
    start = perf_counter()
    lowered = jax.jit(function).lower(*args)
    lowering_seconds = perf_counter() - start
    start = perf_counter()
    compiled = lowered.compile()
    compile_seconds = perf_counter() - start

    # First invocation and allocator/startup effects are outside warm timings.
    for _ in range(3):
        block_until_ready(compiled(*args))
    durations = []
    for _ in range(rounds):
        start = perf_counter()
        result = compiled(*args)
        block_until_ready(result)
        durations.append(perf_counter() - start)

    memory = compiled.memory_analysis()
    memory_bytes = None if memory is None else {
        name: getattr(memory, name, None) for name in (
            "argument_size_in_bytes", "output_size_in_bytes",
            "temp_size_in_bytes", "alias_size_in_bytes",
        )
    }
    return {
        "lowering_seconds": lowering_seconds,
        "xla_compile_seconds": compile_seconds,
        "compile_total_seconds": lowering_seconds + compile_seconds,
        "warm_median_ms": 1e3 * float(np.median(durations)),
        "warm_min_ms": 1e3 * min(durations),
        "memory_bytes": memory_bytes,
    }, result


def run(args):
    coordinate_dtype = args.coordinate_dtype or args.dtype
    if "float64" in (args.dtype, coordinate_dtype):
        jax.config.update("jax_enable_x64", True)
    report = {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": jaxlib.__version__, "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": jax.config.read("jax_enable_x64"),
        "field_dtype": args.dtype, "coordinate_dtype": coordinate_dtype,
        "rounds": args.rounds, "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    for axis_count in args.axes:
        grid = make_grid(axis_count, args.nodes, args.regions, args.pixels,
                         np.dtype(args.dtype), np.dtype(coordinate_dtype))
        block_until_ready(grid)
        for batch in (1, args.batch_size) if args.batch_size != 1 else (1,):
            points = jnp.asarray(np.linspace(0.3, 0.7, batch * axis_count).reshape(batch, axis_count),
                                 dtype=coordinate_dtype)
            if batch == 1:
                points = points[0]
            block_until_ready(points)
            case = {
                "axes": axis_count, "axis_kinds": grid.axis_kinds,
                "grid_shape": grid.field("block").values.shape,
                "grid_bytes": grid.field("block").values.nbytes,
                "query_shape": points.shape, "batch": batch,
            }
            case["forward"], values = measure(forward, (grid, points), args.rounds)
            case["value_and_grad"], (value, gradient) = measure(
                jax.value_and_grad(objective, argnums=1), (grid, points), args.rounds
            )
            assert values.shape == (() if batch == 1 else (batch,)) + (args.regions, args.pixels)
            assert values.dtype == np.dtype(args.dtype)
            assert np.all(np.isfinite(values)) and np.isfinite(value) and np.all(np.isfinite(gradient))
            case["output_shape"] = values.shape
            case["objective"] = float(value)
            case["gradient_norm"] = float(jnp.linalg.norm(gradient))
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--pixels", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--coordinate-dtype", choices=("float32", "float64"))
    parser.add_argument("--output", help="optional JSON results file")
    args = parser.parse_args()
    if args.nodes < 2 or any(value < 1 for value in (
        *args.axes, args.regions, args.pixels, args.batch_size, args.rounds
    )):
        parser.error("axes, regions, pixels, batch-size and rounds must be positive; nodes >= 2")
    run(args)


if __name__ == "__main__":
    main()
