"""Compile and time generic trailing-payload interpolation and coordinate AD.

Run from the repository root, for example:

    PYTHONPATH=src python benchmarks/benchmark_grid_payload.py
    PYTHONPATH=src python benchmarks/benchmark_grid_payload.py --dtype float64
    PYTHONPATH=src python benchmarks/benchmark_grid_payload.py --coordinate-dtype float64

Defaults: B=1, four two-node axes, payload (10, 4000), float32.
Uses the default JAX device, so the same harness can run on CPU or GPU. Set
JAX_PLATFORMS=cpu to select CPU explicitly. Grid arrays are dynamic PyTree
arguments. Host setup/transfers and compilation are excluded from warm timings;
all results are synchronized. These measurements are not CI thresholds.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from _benchmark_utils import emit_result, environment, measure


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


def run(args):
    coordinate_dtype = args.coordinate_dtype or args.dtype
    jax.config.update("jax_enable_x64", "float64" in (args.dtype, coordinate_dtype))
    device = jax.devices()[0]
    cases = []
    for axis_count in args.axes:
        grid = make_grid(axis_count, args.nodes, args.regions, args.pixels,
                         np.dtype(args.dtype), np.dtype(coordinate_dtype))
        for batch in (1, args.batch_size) if args.batch_size != 1 else (1,):
            points = jnp.asarray(np.linspace(0.3, 0.7, batch * axis_count).reshape(batch, axis_count),
                                 dtype=coordinate_dtype)
            if batch == 1:
                points = points[0]
            grid, points = jax.device_put((grid, points), device)
            jax.block_until_ready((grid, points))
            timing, memory = {}, {}
            timing["forward"], memory["forward"], values = measure(
                forward, (grid, points), rounds=args.rounds, warmup=args.warmup)
            timing["value_and_grad"], memory["value_and_grad"], (value, gradient) = measure(
                jax.value_and_grad(objective, argnums=1), (grid, points),
                rounds=args.rounds, warmup=args.warmup)
            assert values.shape == (() if batch == 1 else (batch,)) + (args.regions, args.pixels)
            assert values.dtype == np.dtype(args.dtype)
            assert np.all(np.isfinite(values)) and np.isfinite(value) and np.all(np.isfinite(gradient))
            cases.append({
                "problem": {
                    "dtype": args.dtype, "coordinate_dtype": coordinate_dtype,
                    "batch": batch, "regions": args.regions, "pixels_per_region": args.pixels,
                    "grid_axes": axis_count, "nodes_per_axis": args.nodes,
                    "payload_shape": [args.regions, args.pixels],
                    "grid_shape": list(grid.field("block").values.shape),
                    "grid_bytes": grid.field("block").values.nbytes,
                    "axis_kinds": list(grid.axis_kinds), "query_shape": list(points.shape),
                    "seed": 10 + axis_count,
                },
                "timing": {"rounds": args.rounds, "warmup_calls": args.warmup, **timing},
                "memory": memory,
                "validation": {
                    "output_shape": list(values.shape), "finite_forward_and_gradients": True,
                    "objective": float(value), "gradient_norm": float(jnp.linalg.norm(gradient)),
                },
            })
    report = {"schema_version": 1, "benchmark": "grid_payload", "environment": environment(device)}
    # Keep the normal single-case result directly comparable with SpecModel;
    # retain the old opt-in multi-axis/multi-batch capability in one JSON block.
    report.update(cases[0] if len(cases) == 1 else {"cases": cases})
    emit_result(report, args.json_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=int, nargs="+", default=[4])
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--pixels", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--coordinate-dtype", choices=("float32", "float64"))
    parser.add_argument("--json-out", "--output", dest="json_out", help="optional JSON results file")
    args = parser.parse_args()
    if args.nodes < 2 or any(value < 1 for value in (
        *args.axes, args.regions, args.pixels, args.batch_size, args.rounds, args.warmup
    )):
        parser.error("axes, regions, pixels, batch-size, rounds and warmup must be positive; nodes >= 2")
    run(args)


if __name__ == "__main__":
    main()
