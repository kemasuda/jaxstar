"""Synchronized full single-component CPU/GPU benchmark (standard B=1 case).

PYTHONPATH=src python benchmarks/benchmark_specmodel.py
Defaults: 10 regions, 4000 model pixels, 2000 output pixels, float32.
No data download or inference. The active JAX backend is used, not forced to CPU.
"""

import argparse
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import SpecModel
from jaxstar.specfit._data import _SpectralLibrary
from _benchmark_utils import emit_result, environment, measure


def make_inputs(regions=10, pixels=4000, output_pixels=2000, dtype=np.float32):
    velocity = np.arange(pixels) - (pixels - 1) / 2
    wavelength = np.linspace(15000., 22000., regions)[:, None] * np.exp(velocity / 299792.458)
    axes = {"teff": [5500., 6000.], "logg": [4., 4.5], "feh": [-.5, 0.], "alpha": [0., .4]}
    lines = sum(np.exp(-.5 * ((velocity - center) / width)**2)
                for center, width in ((-900, 4), (-200, 7), (350, 3), (1150, 9)))
    flux = np.empty((2, 2, 2, 2, regions, pixels), dtype=dtype)
    for node in itertools.product(range(2), repeat=4):
        flux[node] = 1 - (.15 + .025 * sum(node)) * lines
    grid = RectilinearGrid(axes={name: np.asarray(value, dtype=dtype) for name, value in axes.items()},
        fields={"flux": Field(flux, tuple(axes), payload_dims=("region", "pixel"))})
    library = _SpectralLibrary(grid, wavelength.astype(dtype), tuple(range(regions)), "benchmark", "vacuum", "normalized")
    params = {"components": ({"atmosphere": {"teff": 5750., "logg": 4.2, "feh": -.15, "alpha": .1},
               "broadening": {"vsini": 6.3, "vmacro": 3.1, "u1": .5, "u2": .2},
               "rv": np.linspace(-15., 15., regions)},),
               "instrument": {"resolving_power": np.linspace(68000., 72000., regions)}}
    # Evaluation geometry is independent of detector/model sample counts.
    # 150 native samples of margin cover the stated kernel support and RVs.
    observed = np.stack([np.linspace(row[150], row[-151], output_pixels)
                         for row in np.asarray(library.wavelength)]).astype(dtype)
    return SpecModel(library), jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), params), observed


def benchmark(regions, pixels, output_pixels, rounds, dtype, warmup=3):
    model, params, wavelength = make_inputs(regions, pixels, output_pixels, dtype)
    device = jax.devices()[0]
    model, params, wavelength = jax.device_put((model, params, wavelength), device)
    jax.block_until_ready((model, params, wavelength))
    forward = lambda m, p, w: m(p, w)
    # A nonlinear deterministic functional exercises physical derivatives;
    # this benchmark has no observed flux, uncertainty or likelihood.
    objective = lambda m, p, w: jnp.mean(m(p, w)**2 * jnp.linspace(.5, 1.5, w.shape[-1]))
    timings, memory, results = {}, {}, {}
    for name, function in (("forward", forward), ("value_and_grad", jax.value_and_grad(objective, argnums=1))):
        timings[name], memory[name], results[name] = measure(
            function, (model, params, wavelength), rounds=rounds, warmup=warmup)
    flux = np.asarray(results["forward"])
    value, gradient = results["value_and_grad"]
    assert flux.shape == (regions, output_pixels) and flux.dtype == dtype
    assert np.all(np.isfinite(flux)) and np.isfinite(value)
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(gradient))
    component = params["components"][0]
    return {
        "schema_version": 1,
        "benchmark": "specmodel",
        "environment": environment(device),
        "problem": {
            "dtype": np.dtype(dtype).name, "batch": 1, "regions": regions,
            "model_pixels_per_region": pixels, "output_pixels_per_region": output_pixels,
            "grid_axes": len(model.spectra.grid.axis_names), "nodes_per_axis": 2,
            "velocity_step_kms": 1.0, "vmax_kms": model.vmax,
            "model_leaf_bytes": sum(x.nbytes for x in jax.tree_util.tree_leaves(model)),
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
        "timing": {"rounds": rounds, "warmup_calls": warmup, **timings},
        "memory": memory,
        "validation": {"finite_forward_and_gradients": True, "objective": float(value)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--model-pixels", "--pixels", dest="pixels", type=int, default=4000)
    parser.add_argument("--output-pixels", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--json-out", help="optional JSON file, e.g. benchmark-results/specmodel.json")
    args = parser.parse_args()
    if min(args.regions, args.rounds, args.warmup) < 1 or args.pixels < 401 or args.output_pixels < 2:
        parser.error("regions/rounds/warmup must be positive, model-pixels >= 401, output-pixels >= 2")
    # Public configuration API works on both older JAX and JAX 0.11.
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    result = benchmark(args.regions, args.pixels, args.output_pixels, args.rounds,
                       np.dtype(args.dtype), args.warmup)
    emit_result(result, args.json_out)


if __name__ == "__main__":
    main()
