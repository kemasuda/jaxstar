"""Measure the current ``MistGridIso.values`` interpolation path.

Run from the repository root, for example:

    PYTHONPATH=src python benchmarks/benchmark_mistgrid.py

The reported times exclude JIT compilation. They are intended for comparing
the old and replacement implementations on the same machine, not as CI pass
or fail thresholds.
"""

import argparse
import platform
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from jaxstar.mistfit import MistGridIso


GRID_SHAPE = (48, 16, 256)
FIELD_NAMES = (
    "mass",
    "teff",
    "logg",
    "radius",
    "kmag",
    "jmag",
    "hmag",
    "dmdeep",
    "star_mass",
    "feh_photosphere",
)


def create_synthetic_grid(path):
    """Write a regular three-axis grid with ten scalar fields."""
    logage = np.linspace(8.0, 10.0, GRID_SHAPE[0], dtype=np.float32)
    feh = np.linspace(-1.0, 0.5, GRID_SHAPE[1], dtype=np.float32)
    eep = np.linspace(0.0, 600.0, GRID_SHAPE[2], dtype=np.float32)

    base = (
        logage[:, None, None]
        + 2.0 * feh[None, :, None]
        + eep[None, None, :] / 100.0
    )
    fields = {
        name: (index + 1) * base
        for index, name in enumerate(FIELD_NAMES)
    }
    np.savez(
        path,
        logagrid=logage,
        fgrid=feh,
        eepgrid=eep,
        **fields,
    )


def block_until_ready(values):
    """Synchronize every field because JAX dispatch is asynchronous."""
    for value in values:
        value.block_until_ready()


def median_runtime(function, rounds, warmups=2):
    for _ in range(warmups):
        block_until_ready(function())

    durations = []
    for _ in range(rounds):
        start = perf_counter()
        block_until_ready(function())
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def run(rounds, batch_size):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic_mistgrid.npz"
        create_synthetic_grid(path)

        grid = MistGridIso(path=path)
        grid.set_keys(FIELD_NAMES)

        def scalar_query():
            return grid.values(age=9.0, feh=-0.25, eep=300.0)

        batch_age = jnp.linspace(8.1, 9.9, batch_size)
        batch_feh = jnp.linspace(-0.9, 0.4, batch_size)
        batch_eep = jnp.linspace(10.0, 590.0, batch_size)

        def batch_query():
            return grid.values(
                age=batch_age,
                feh=batch_feh,
                eep=batch_eep,
            )

        scalar_seconds = median_runtime(scalar_query, rounds)
        batch_seconds = median_runtime(batch_query, rounds)

    print(f"Python: {platform.python_version()}")
    print(f"JAX: {jax.__version__}")
    print(f"jaxlib: {jaxlib.__version__}")
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Device: {jax.devices()[0]}")
    print(f"Grid: {GRID_SHAPE}, fields: {len(FIELD_NAMES)}, rounds: {rounds}")
    print(f"Scalar: {1e3 * scalar_seconds:.3f} ms/call")
    print(
        f"Batch ({batch_size}): {1e3 * batch_seconds:.3f} ms/call "
        f"({1e6 * batch_seconds / batch_size:.3f} us/query)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run(rounds=args.rounds, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
