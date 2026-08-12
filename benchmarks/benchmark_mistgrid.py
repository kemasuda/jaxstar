"""Measure the current ``MistGridIso.values`` interpolation path.

Run from the repository root, for example:

    PYTHONPATH=src python benchmarks/benchmark_mistgrid.py

The reported times exclude JIT compilation. They compare the legacy MIST
kernel and the generic grid core in the same process on identical arrays.
They are not CI pass or fail thresholds.
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

from jaxstar.grid import Field, RectilinearGrid
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
    return logage, feh, eep, fields


def block_until_ready(values):
    """Synchronize every field because JAX dispatch is asynchronous."""
    for value in jax.tree_util.tree_leaves(values):
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
        logage, feh, eep, fields = create_synthetic_grid(path)

        legacy_grid = MistGridIso(path=path)
        legacy_grid.set_keys(FIELD_NAMES)
        core_grid = RectilinearGrid(
            axes={"age": logage, "feh": feh, "eep": eep},
            fields={
                name: Field(values, dims=("age", "feh", "eep"))
                for name, values in fields.items()
            },
        )

        def legacy_scalar_query():
            return legacy_grid.values(age=9.0, feh=-0.25, eep=300.0)

        @jax.jit
        def core_query(age, feh, eep):
            return core_grid.interpolate(
                {"age": age, "feh": feh, "eep": eep},
                keys=FIELD_NAMES,
            )

        def core_scalar_query():
            return core_query(9.0, -0.25, 300.0)

        batch_age = jnp.linspace(8.1, 9.9, batch_size)
        batch_feh = jnp.linspace(-0.9, 0.4, batch_size)
        batch_eep = jnp.linspace(10.0, 590.0, batch_size)

        def legacy_batch_query():
            return legacy_grid.values(
                age=batch_age,
                feh=batch_feh,
                eep=batch_eep,
            )

        def core_batch_query():
            return core_query(batch_age, batch_feh, batch_eep)

        legacy_scalar = legacy_scalar_query()
        core_scalar = core_scalar_query()
        legacy_batch = legacy_batch_query()
        core_batch = core_batch_query()
        block_until_ready((legacy_scalar, core_scalar, legacy_batch, core_batch))
        maximum_absolute_difference = 0.0
        for index, name in enumerate(FIELD_NAMES):
            legacy_values = np.asarray(legacy_scalar[index])
            core_values = np.asarray(core_scalar[name])
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(np.max(np.abs(legacy_values - core_values))),
            )
            np.testing.assert_allclose(
                legacy_values,
                core_values,
                rtol=1e-5,
                atol=1e-6,
            )

            legacy_values = np.asarray(legacy_batch[index])
            core_values = np.asarray(core_batch[name])
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(np.max(np.abs(legacy_values - core_values))),
            )
            np.testing.assert_allclose(
                legacy_values,
                core_values,
                rtol=1e-5,
                atol=1e-6,
            )

        legacy_scalar_seconds = median_runtime(legacy_scalar_query, rounds)
        core_scalar_seconds = median_runtime(core_scalar_query, rounds)
        legacy_batch_seconds = median_runtime(legacy_batch_query, rounds)
        core_batch_seconds = median_runtime(core_batch_query, rounds)

    print(f"Python: {platform.python_version()}")
    print(f"JAX: {jax.__version__}")
    print(f"jaxlib: {jaxlib.__version__}")
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Device: {jax.devices()[0]}")
    print(f"Grid: {GRID_SHAPE}, fields: {len(FIELD_NAMES)}, rounds: {rounds}")
    print(f"Maximum absolute value difference: {maximum_absolute_difference:.3g}")
    print(f"Legacy scalar: {1e3 * legacy_scalar_seconds:.3f} ms/call")
    print(f"Core scalar: {1e3 * core_scalar_seconds:.3f} ms/call")
    print(
        f"Core / legacy scalar: "
        f"{core_scalar_seconds / legacy_scalar_seconds:.3f}x "
        "(greater than 1 is slower)"
    )
    print(
        f"Legacy batch ({batch_size}): "
        f"{1e3 * legacy_batch_seconds:.3f} ms/call "
        f"({1e6 * legacy_batch_seconds / batch_size:.3f} us/query)"
    )
    print(
        f"Core batch ({batch_size}): {1e3 * core_batch_seconds:.3f} ms/call "
        f"({1e6 * core_batch_seconds / batch_size:.3f} us/query)"
    )
    print(
        f"Core / legacy batch: {core_batch_seconds / legacy_batch_seconds:.3f}x "
        "(greater than 1 is slower)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run(rounds=args.rounds, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
