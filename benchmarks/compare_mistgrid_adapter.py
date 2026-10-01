"""Compare MistGridIso and NamedMistGridIso, including both MistFit backends.

Run from the repository root with a working project environment:

    PYTHONPATH=src python benchmarks/compare_mistgrid_adapter.py
    PYTHONPATH=src python benchmarks/compare_mistgrid_adapter.py --grid /path/to/mistgrid_iso.npz
    PYTHONPATH=src python benchmarks/compare_mistgrid_adapter.py --grid /path/to/mistgrid_iso.npz --typical-fields
    PYTHONPATH=src python benchmarks/compare_mistgrid_adapter.py --grid /path/to/mistgrid_iso.npz --mistfit --rounds 200
    PYTHONPATH=src python benchmarks/compare_mistgrid_adapter.py --grid /path/to/mistgrid_iso.npz --inverse --rounds 200

The optional real-grid comparison reads an existing NPZ and never generates or
downloads one. MistGridIso remains the legacy numerical baseline.
Known differences are printed rather than treated as compatibility checks.

Runtime results are warm-call medians with JAX work synchronized. ``direct``
includes each Python values() call; ``outer_jit`` measures the same lookup
inside a larger compiled function. ``core_closure`` measures the grid directly,
as in benchmark_mistgrid.py. ``--mistfit`` additionally compares the actual
MistFit model's unconstrained negative log joint and gradient, as used by NUTS,
and runs a short sampling smoke check. Observations are synthetic, generated
from the supplied real grid. This is not a posterior convergence assessment.
Float32 gradient differences are reported, without claiming exact equivalence.
Repeat with JAX_ENABLE_X64=1 to check gradients at tighter tolerances.
``--inverse`` compares mass-to-EEP and values_given_mass, including runtime.
Inverse gradients on a real curve containing invalid fields are diagnostic;
they may be nonfinite in both implementations.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.mistfit import MistFit, MistGridIso, NamedMistGridIso


TYPICAL_MISTFIT_FIELDS = (
    "kmag", "teff", "logg", "mass", "radius", "feh_photosphere",
    "star_mass", "dmdeep", "mmin", "mmax", "bpmag2", "rpmag2",
)


class NamedGridAdapter:
    """Expose legacy-shaped calls solely to reuse interpolation comparisons.

    This wraps the public named API; set_keys only records the next selection.
    MistFit comparisons below use grid_backend directly and need no adapter.
    """

    def __init__(self, path):
        self.named = NamedMistGridIso(path=path)
        self._grid = self.named._grid
        self.keys = None

    def set_keys(self, keys):
        self.keys = tuple(keys)

    def values(self, age, feh, eep):
        result = self.named.values(age, feh, eep, keys=self.keys)
        return [result[name] for name in self.keys]


def create_tiny_npz(path):
    age = np.array([8.0, 9.0], dtype=np.float32)
    feh = np.array([-0.5, 0.5], dtype=np.float32)
    eep = np.array([100.0, 200.0], dtype=np.float32)
    a, f, e = np.meshgrid(age, feh, eep, indexing="ij")
    mass = a + 2.0 * f + e / 100.0
    teff = 5000.0 + 100.0 * mass
    invalid = mass.copy()
    invalid[-1, -1, -1] = -np.inf
    np.savez(
        path,
        logagrid=age,
        fgrid=feh,
        eepgrid=eep,
        mass=mass,
        teff=teff,
        invalid=invalid,
    )


def compare_equal(label, legacy, named, coordinates):
    old = legacy.values(*coordinates)
    new = named.values(*coordinates)
    assert len(old) == len(new) == len(named.keys)
    for old_field, new_field in zip(old, new):
        np.testing.assert_allclose(old_field, new_field, rtol=1e-5, atol=1e-6)
    print(f"{label}: equal; {[np.asarray(field) for field in new]}")


def report_difference(label, legacy, named, coordinates):
    old = [np.asarray(field) for field in legacy.values(*coordinates)]
    new = [np.asarray(field) for field in named.values(*coordinates)]
    print(f"{label}: legacy={old}, named={new}")


def block_until_ready(values):
    for value in jax.tree_util.tree_leaves(values):
        value.block_until_ready()


def benchmark_pair(label, legacy, named, scalar, batch, rounds):
    """Time direct and outer-jitted calls after warm-up and JAX synchronization."""
    if rounds == 0:
        return

    @jax.jit
    def legacy_outer(age, feh, eep):
        return legacy.values(age, feh, eep)

    @jax.jit
    def named_outer(age, feh, eep):
        return named.values(age, feh, eep)

    @jax.jit
    def core_closure(age, feh, eep):
        return named._grid.interpolate(
            {"age": age, "feh": feh, "eep": eep}, keys=named.keys
        )

    for style, implementations in (
        ("direct", (legacy.values, named.values)),
        ("outer_jit", (legacy_outer, named_outer)),
        ("core_closure", (legacy.values, core_closure)),
    ):
        for case, coordinates in (("scalar", scalar), ("batch", batch)):
            functions = tuple(
                lambda implementation=implementation: implementation(*coordinates)
                for implementation in implementations
            )
            for _ in range(3):
                for function in functions:
                    block_until_ready(function())

            timings = ([], [])
            for round_index in range(rounds):
                for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                    start = perf_counter()
                    block_until_ready(functions[index]())
                    timings[index].append(perf_counter() - start)

            old_seconds, new_seconds = (
                float(np.median(times)) for times in timings
            )
            print(
                f"{label} {style} {case}: "
                f"legacy={old_seconds * 1e3:.3f} ms, "
                f"named={new_seconds * 1e3:.3f} ms, "
                f"named/legacy={new_seconds / old_seconds:.3f}x "
                "(greater than 1 is slower)"
            )


def compare_synthetic(rounds, batch_size):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tiny_mistgrid.npz"
        create_tiny_npz(path)

        legacy = MistGridIso(path=path)
        named = NamedGridAdapter(path=path)
        for grid in (legacy, named):
            grid.set_keys(["mass", "teff"])

        compare_equal("interior", legacy, named, (8.5, 0.0, 150.0))
        compare_equal("lower corner", legacy, named, (8.0, -0.5, 100.0))
        compare_equal("outside", legacy, named, (7.9, 0.0, 150.0))
        compare_equal(
            "broadcast batch",
            legacy,
            named,
            (
                8.5,
                jnp.array([[-0.25], [0.25]]),
                jnp.array([[125.0, 150.0, 175.0]]),
            ),
        )

        @jax.jit
        def legacy_mass(point):
            return legacy.values(*point)[0]

        @jax.jit
        def named_mass(point):
            return named.values(*point)[0]

        point = jnp.array([8.5, 0.0, 150.0], dtype=jnp.float32)
        np.testing.assert_allclose(legacy_mass(point), named_mass(point))
        np.testing.assert_allclose(
            jax.grad(legacy_mass)(point),
            jax.grad(named_mass)(point),
            rtol=1e-5,
        )
        print("JIT and gradient: equal")

        # Existing characterization tests record this upper-boundary difference.
        report_difference("exact upper age node", legacy, named, (9.0, 0.0, 150.0))

        # MIST files use -inf for invalid cells. This is reported, not approved
        # as a numerical policy for the eventual adapter.
        legacy_invalid = MistGridIso(path=path)
        named_invalid = NamedGridAdapter(path=path)
        for grid in (legacy_invalid, named_invalid):
            grid.set_keys(["invalid"])
        report_difference(
            "cell next to -inf", legacy_invalid, named_invalid,
            (8.75, 0.25, 175.0),
        )
        report_difference(
            "finite corner next to -inf", legacy_invalid, named_invalid,
            (8.0, -0.5, 100.0),
        )

        # The old method's static-self JIT cache can keep the first key set.
        legacy_selection = MistGridIso(path=path)
        named_selection = NamedGridAdapter(path=path)
        for grid in (legacy_selection, named_selection):
            grid.set_keys(["mass"])
            grid.values(8.5, 0.0, 150.0)[0].block_until_ready()
            grid.set_keys(["teff"])
        report_difference(
            "set_keys after first call", legacy_selection, named_selection,
            (8.5, 0.0, 150.0),
        )

        benchmark_pair(
            "synthetic 2 fields", legacy, named,
            (8.5, 0.0, 150.0),
            (
                jnp.full(batch_size, 8.5),
                jnp.linspace(-0.4, 0.4, batch_size),
                jnp.linspace(110.0, 190.0, batch_size),
            ),
            rounds,
        )


def finite_cells(finite):
    """Find cells whose eight corners all have finite selected field values."""
    shape = tuple(size - 1 for size in finite.shape)
    cells = np.ones(shape, dtype=bool)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                cells &= finite[
                    i:i + shape[0], j:j + shape[1], k:k + shape[2]
                ]
    return cells


def sample_cell_centers(axes, flat_indices, cell_shape, count):
    chosen = flat_indices[
        np.linspace(0, flat_indices.size - 1, num=count, dtype=int)
    ]
    index_arrays = np.unravel_index(chosen, cell_shape)
    return tuple(
        jnp.asarray((axis[indices] + axis[indices + 1]) / 2)
        for axis, indices in zip(axes, index_arrays)
    ), index_arrays


def compare_real_grid(path, rounds, batch_size, typical_fields):
    """Compare selected real MIST fields at reproducible finite cell centers."""
    field_names = ("mass", "teff", "kmag")
    with np.load(path) as data:
        axes = tuple(data[name] for name in ("logagrid", "fgrid", "eepgrid"))
        finite = np.logical_and.reduce(
            [np.isfinite(data[name]) for name in field_names]
        )
        valid_cells = finite_cells(finite)
        valid_indices = np.flatnonzero(valid_cells)
        if valid_indices.size == 0:
            raise ValueError("no cells with eight finite corners")
        coordinates, index_arrays = sample_cell_centers(
            axes, valid_indices, valid_cells.shape, 32
        )
        batch_coordinates, _ = sample_cell_centers(
            axes, valid_indices, valid_cells.shape, batch_size
        )

        invalid_indices = np.argwhere(~finite)
        invalid_coordinates = None
        if invalid_indices.size:
            index = invalid_indices[len(invalid_indices) // 2]
            invalid_coordinates = tuple(axis[i] for axis, i in zip(axes, index))

    legacy = MistGridIso(path=path)
    named = NamedGridAdapter(path=path)
    for grid in (legacy, named):
        grid.set_keys(field_names)

    old = legacy.values(*coordinates)
    new = named.values(*coordinates)
    print(f"real NPZ: {path}")
    print(f"axes: {[len(axis) for axis in axes]}; finite cells: {valid_indices.size}")
    print(f"adapter axis kinds: {named._grid.axis_kinds}")
    print(f"selected finite cell centers: {len(coordinates[0])}")
    for name, old_field, new_field in zip(field_names, old, new):
        old_values = np.asarray(old_field)
        new_values = np.asarray(new_field)
        np.testing.assert_allclose(old_values, new_values, rtol=1e-5, atol=1e-4)
        difference = np.max(np.abs(old_values - new_values))
        print(f"{name}: equal within tolerance; max absolute difference={difference:.6g}")

    # Include one actual finite node and the global upper age boundary. These
    # may show the legacy upper-boundary behavior recorded in the tests.
    sample_index = tuple(
        indices[len(coordinates[0]) // 2] for indices in index_arrays
    )
    finite_node = tuple(axis[i] for axis, i in zip(axes, sample_index))
    report_difference("real finite node", legacy, named, finite_node)
    upper_age = (axes[0][-1], coordinates[1][len(coordinates[0]) // 2],
                 coordinates[2][len(coordinates[0]) // 2])
    report_difference("real upper age boundary", legacy, named, upper_age)
    if invalid_coordinates is not None:
        report_difference(
            "real nonfinite field node", legacy, named, invalid_coordinates
        )

    scalar = tuple(query[len(query) // 2] for query in coordinates)
    benchmark_pair(
        "real NPZ 3 fields", legacy, named, scalar, batch_coordinates, rounds
    )

    if typical_fields:
        with np.load(path) as data:
            finite_typical = np.ones(tuple(len(axis) for axis in axes), dtype=bool)
            for name in TYPICAL_MISTFIT_FIELDS:
                finite_typical &= np.isfinite(data[name])
            typical_cells = finite_cells(finite_typical)
            typical_indices = np.flatnonzero(typical_cells)
            if typical_indices.size == 0:
                raise ValueError("no cells finite for all typical MistFit fields")
            typical_batch, _ = sample_cell_centers(
                axes, typical_indices, typical_cells.shape, batch_size
            )
        typical_scalar = tuple(query[len(query) // 2] for query in typical_batch)
        legacy_typical = MistGridIso(path=path)
        named_typical = NamedGridAdapter(path=path)
        for grid in (legacy_typical, named_typical):
            grid.set_keys(TYPICAL_MISTFIT_FIELDS)
        for old_field, new_field in zip(
            legacy_typical.values(*typical_scalar),
            named_typical.values(*typical_scalar),
        ):
            np.testing.assert_allclose(old_field, new_field, rtol=1e-5, atol=1e-4)
        print(f"typical MistFit fields: {len(TYPICAL_MISTFIT_FIELDS)}")
        benchmark_pair(
            "real NPZ typical fields", legacy_typical, named_typical,
            typical_scalar, typical_batch, rounds,
        )


def compare_inverse(path, rounds):
    """Use the same scalar mass queries and retained interp() in both classes."""
    fields = ("mass", "teff", "kmag")
    legacy, named = MistGridIso(path=path), NamedMistGridIso(path=path)
    legacy.set_keys(fields)
    age, feh = 9.275, -0.125  # Same interior age/metallicity as compare_mistfit.
    # Include two low-mass brackets near the invalid prefix and two ordinary
    # interior queries. These have enough mass variation to compare tightly.
    for mass in (0.1068168879, 0.1105020717, 1.0, 1.31):
        point = (age, feh, mass)
        old_eep = legacy.eep_given_mass(*point)[0]
        new_eep = named.eep_given_mass(*point)
        old_fields = legacy.values_given_mass(*point)
        new_fields = named.values_given_mass(*point, keys=fields)
        tolerance = (
            dict(rtol=1e-7, atol=1e-8) if jax.config.read("jax_enable_x64")
            else dict(rtol=1e-5, atol=1e-4)
        )
        np.testing.assert_allclose(old_eep, new_eep, **tolerance)
        np.testing.assert_allclose(old_fields, list(new_fields.values()), **tolerance)
        assert np.isfinite(old_eep) and np.all(np.isfinite(old_fields))
        print(f"inverse reference mass={mass}: values agree; "
              f"EEP legacy={float(old_eep):.9g}, named={float(new_eep):.9g}")

    eep_nodes = jnp.asarray(legacy.dgrid["eepgrid"])
    curves = np.asarray(legacy.values(age, feh, eep_nodes))
    finite = np.all(np.isfinite(curves), axis=0)
    intervals = np.flatnonzero(
        finite[:-1] & finite[1:] & (np.diff(curves[0]) > 0)
    )
    if intervals.size == 0:
        raise ValueError("no finite increasing mass intervals at age=9.275, feh=-0.125")
    selected = intervals[np.linspace(0, len(intervals) - 1, 32, dtype=int)]
    masses = (curves[0, selected] + curves[0, selected + 1]) / 2
    points = jnp.stack((jnp.full(32, age), jnp.full(32, feh), jnp.asarray(masses)), axis=1)

    old_eep = jax.jit(jax.vmap(lambda p: legacy.eep_given_mass(*p)[0]))(points)
    new_eep = jax.jit(jax.vmap(lambda p: named.eep_given_mass(*p)))(points)
    old_values = jax.jit(jax.vmap(lambda p: legacy.values_given_mass(*p)))(points)
    new_values = jax.jit(jax.vmap(
        lambda p: named.values_given_mass(*p, keys=fields)
    ))(points)
    print("inverse diagnostic: 32 scalar queries from finite increasing mass intervals; "
          f"age={age}, feh={feh}")
    for label, old, new in (
        ("EEP", old_eep, new_eep),
        *((name, old, new_values[name]) for name, old in zip(fields, old_values)),
    ):
        assert np.all(np.isfinite(old)) and np.all(np.isfinite(new)), label
        difference = float(np.max(np.abs(np.asarray(old) - np.asarray(new))))
        print(f"inverse {label}: both finite; max absolute difference={difference:.6g}")
    worst = int(np.argmax(np.abs(np.asarray(old_eep) - np.asarray(new_eep))))
    interval = selected[worst]
    print(f"inverse largest EEP difference: mass={masses[worst]:.12g}, "
          f"source interval={float(eep_nodes[interval])}..{float(eep_nodes[interval + 1])}, "
          f"mass step={curves[0, interval + 1] - curves[0, interval]:.6g}, "
          f"EEP legacy={float(old_eep[worst]):.12g}, named={float(new_eep[worst]):.12g}")
    print("inverse reference points checked numerically; the 32-point sweep is "
          "diagnostic in both precisions (near-flat mass curves amplify roundoff)")

    scalar = jnp.asarray([age, feh, 1.31])
    for label, function in (
        ("legacy", lambda p: legacy.eep_given_mass(*p)[0]),
        ("named", lambda p: named.eep_given_mass(*p)),
    ):
        print(f"inverse {label} gradient (age, feh, mass): "
              f"{np.asarray(jax.jit(jax.grad(function))(scalar))}")
    # Keep failure cases visible; no invalid-value or mass-range policy changes.
    mass = 2.0
    old = legacy.eep_given_mass(age, feh, mass)[0]
    new = named.eep_given_mass(age, feh, mass)
    old_fields = np.asarray(legacy.values_given_mass(age, feh, mass))
    new_fields = named.values_given_mass(age, feh, mass, keys=fields)
    print(f"inverse unusual mass={mass}: EEP legacy={float(old)}, named={float(new)}; "
          f"fields legacy={old_fields}, named={np.asarray(list(new_fields.values()))}")

    if not rounds:
        return
    operations = (
        ("eep_given_mass", lambda p: legacy.eep_given_mass(*p)[0],
         lambda p: named.eep_given_mass(*p)),
        ("values_given_mass", lambda p: legacy.values_given_mass(*p),
         lambda p: named.values_given_mass(*p, keys=fields)),
    )
    arguments = tuple(scalar)  # Split coordinates before timing Python calls.
    for operation, old, new in operations:
        for style, functions in (("direct", (old, new)),
                                 ("outer_jit", (jax.jit(old), jax.jit(new)))):
            for _ in range(3):
                for function in functions:
                    block_until_ready(function(arguments))
            timings = ([], [])
            for round_index in range(rounds):
                for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                    start = perf_counter()
                    block_until_ready(functions[index](arguments))
                    timings[index].append(perf_counter() - start)
            old_seconds, new_seconds = (float(np.median(t)) for t in timings)
            print(f"inverse {operation} {style}: legacy={old_seconds * 1e3:.3f} ms, "
                  f"named={new_seconds * 1e3:.3f} ms, "
                  f"named/legacy={new_seconds / old_seconds:.3f}x")


def compare_mistfit(path, rounds):
    """Compare both public MistFit backends with the actual default model."""
    from numpyro.infer import MCMC, NUTS, init_to_value
    from numpyro.infer.util import initialize_model, unconstrain_fn

    legacy_fit = MistFit(path=path)
    named_fit = MistFit(path=path, grid_backend="named")
    fits = (legacy_fit, named_fit)
    labels = ("legacy", "named")

    # Three nearby interior points, away from grid nodes and invalid cells.
    # age is in Gyr, matching model()'s default linear_age=True parameterization.
    points = [
        dict(age=10**a / 1e9, feh_init=f, eep=e, distance=d)
        for a, f, e, d in (
            (9.275, -0.125, 350.5, 0.100),
            (9.280, -0.120, 350.6, 0.101),
            (9.270, -0.130, 350.4, 0.099),
        )
    ]
    reference = points[0]
    legacy_fit.mg.set_keys(("teff", "logg", "feh_photosphere", "kmag"))
    teff, logg, feh, kmag = map(float, legacy_fit.mg.values(
        np.log10(reference["age"] * 1e9), reference["feh_init"], reference["eep"]
    ))
    obskeys = ("teff", "logg", "feh", "kmag", "parallax")
    obsvals = (teff, logg, feh, kmag + 5 * np.log10(reference["distance"]) + 10,
               1 / reference["distance"])
    obserrs = (100.0, 0.1, 0.1, 0.03, 0.1)
    # Fresh legacy instance: set_keys after the first JIT call can retain old keys.
    legacy_fit.mg = MistGridIso(path=path)
    for fit in fits:
        fit.set_data(obskeys, obsvals, obserrs)

    infos = [
        initialize_model(
            jax.random.PRNGKey(0), fit.model,
            init_strategy=init_to_value(values=reference),
        )
        for fit in fits
    ]
    functions = [jax.jit(jax.value_and_grad(info.potential_fn)) for info in infos]
    arguments = [unconstrain_fn(legacy_fit.model, (), {}, point) for point in points]
    print("MistFit: synthetic observations; default model; 3 interior points")
    print(f"observations: {dict(zip(obskeys, obsvals))}")
    print("potential = negative log joint including unconstraining Jacobians")
    for label, function in zip(labels, functions):
        start = perf_counter()
        block_until_ready(function(arguments[0]))
        print(f"MistFit {label} first JIT call: {perf_counter() - start:.3f} s "
              "(compile + execute; excludes model initialization)")

    for index, argument in enumerate(arguments):
        old, new = (function(argument) for function in functions)
        for old_leaf, new_leaf in zip(
            jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new)
        ):
            assert np.all(np.isfinite(old_leaf)) and np.all(np.isfinite(new_leaf))
            if jax.config.read("jax_enable_x64"):
                np.testing.assert_allclose(old_leaf, new_leaf, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(old[0], new[0], rtol=1e-5, atol=1e-4)
        gradient_differences = {
            name: float(np.max(np.abs(np.asarray(old[1][name]) - new[1][name])))
            for name in old[1]
        }
        print(f"MistFit point {index + 1}: potential {float(old[0]):.7g} / "
              f"{float(new[0]):.7g}; abs diff={abs(float(old[0] - new[0])):.6g}")
        print(f"  gradient abs differences (unconstrained): {gradient_differences}")
    if jax.config.read("jax_enable_x64"):
        print("MistFit x64 potential and gradient: equal within rtol=1e-7, atol=1e-8")
    else:
        print("MistFit float32 gradients: diagnostic differences above; "
              "repeat with JAX_ENABLE_X64=1 for the tight equivalence check")

    if rounds:
        for _ in range(3):
            for function in functions:
                block_until_ready(function(arguments[0]))
        timings = ([], [])
        for round_index in range(rounds):
            for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                start = perf_counter()
                block_until_ready(functions[index](arguments[0]))
                timings[index].append(perf_counter() - start)
        old_seconds, new_seconds = (float(np.median(times)) for times in timings)
        print(f"MistFit JIT potential + gradient: legacy={old_seconds * 1e3:.3f} ms, "
              f"named={new_seconds * 1e3:.3f} ms, "
              f"named/legacy={new_seconds / old_seconds:.3f}x")

    for label, fit in zip(labels, fits):
        kernel = NUTS(
            fit.model, init_strategy=init_to_value(values=reference),
            target_accept_prob=0.9, max_tree_depth=5,
        )
        sampler = MCMC(kernel, num_warmup=20, num_samples=20, progress_bar=False)
        sampler.run(jax.random.PRNGKey(1), extra_fields=("potential_energy",))
        samples = sampler.get_samples()
        assert all(np.all(np.isfinite(samples[name])) for name in reference)
        extras = sampler.get_extra_fields()
        assert np.all(np.isfinite(extras["potential_energy"]))
        print(f"MistFit {label} NUTS smoke: 20 warmup + 20 samples, "
              f"finite latents and potential; divergences={int(extras['diverging'].sum())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path,
        help="existing MIST NPZ for an additional read-only comparison",
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--typical-fields", action="store_true",
        help="also time the 12 fields selected by MistFit.set_data()",
    )
    parser.add_argument(
        "--mistfit", action="store_true",
        help="compare the full default MistFit potential and gradient, then smoke-test NUTS",
    )
    parser.add_argument(
        "--inverse", action="store_true",
        help="compare scalar mass-to-EEP lookups, named outputs, gradients, and runtime",
    )
    args = parser.parse_args()
    if args.rounds < 0 or args.batch_size < 1:
        parser.error("--rounds must be nonnegative and --batch-size positive")
    if args.typical_fields and args.grid is None:
        parser.error("--typical-fields requires --grid")
    if args.mistfit and args.grid is None:
        parser.error("--mistfit requires --grid")
    if args.inverse and args.grid is None:
        parser.error("--inverse requires --grid")
    print(f"JAX: {jax.__version__}; backend: {jax.default_backend()}; "
          f"device: {jax.devices()[0]}; "
          f"x64: {jax.config.read('jax_enable_x64')}")
    print(f"warm-up: 3 calls each; timed rounds: {args.rounds}; "
          f"batch size: {args.batch_size}")
    compare_synthetic(args.rounds, args.batch_size)
    if args.grid is not None:
        if not args.grid.is_file():
            parser.error(f"grid file does not exist: {args.grid}")
        compare_real_grid(
            args.grid, args.rounds, args.batch_size, args.typical_fields
        )
        if args.mistfit:
            compare_mistfit(args.grid, args.rounds)
        if args.inverse:
            compare_inverse(args.grid, args.rounds)


if __name__ == "__main__":
    main()
