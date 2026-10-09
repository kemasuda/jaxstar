"""Extend the native mass domain using existing archives; keep FeH and EEP fixed.

This is a grid-coverage experiment, not an interpolation-order or isochrone
comparison. Original grid artifacts are kept intact. No downloads are made.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from _benchmark_utils import environment
from mist_track_inference import centered_age_derivative
from validate_mist_bolometric_corrections import BANDS, file_record, read_bc_archive
from validate_mist_track_grid import (DIMS, OUTPUTS, SCALARS, evaluate, load_grid,
                                      numerical_checks, points_on, read_archives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("benchmark-results/mist-bc-inputs"))
    parser.add_argument("--baseline-grid", type=Path, default=Path("benchmark-results/mist-track-grid/grid.npz"))
    parser.add_argument("--mass-max", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/mist-track-grid-extended"))
    args = parser.parse_args()
    if args.mass_max <= 1.2 or not np.isfinite(args.mass_max):
        parser.error("The extended mass maximum must exceed 1.2 solar masses")
    out = args.output_dir
    if (out / "grid.npz").resolve() == args.baseline_grid.resolve():
        parser.error("Keep the baseline grid intact; choose a separate output directory")
    jax.config.update("jax_enable_x64", True)
    out.mkdir(parents=True, exist_ok=True)
    bc_path = args.inputs / "UBVRIplus.txz"
    archives = [args.inputs / f"MIST_v1.2_feh_{z}_afe_p0.0_vvcrit0.4_EEPS.txz"
                for z in ("m0.25", "p0.00", "p0.25")]
    bc_axes, bc_fields = read_bc_archive(bc_path)
    axes, values, tracks = read_archives(archives, bc_axes, bc_fields, mass_range=(.8, args.mass_max))
    if axes[0][-1] != args.mass_max:
        raise ValueError("Requested maximum must be a native mass present in all input archives")
    centered = centered_age_derivative(values[..., SCALARS.index("age_gyr")], axes[2])

    # Adding masses must not change any native field on the old domain.
    differences = {}
    with np.load(args.baseline_grid, allow_pickle=False) as old:
        np.testing.assert_array_equal(axes[1], old[DIMS[1]])
        np.testing.assert_array_equal(axes[2], old[DIMS[2]])
        np.testing.assert_array_equal(np.array(BANDS), old["bands"])
        indices = np.searchsorted(axes[0], old[DIMS[0]])
        np.testing.assert_array_equal(axes[0][indices], old[DIMS[0]])
        for i, name in enumerate(SCALARS):
            np.testing.assert_array_equal(values[indices, ..., i], old[name])
            differences[name] = float(np.max(np.abs(values[indices, ..., i] - old[name])))
        np.testing.assert_array_equal(values[indices, ..., len(SCALARS):], old["absolute_magnitudes"])
        np.testing.assert_array_equal(centered[indices], centered_age_derivative(old["age_gyr"], old["eep"]))
        differences["absolute_magnitudes"] = 0.
        differences["centered_dtdeep_gyr"] = 0.
        old_masses = old[DIMS[0]].tolist()

    path = out / "grid.npz"
    np.savez_compressed(path, **dict(zip(DIMS, axes)),
                        **{name: values[..., i] for i, name in enumerate(SCALARS)},
                        absolute_magnitudes=values[..., len(SCALARS):], bands=np.array(BANDS))
    loaded = load_grid(path)
    np.testing.assert_allclose(np.asarray(evaluate(loaded, points_on(axes))),
                               values.reshape(-1, len(OUTPUTS)), rtol=2e-13, atol=2e-13)
    # This audits gradients of value interpolation, not the prior derivative
    # choice; inference separately uses the central-difference table above.
    checks = numerical_checks(axes, values, out)
    report = {"scope": "Native mass-range extension only; FeH/EEP unchanged; no downloads",
              "environment": environment(jax.devices()[0]),
              "axes": dict(zip(DIMS, [axis.tolist() for axis in axes])), "shape": list(values.shape),
              "native_tracks": len(tracks), "added_masses": [float(m) for m in axes[0] if m not in old_masses],
              "common_native_field_max_absolute_differences": differences,
              "dtdeep_min_max_gyr_per_eep": [float(centered.min()), float(centered.max())],
              "value_interpolator_implementation_checks": checks,
              "inputs": [file_record(args.baseline_grid), file_record(bc_path)] + [file_record(p) for p in archives],
              "implementations": [file_record(Path(__file__).with_name(name)) for name in
                  ("build_mist_track_extension.py", "validate_mist_track_grid.py", "validate_mist_eep_tracks.py",
                   "validate_mist_bolometric_corrections.py", "mist_track_inference.py")],
              "grid": file_record(path), "tracks": tracks,
              "limitations": ["FeH and evolutionary-phase ranges remain restricted",
                              "implementation checks do not measure physical off-grid accuracy",
                              "NUTS inference uses separately interpolated centered physical-age differences"]}
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(tracks)} tracks, grid shape {values.shape[:3]}, to {path}; "
          "all original native fields and centered derivatives unchanged.", flush=True)


if __name__ == "__main__":
    main()
