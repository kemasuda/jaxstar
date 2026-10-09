"""Build a small MIST track grid and audit held-out tracks and JAX derivatives.

Run with PYTHONPATH=src; see benchmarks/README.md for inputs and interpretation.
This offline experiment uses the existing RectilinearGrid, enables JAX x64,
and does not change MistFit or perform inference. Scientific residuals are
reported, not subjected to an arbitrary accuracy acceptance threshold.
"""

import argparse
import json
from pathlib import Path
import platform
import tarfile
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

from jaxstar.grid import Field, RectilinearGrid
from validate_mist_bolometric_corrections import BANDS, file_record, read_bc_archive
from validate_mist_eep_tracks import bc_coverage, read_track, track_bc


DIMS = ("initial_mass", "initial_feh", "eep")
SCALARS = ("logage", "age_gyr", "logTeff", "logg", "logL", "logR",
           "star_mass", "bc_metallicity", "cmd_surface_feh")
OUTPUTS = SCALARS + BANDS
EEP = np.arange(202, 551, dtype=np.float64)
COARSE_MASSES = np.array([.8, .9, 1., 1.1, 1.2])
REFINED_MASSES = np.array([.8, .9, .94, .98, 1., 1.02, 1.06, 1.1, 1.14, 1.18, 1.2])


def read_archives(paths, bc_axes, bc_fields, *, mass_range=(.8, 1.2)):
    """Read selected members via a temporary file; never extract archive paths."""
    if len(mass_range) != 2 or not np.all(np.isfinite(mass_range)) or not 0 < mass_range[0] < mass_range[1]:
        raise ValueError("mass_range must contain two positive increasing finite bounds")
    tracks, records = {}, []
    for path in paths:
        count = 0
        with TemporaryDirectory(prefix="jaxstar-track-") as tmp, tarfile.open(path, "r|xz") as archive:
            local = Path(tmp) / "selected.track.eep"
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or not name.endswith("M.track.eep"):
                    continue
                try:
                    named_mass = int(name.split("M.")[0]) / 100
                except ValueError:
                    continue
                if not mass_range[0] <= named_mass <= mass_range[1]:
                    continue
                with archive.extractfile(member) as stream:
                    local.write_bytes(stream.read())
                data, eep, meta = read_track(local)
                mass, composition = meta["initial_mass"], meta["composition"]
                feh = composition["[Fe/H]"]
                if (composition["[a/Fe]"], composition["v/vcrit"]) != (0., .4):
                    raise ValueError("This experiment requires [a/Fe]=0, v/vcrit=0.4")
                if mass != named_mass or meta["primary_eeps"][:4] != [1, 202, 353, 454]:
                    raise ValueError("Unexpected mass or primary EEP alignment")
                focus = (eep >= EEP[0]) & (eep <= EEP[-1])
                np.testing.assert_array_equal(eep[focus], EEP)
                d = {k: v[focus] for k, v in data.items()}
                if not np.all(np.diff(d["star_age"]) > 0):
                    raise ValueError("Native ages must strictly increase in the selected EEP range")
                bc_z = np.minimum(d["log_surf_z"] - np.log10(.0134), feh + .1)
                queries = np.column_stack((bc_z, d["log_Teff"], d["log_g"]))
                if not np.all(bc_coverage(bc_axes, queries)[1]):
                    raise ValueError(f"Incomplete BC coverage: {member.name}")
                magnitudes = 4.74 - 2.5 * d["log_L"][:, None] - track_bc(bc_axes, bc_fields, queries)
                scalar_values = (np.log10(d["star_age"]), d["star_age"] / 1e9,
                                 d["log_Teff"], d["log_g"], d["log_L"], d["log_R"],
                                 d["star_mass"], bc_z, bc_z - np.log10(d["surface_h1"] / .7381))
                values = np.column_stack((*scalar_values, magnitudes))
                if not np.all(np.isfinite(values)) or (mass, feh) in tracks:
                    raise ValueError("Nonfinite fields or duplicate mass/metallicity track")
                tracks[mass, feh] = values
                records.append({k: v for k, v in meta.items() if k != "path"} |
                               {"archive": str(path.resolve()), "member": member.name})
                count += 1
        print(f"Read {count} tracks from {path.name}", flush=True)
    masses = np.array(sorted({m for m, z in tracks}))
    metallicities = np.array(sorted({z for m, z in tracks}))
    np.testing.assert_array_equal(metallicities, [-.25, 0, .25])
    if any((m, z) not in tracks for m in masses for z in metallicities):
        raise ValueError("Missing mass/metallicity combinations; no padding permitted")
    values = np.array([[tracks[m, z] for z in metallicities] for m in masses])
    return (masses, metallicities, EEP.copy()), values, records


def build_grid(axes, values):
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before constructing a track grid")
    fields = {name: Field(values[..., i], DIMS) for i, name in enumerate(SCALARS)}
    fields["absolute_magnitudes"] = Field(values[..., len(SCALARS):], DIMS, payload_dims=("band",))
    return RectilinearGrid(axes=dict(zip(DIMS, axes)), fields=fields, fill_value=np.nan)


def load_grid(path):
    """Load this experiment's NPZ as a reusable named JAX grid (x64 required)."""
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["bands"], BANDS)
        axes = tuple(data[name] for name in DIMS)
        values = np.concatenate([data[name][..., None] for name in SCALARS] +
                                [data["absolute_magnitudes"]], axis=-1)
    return build_grid(axes, values)


@jax.jit
def evaluate(grid, points):
    result = grid.interpolate({name: points[..., i] for i, name in enumerate(DIMS)})
    return jnp.concatenate([result[name][..., None] for name in SCALARS] +
                           [result["absolute_magnitudes"]], axis=-1)


def age_years(grid, point):
    return 10 ** grid.interpolate(dict(zip(DIMS, point)), keys="logage")["logage"]


age_jacobian = jax.jit(jax.vmap(jax.grad(age_years, argnums=1), in_axes=(None, 0)))


def subset(axes, values, masses=None, fehs=None, eeps=None):
    selections = (masses, fehs, eeps)
    indices = []
    for axis, selected in zip(axes, selections):
        if selected is None:
            selected = axis
        if not np.all(np.isin(selected, axis)):
            raise ValueError("Requested training nodes are absent from the packaged tracks")
        indices.append(np.flatnonzero(np.isin(axis, selected)))
    return tuple(a[i] for a, i in zip(axes, indices)), values[np.ix_(*indices)]


def points_on(axes):
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def error_stats(error, points):
    absolute = np.abs(error)
    worst = np.unravel_index(np.argmax(absolute), absolute.shape)
    return {"median_abs": float(np.median(absolute)), "p95_abs": float(np.quantile(absolute, .95)),
            "max_abs": float(absolute[worst]), "worst_coordinates": points[worst[0]].tolist()}


def residual_metrics(predicted, reference, points):
    errors = {
        "age_relative": 10 ** (predicted[:, 0] - reference[:, 0]) - 1,
        "linear_age_relative": predicted[:, 1] / reference[:, 1] - 1,
        "teff_K": 10 ** predicted[:, 2] - 10 ** reference[:, 2],
        "logg_dex": predicted[:, 3] - reference[:, 3],
        "logL_dex": predicted[:, 4] - reference[:, 4],
        "surface_feh_dex": predicted[:, 8] - reference[:, 8],
        "all_bands_mag": predicted[:, len(SCALARS):] - reference[:, len(SCALARS):],
    }
    errors.update({name + "_mag": predicted[:, i] - reference[:, i]
                   for i, name in enumerate(OUTPUTS) if name in BANDS})
    return {k: error_stats(v, points) for k, v in errors.items()}, errors


def heldout_case(name, training, testing, outdir, with_jacobian=True):
    axes, values = training
    test_axes, truth = testing
    grid = build_grid(axes, values)
    points = points_on(test_axes)
    # Every query must have at least one coordinate absent from the training grid.
    if np.any(np.logical_and.reduce([np.isin(points[:, i], a) for i, a in enumerate(axes)])):
        raise AssertionError("Held-out data leaked into training nodes")
    prediction = np.asarray(evaluate(grid, points))
    reference = truth.reshape(-1, len(OUTPUTS))
    if not np.all(np.isfinite(prediction)):
        raise AssertionError("Nonfinite in-domain predictions")
    metrics, errors = residual_metrics(prediction, reference, points)
    phases = {}
    for lo, hi in ((202, 353), (354, 454), (455, 550)):
        selected = (points[:, 2] >= lo) & (points[:, 2] <= hi)
        phases[f"{lo}-{hi}"] = residual_metrics(prediction[selected], reference[selected], points[selected])[0]
    report = {"training_axes": dict(zip(DIMS, [a.tolist() for a in axes])),
              "test_axes": dict(zip(DIMS, [a.tolist() for a in test_axes])),
              "n_test_nodes": len(points), "metrics": metrics, "by_eep_range": phases}
    saved = dict(points=points, prediction=prediction, reference=reference)
    if with_jacobian:
        midpoint_axes = (*test_axes[:2], (test_axes[2][:-1] + test_axes[2][1:]) / 2)
        midpoint_points = points_on(midpoint_axes)
        logage = truth[..., 0]
        truth_dt = (np.log(10) * 10 ** ((logage[..., :-1] + logage[..., 1:]) / 2)
                    * np.diff(logage, axis=-1) / np.diff(test_axes[2])).ravel()
        predicted_dt = np.asarray(age_jacobian(grid, midpoint_points))[:, 2]
        if not np.all(np.isfinite(predicted_dt) & (predicted_dt > 0)):
            raise AssertionError("Nonpositive interpolated dt/dEEP")
        dt_error = predicted_dt / truth_dt - 1
        report["dt_deep_relative"] = error_stats(dt_error, midpoint_points)
        worst = np.argmax(np.abs(dt_error))
        report["dt_deep_relative"].update(
            worst_reference_yr=float(truth_dt[worst]), worst_prediction_yr=float(predicted_dt[worst]),
            worst_prediction_over_reference=float(predicted_dt[worst] / truth_dt[worst]),
            fraction_abs_error_above_50_percent=float(np.mean(np.abs(dt_error) > .5)))
        saved.update(midpoint_points=midpoint_points, predicted_dt_deep_yr=predicted_dt,
                     reference_dt_deep_yr=truth_dt)
    np.savez_compressed(outdir / f"{name}.npz", **saved)
    print(f"{name}: age p95={100 * metrics['age_relative']['p95_abs']:.3f}%, "
          f"Teff p95={metrics['teff_K']['p95_abs']:.2f} K, "
          f"photometry p95/max={metrics['all_bands_mag']['p95_abs']:.4f}/"
          f"{metrics['all_bands_mag']['max_abs']:.4f} mag", flush=True)
    return report, (points, errors)


def numerical_checks(axes, values, outdir):
    """Separate interpolation implementation checks from scientific accuracy."""
    grid = build_grid(axes, values)
    nodes = points_on(axes)
    at_nodes = np.asarray(evaluate(grid, nodes))
    np.testing.assert_allclose(at_nodes, values.reshape(at_nodes.shape), rtol=2e-13, atol=2e-13)
    rng = np.random.default_rng(42)
    cells = np.column_stack([rng.integers(0, len(a) - 1, 128) for a in axes])
    lower = np.column_stack([a[cells[:, i]] for i, a in enumerate(axes)])
    widths = np.column_stack([np.diff(a)[cells[:, i]] for i, a in enumerate(axes)])
    points = lower + rng.uniform(.2, .8, size=lower.shape) * widths
    scipy_grid = RegularGridInterpolator(axes, values, bounds_error=True)
    predicted = np.asarray(evaluate(grid, points))
    np.testing.assert_allclose(predicted, scipy_grid(points), rtol=2e-13, atol=2e-13)
    forward = np.asarray(jax.jit(jax.vmap(jax.jacfwd(evaluate, argnums=1), in_axes=(None, 0)))(grid, points))
    reverse = np.asarray(jax.jit(jax.vmap(jax.jacrev(evaluate, argnums=1), in_axes=(None, 0)))(grid, points))
    np.testing.assert_allclose(forward, reverse, rtol=1e-10, atol=1e-10)
    finite_difference = np.empty_like(forward)
    for i in range(3):
        offset = np.zeros_like(points)
        offset[:, i] = widths[:, i] * 1e-3
        finite_difference[..., i] = ((scipy_grid(points + offset) - scipy_grid(points - offset)) /
                                    (2 * offset[:, i, None]))
    scaled_error = np.abs(forward - finite_difference) / (1 + np.abs(finite_difference))
    if np.max(scaled_error) > 5e-7:
        raise AssertionError(f"JAX/independent finite-difference mismatch: {scaled_error.max()}")
    jac = np.asarray(age_jacobian(grid, points))
    np.testing.assert_allclose(jac, np.log(10) * 10 ** predicted[:, 0, None] * forward[:, 0, :],
                               rtol=2e-11, atol=1e-4)
    if not np.all(jac[:, 2] > 0):
        raise AssertionError("Random-cell dt/dEEP must be positive")
    # Convex interpolation of positive per-track log-age secants preserves the sign.
    native_slopes = np.diff(values[..., 0], axis=2) / np.diff(axes[2])
    if not np.all(native_slopes > 0):
        raise AssertionError("Nonpositive node log-age secant")
    invalid = []
    for i, a in enumerate(axes):
        for value in (a[0] - .01, a[-1] + .01, np.nan, np.inf):
            point = np.array([x.mean() for x in axes])
            point[i] = value
            invalid.append(point)
    if not np.all(np.isnan(np.asarray(evaluate(grid, np.array(invalid))))):
        raise AssertionError("Outside-domain values must be masked, not extrapolated")
    # Expose jumps at representative knots; piecewise linear does not mean C1.
    jumps = {}
    for dimension, name in enumerate(DIMS):
        knot_points = np.tile(np.array([.973, -.071, 407.37]), (len(axes[dimension]) - 2, 1))
        knot_points[:, dimension] = axes[dimension][1:-1]
        epsilon = min(np.diff(axes[dimension])) * 1e-7
        offset = np.zeros_like(knot_points)
        offset[:, dimension] = epsilon
        left = np.asarray(jax.jit(jax.vmap(jax.jacfwd(evaluate, argnums=1), in_axes=(None, 0)))(grid, knot_points - offset))
        right = np.asarray(jax.jit(jax.vmap(jax.jacfwd(evaluate, argnums=1), in_axes=(None, 0)))(grid, knot_points + offset))
        jump = np.abs(right[..., dimension] - left[..., dimension])
        jumps[name] = {"n_knots": len(knot_points),
                       "max_logage_slope_jump": float(jump[:, 0].max()),
                       "max_G_slope_jump": float(jump[:, OUTPUTS.index('Gaia_G_EDR3')].max())}
    np.savez_compressed(outdir / "gradient_checks.npz", points=points, autodiff=forward,
                        finite_difference=finite_difference, age_jacobian_yr=jac)
    return {"n_reproduced_nodes": len(nodes), "n_random_interior_queries": len(points),
            "max_scaled_gradient_error": float(scaled_error.max()),
            "minimum_native_dlogage_deep": float(native_slopes.min()),
            "random_dt_deep_min_max_yr": [float(jac[:, 2].min()), float(jac[:, 2].max())],
            "invalid_queries_masked": len(invalid), "representative_knot_jumps": jumps}


def make_plot(outdir, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for name in ("mass_coarse", "mass_refined", "feh_holdout", "joint_refined"):
        points, errors = cases[name]
        eeps = np.unique(points[:, 2])
        for ax, field, scale, label in zip(
            axs, ("age_relative", "teff_K", "Gaia_G_EDR3_mag"), (100, 1, 1),
            ("Maximum age error [%]", "Maximum Teff error [K]", "Maximum Gaia G error [mag]"),
        ):
            maxima = [np.max(np.abs(errors[field][points[:, 2] == e])) * scale for e in eeps]
            ax.plot(eeps, maxima, label=name)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    for ax in axs:
        for e in (353, 454):
            ax.axvline(e, color="black", alpha=.2, lw=.7)
    axs[0].legend(ncol=2, fontsize=9)
    axs[-1].set_xlabel("EEP")
    fig.suptitle("MIST v1.2 held-out tracks: linear mass / [Fe/H] / EEP interpolation\n"
                 "0.8–1.2 solar masses; [Fe/H] = −0.25, 0, +0.25; age from interpolated log-age")
    fig.savefig(outdir / "heldout_errors.png", dpi=170)
    plt.close(fig)


def make_feature_plot(outdir):
    """Show where rapid evolution shifts between neighboring native tracks."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cases = ("mass_refined", "feh_holdout", "joint_refined")
    for i, name in enumerate(cases):
        with np.load(outdir / f"{name}.npz", allow_pickle=False) as data:
            p, pred, ref = data["points"], data["prediction"], data["reference"]
            selected = (p[:, 0] == 1.16) & (p[:, 1] == 0) & (p[:, 2] >= 400) & (p[:, 2] <= 455)
            axs[0, 0].plot(p[selected, 2], 10 ** pred[selected, 0] / 1e9, label=name)
            if i == 0:
                axs[0, 0].plot(p[selected, 2], 10 ** ref[selected, 0] / 1e9, color="black", label="native reference")
            p = data["midpoint_points"]
            selected = (p[:, 0] == 1.16) & (p[:, 1] == 0) & (p[:, 2] >= 400) & (p[:, 2] <= 455)
            axs[1, 0].semilogy(p[selected, 2], data["predicted_dt_deep_yr"][selected], label=name)
            if i == 0:
                axs[1, 0].semilogy(p[selected, 2], data["reference_dt_deep_yr"][selected], color="black", label="native reference")
    with np.load(outdir / "grid.npz", allow_pickle=False) as data:
        eep = data["eep"]
        mid = (eep[:-1] + eep[1:]) / 2
        selected = (mid >= 400) & (mid <= 455)
        for mass in (1.14, 1.16, 1.18):
            mi = np.flatnonzero(data["initial_mass"] == mass)[0]
            logage = data["logage"][mi, 1]
            dt = np.log(10) * 10 ** ((logage[:-1] + logage[1:]) / 2) * np.diff(logage)
            axs[0, 1].semilogy(mid[selected], dt[selected], label=f"native M={mass:g}, [Fe/H]=0")
    with np.load(outdir / "feh_holdout.npz", allow_pickle=False) as data:
        p = data["points"]
        selected = (p[:, 0] == 1.14) & (p[:, 2] >= 500) & (p[:, 2] <= 535)
        index = OUTPUTS.index("Gaia_G_EDR3")
        axs[1, 1].plot(p[selected, 2], data["reference"][selected, index], color="black", label="native [Fe/H]=0")
        axs[1, 1].plot(p[selected, 2], data["prediction"][selected, index], label="interpolated from ±0.25")
        axs[1, 1].invert_yaxis()
    axs[0, 0].set(title="Held-out M=1.16, [Fe/H]=0", ylabel="Age [Gyr]")
    axs[1, 0].set(ylabel="dt/dEEP [yr]", xlabel="EEP")
    axs[0, 1].set(title="Native rapid-evolution intervals shift with mass", ylabel="dt/dEEP [yr]", xlabel="EEP")
    axs[1, 1].set(title="Held-out M=1.14, [Fe/H]=0", ylabel="Absolute Gaia G [mag]", xlabel="EEP")
    for ax in axs.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Local interpolation failures in the coarsened validation grids")
    fig.savefig(outdir / "feature_diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/mist-track-grid"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    bc_axes, bc_fields = read_bc_archive(args.bc)
    axes, values, records = read_archives(args.archive, bc_axes, bc_fields)
    if not np.all(np.isin(COARSE_MASSES, REFINED_MASSES)):
        raise AssertionError("Refinement must retain all coarse training masses")
    heldout_masses = axes[0][~np.isin(axes[0], REFINED_MASSES)]
    if not len(heldout_masses):
        raise ValueError("No independent mass holdouts")
    np.savez_compressed(outdir / "grid.npz", **dict(zip(DIMS, axes)),
                        **{name: values[..., i] for i, name in enumerate(SCALARS)},
                        absolute_magnitudes=values[..., len(SCALARS):], bands=np.array(BANDS))
    restored = load_grid(outdir / "grid.npz")
    np.testing.assert_allclose(np.asarray(evaluate(restored, points_on(axes))),
                               values.reshape(-1, len(OUTPUTS)), rtol=2e-13, atol=2e-13)
    checks = numerical_checks(axes, values, outdir)
    setups = {
        "mass_coarse": (subset(axes, values, masses=COARSE_MASSES), subset(axes, values, masses=heldout_masses)),
        "mass_refined": (subset(axes, values, masses=REFINED_MASSES), subset(axes, values, masses=heldout_masses)),
        "feh_holdout": (subset(axes, values, fehs=[-.25, .25]), subset(axes, values, fehs=[0.])),
        "joint_refined": (subset(axes, values, masses=REFINED_MASSES, fehs=[-.25, .25]),
                          subset(axes, values, masses=heldout_masses, fehs=[0.])),
        "eep_stride2": (subset(axes, values, eeps=EEP[::2]), subset(axes, values, eeps=EEP[1::2])),
    }
    reports, plot_data = {}, {}
    for name, (training, testing) in setups.items():
        reports[name], plot_data[name] = heldout_case(name, training, testing, outdir,
                                                    with_jacobian=name != "eep_stride2")
    summary = {
        "scope": "Small track-grid characterization; no inference or production backend change",
        "axes": dict(zip(DIMS, [a.tolist() for a in axes])), "shape": list(values.shape),
        "outputs": OUTPUTS, "Av": 0, "Rv": 3.1,
        "age_convention": "Trilinear log10(age/yr); age=10**logage; dt/dEEP from the same JAX function. Linear age_gyr is an explicit comparison only.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "jax": jax.__version__, "x64": jax.config.x64_enabled},
        "inputs": [file_record(args.bc)] + [file_record(p) for p in args.archive],
        "implementations": [file_record(Path(__file__).with_name(n)) for n in
                            ("validate_mist_track_grid.py", "validate_mist_eep_tracks.py",
                             "validate_mist_bolometric_corrections.py")] +
                           [file_record(Path(__file__).resolve().parents[1] / "src/jaxstar/grid/core.py")],
        "tracks": records, "numerical_checks": checks, "heldout": reports,
        "limitations": ["No application age cut applied", "Metallicity holdout spans 0.5 dex, twice the native 0.25 dex spacing",
                        "Withheld mass tests use coarser spacing than the final full grid",
                        "EEP test uses stride 2, not an independent sub-EEP evolution calculation",
                        "Gradients are piecewise; knot jumps and stellar-physics accuracy are not cured by autodiff"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plot(outdir, plot_data)
    make_feature_plot(outdir)
    print(f"Saved {values.shape[:3]} float64 grid and validation to {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
