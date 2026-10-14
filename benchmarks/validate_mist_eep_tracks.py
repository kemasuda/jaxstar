"""Apply packaged MIST BCs to a few real EEP tracks and audit their age axes.

This offline experiment characterizes native track nodes; it does not implement
interpolation across mass/metallicity or change MistFit. Inputs are read-only.
The selected EEP range must have complete photometry. Outside that range, BC
coverage limitations are reported and retained as NaNs with an explicit mask.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from validate_mist_bolometric_corrections import (
    BANDS, SOURCE_ROOT, file_record, read_bc_archive, source_recipe_bc, steffen_middle,
)


def read_track(path):
    """Recover the EEP numbers from the file header, not a guessed row offset."""
    header = []
    with path.open() as stream:
        for line in stream:
            header.append(line.strip())
            words = line.lstrip("#").split()
            if words[:1] == ["star_age"]:
                columns = words
                break
        else:
            raise ValueError(f"No track column header in {path}")
    if not any("MIST version number" in s and s.split("=")[-1].strip() == "1.2"
               for s in header):
        raise ValueError("This experiment requires MIST v1.2 tracks")
    metadata = composition = primary_eeps = None
    for i, line in enumerate(header):
        words = line.lstrip("#").split()
        if words[:2] == ["Yinit", "Zinit"]:
            composition = dict(zip(words, map(float, header[i + 1].lstrip("#").split())))
        elif words[:2] == ["initial_mass", "N_pts"]:
            metadata = dict(zip(words, header[i + 1].lstrip("#").split()))
        elif words[:1] == ["EEPs:"]:
            primary_eeps = np.array(list(map(int, words[1:])))
    if metadata is None or composition is None or primary_eeps is None:
        raise ValueError("Missing track composition, dimensions or EEP metadata")
    try:
        n_points = int(metadata["N_pts"])
        n_columns = int(metadata["N_col"])
        mass = float(metadata["initial_mass"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Missing or malformed track metadata") from exc
    if len(primary_eeps) != int(metadata["N_EEP"]) or not np.all(np.diff(primary_eeps) > 0):
        raise ValueError("Primary EEP labels must strictly increase")
    values = np.loadtxt(path, ndmin=2)
    if values.shape != (n_points, n_columns) or len(columns) != n_columns:
        raise ValueError("Track dimensions disagree with the header")
    eep = np.arange(primary_eeps[0], primary_eeps[-1] + 1)
    if len(eep) != n_points:
        raise ValueError("Track rows do not match the declared EEP range")
    data = {name: values[:, i] for i, name in enumerate(columns)}
    required = ("star_age", "star_mass", "log_Teff", "log_g", "log_L", "log_R",
                "log_surf_z", "surface_h1")
    for name in required:
        if name not in data or not np.all(np.isfinite(data[name])):
            raise ValueError(f"Missing/nonfinite required track column: {name}")
    if np.any(data["star_age"] <= 0) or np.any(data["surface_h1"] <= 0):
        raise ValueError("Age and surface hydrogen must be positive")
    return data, eep, {
        **file_record(path), "initial_mass": mass, "composition": composition,
        "n_points": n_points, "primary_eeps": primary_eeps.tolist(), "header": header,
    }


def rounded_axis_report(values, eep):
    """Quantize nodes to float32 before differencing, without hiding zero cells."""
    reference = np.diff(values)
    quantized = np.diff(values.astype(np.float32).astype(np.float64))
    nonpositive = quantized <= 0
    nonzero_reference = reference != 0
    relative_error = np.abs((quantized[nonzero_reference] - reference[nonzero_reference])
                            / reference[nonzero_reference])
    return {
        "zero_intervals": int(np.count_nonzero(quantized == 0)),
        "negative_intervals": int(np.count_nonzero(quantized < 0)),
        "nonpositive_left_eeps": eep[:-1][nonpositive].tolist(),
        "p95_relative_step_error": float(np.quantile(relative_error, 0.95)),
        "max_relative_step_error": float(np.max(relative_error)),
    }


def age_report(age, eep):
    """Secants are derivatives of linear-age interpolation on each EEP cell."""
    secants = np.diff(age) / np.diff(eep)
    logage = np.log10(age)
    log_secants = np.diff(logage) / np.diff(eep)
    # If logage is interpolated linearly instead, dt/dEEP varies within a cell.
    # Evaluate that derivative at its midpoint; do not call it a node derivative.
    midpoint_age = 10 ** ((logage[:-1] + logage[1:]) / 2)
    logage_jacobian = np.log(10) * midpoint_age * log_secants
    report = {
        "n_nodes": len(age), "n_intervals": len(secants),
        "eep_range": [int(eep[0]), int(eep[-1])],
        "age_range_yr": [float(age.min()), float(age.max())],
        "age64_nonpositive_steps": int(np.count_nonzero(secants <= 0)),
        "logage64_nonpositive_steps": int(np.count_nonzero(log_secants <= 0)),
        "dt_deep_linear_age_min_max_yr": [float(secants.min()), float(secants.max())],
        "smallest_step_left_eep": int(eep[np.argmin(secants)]),
        "dt_deep_logage_midpoint_min_max_yr": [float(logage_jacobian.min()), float(logage_jacobian.max())],
        "float32_age": rounded_axis_report(age, eep),
        "float32_logage": rounded_axis_report(logage, eep),
    }
    return report, secants, logage_jacobian


def bc_coverage(axes, queries):
    """Separate true table bounds from this experiment's interior-stencil limit."""
    finite = np.all(np.isfinite(queries), axis=1)
    in_table = finite.copy()
    for axis, coordinate in zip(axes, queries.T):
        in_table &= (coordinate >= axis[0]) & (coordinate <= axis[-1])
    iz, it, ig = [np.searchsorted(a, q, side="right") - 1 for a, q in zip(axes, queries.T)]
    supported = (in_table & (iz >= 1) & (iz < len(axes[0]) - 2)
                 & (ig >= 0) & (ig < len(axes[2]) - 1))
    interior = supported & (it >= 1) & (it < len(axes[1]) - 2)
    return in_table, supported, interior


def track_bc(axes, fields, queries):
    """Use the validated interior recipe plus the source's linear T-edge cells.

    BC_tables.f90 uses linear interpolation in logT in the first and last
    temperature intervals. Gravity remains linear and metallicity uses the
    same four-point monotone recipe. No extrapolation/clipping outside the
    table is permitted; metallicity/gravity edge policies remain unsupported.
    """
    _, supported, interior = bc_coverage(axes, queries)
    if not np.all(supported):
        raise ValueError("Unsupported BC coordinates")
    result = np.empty((len(queries), fields.shape[-1]))
    result[interior] = source_recipe_bc(axes, fields, queries[interior])
    edge = queries[~interior]
    if len(edge):
        z, t, g = axes
        iz, it, ig = [np.searchsorted(a, q, side="right") - 1 for a, q in zip(axes, edge.T)]
        # The exact maximum temperature belongs to the last in-bounds cell.
        it = np.minimum(it, len(t) - 2)
        tw = ((edge[:, 1] - t[it]) / (t[it + 1] - t[it]))[:, None]
        gw = ((edge[:, 2] - g[ig]) / (g[ig + 1] - g[ig]))[:, None]
        planes = []
        for dz in (-1, 0, 1, 2):
            lo = (1 - tw) * fields[iz + dz, it, ig] + tw * fields[iz + dz, it + 1, ig]
            hi = (1 - tw) * fields[iz + dz, it, ig + 1] + tw * fields[iz + dz, it + 1, ig + 1]
            planes.append((1 - gw) * lo + gw * hi)
        x = z[iz[:, None] + np.arange(-1, 3)]
        result[~interior] = steffen_middle(x, np.stack(planes, axis=1), edge[:, 0])
    return result


def check_temperature_edges(axes, fields):
    """Check real table nodes and cell midpoints independently of track values."""
    z, t, g = axes
    zi, gi = len(z) // 2, len(g) // 2
    queries, expected = [], []
    for ti in (0, len(t) - 2):
        for fraction in (0, 0.5, 1):
            queries.append([z[zi], (1 - fraction) * t[ti] + fraction * t[ti + 1], g[gi]])
            expected.append((1 - fraction) * fields[zi, ti, gi] + fraction * fields[zi, ti + 1, gi])
    error = float(np.max(np.abs(track_bc(axes, fields, np.array(queries)) - expected)))
    if error > 1e-12:
        raise AssertionError(f"BC temperature edge/node check failed: {error}")
    return {"n_queries": len(queries), "max_abs_mag": error}


def convert_track(path, axes, fields, eep_range):
    data, eep, metadata = read_track(path)
    focus = (eep >= eep_range[0]) & (eep <= eep_range[1])
    if np.count_nonzero(focus) < 2:
        raise ValueError("The focus range needs at least two track nodes")
    initial_feh = metadata["composition"]["[Fe/H]"]
    raw_metallicity = data["log_surf_z"] - np.log10(0.0134)
    bc_metallicity = np.minimum(raw_metallicity, initial_feh + 0.1)
    queries = np.column_stack((bc_metallicity, data["log_Teff"], data["log_g"]))
    in_table, valid, interior = bc_coverage(axes, queries)
    if not np.all(valid[focus]):
        raise ValueError(f"Focus photometry is incomplete: EEPs {eep[focus & ~valid].tolist()}")
    magnitudes = np.full((len(eep), len(BANDS)), np.nan)
    magnitudes[valid] = (4.74 - 2.5 * data["log_L"][valid, None]
                         - track_bc(axes, fields, queries[valid]))
    if not np.all(np.isfinite(magnitudes[valid])):
        raise ValueError("Nonfinite photometry inside the supported BC domain")
    full_age, dt_deep, logage_dt_deep = age_report(data["star_age"], eep)
    focus_age, _, _ = age_report(data["star_age"][focus], eep[focus])
    if full_age["age64_nonpositive_steps"] or full_age["logage64_nonpositive_steps"]:
        raise ValueError("Non-monotonic native age axis; do not silently repair it")
    report = {
        **metadata, "age_full": full_age, "age_focus": focus_age,
        "focus_nodes_within_current_age_bounds": int(np.count_nonzero(
            focus & (data["star_age"] >= 1e8) & (data["star_age"] <= 10 ** 10.14))),
        "bc_coverage": {
            "valid_full_nodes": int(valid.sum()), "valid_focus_nodes": int(valid[focus].sum()),
            "outside_table_eeps": eep[~in_table].tolist(),
            "unsupported_edge_stencil_eeps": eep[in_table & ~valid].tolist(),
            "linear_temperature_edge_eeps": eep[valid & ~interior].tolist(),
            "composition_capped_nodes": int(np.count_nonzero(raw_metallicity > initial_feh + 0.1)),
        },
        "focus_magnitude_ranges": {band: [float(v.min()), float(v.max())]
                                   for band, v in zip(BANDS, magnitudes[focus].T)},
    }
    arrays = {
        "eep": eep, "initial_mass": np.array(metadata["initial_mass"]),
        "initial_feh": np.array(initial_feh), "age_yr": data["star_age"],
        "logage": np.log10(data["star_age"]), "star_mass": data["star_mass"],
        "logTeff": data["log_Teff"], "logg": data["log_g"], "logL": data["log_L"],
        "radius": 10 ** data["log_R"], "surface_h1": data["surface_h1"],
        "log_surf_z": data["log_surf_z"], "bc_metallicity": bc_metallicity,
        # Keep the CMD-compatible proxy distinct from the BC coordinate.
        "cmd_surface_feh": bc_metallicity - np.log10(data["surface_h1"] / 0.7381),
        "bands": np.array(BANDS), "absolute_magnitudes": magnitudes,
        "bc_valid": valid, "bc_in_table": in_table, "bc_interior": interior, "focus": focus,
        "eep_midpoints": (eep[:-1] + eep[1:]) / 2,
        "dt_deep_linear_age_yr": dt_deep,
        "dt_deep_logage_midpoint_yr": logage_dt_deep,
    }
    return report, arrays


def make_plots(outdir, tracks, eep_range, composition):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for i, track in enumerate(tracks):
        label = f"{track['initial_mass']:.1f} solar masses"
        color = colors[i % len(colors)]
        eep, focus = track["eep"], track["focus"]
        for column, mask in enumerate((np.ones(len(eep), bool), focus)):
            intervals = mask[:-1] & mask[1:]
            axes[0, column].plot(eep[mask], track["age_yr"][mask] / 1e9, color=color, label=label)
            axes[1, column].semilogy(track["eep_midpoints"][intervals],
                                    track["dt_deep_linear_age_yr"][intervals], color=color)
    for col, title in enumerate(("Full native tracks", f"Focus: EEP {eep_range[0]}–{eep_range[1]}")):
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Age [Gyr]")
        axes[1, col].set_ylabel("Age interval / EEP interval [yr]")
        axes[1, col].set_xlabel("EEP")
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
            for boundary in (202, 353, 454, 605, 631, 707, 808, 1409):
                if col == 0 or eep_range[0] <= boundary <= eep_range[1]:
                    ax.axvline(boundary, color="black", lw=0.5, alpha=0.16)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("MIST v1.2: native float64 age and EEP spacing (all track ages retained)")
    fig.savefig(outdir / "age_diagnostics.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    g_index, bp_index, rp_index = [BANDS.index(f"Gaia_{b}_EDR3") for b in ("G", "BP", "RP")]
    for i, track in enumerate(tracks):
        mask = track["focus"]
        mags = track["absolute_magnitudes"][mask]
        color = colors[i % len(colors)]
        label = f"{track['initial_mass']:.1f} solar masses"
        axes[0].plot(10 ** track["logTeff"][mask], track["logL"][mask], color=color, label=label)
        axes[1].plot(mags[:, bp_index] - mags[:, rp_index], mags[:, g_index], color=color, label=label)
    axes[0].invert_xaxis()
    axes[0].set(xlabel="Effective temperature [K]", ylabel="log10(L / solar luminosity)", title="Native stellar tracks")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Gaia EDR3 BP − RP [mag]", ylabel="Absolute Gaia EDR3 G [mag]", title="Packaged BC photometry, Av = 0")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(f"Initial [Fe/H] = {composition['[Fe/H]']:g}, v/vcrit = {composition['v/vcrit']:g}; "
                 f"EEP {eep_range[0]}–{eep_range[1]}")
    fig.savefig(outdir / "photometry.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=Path, required=True, help="Packaged UBVRIplus BC archive")
    parser.add_argument("--track", type=Path, action="append", required=True, help="Native .track.eep; repeat per mass")
    parser.add_argument("--eep-range", type=int, nargs=2, default=(202, 550))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/mist-track-validation"))
    args = parser.parse_args()
    axes, fields = read_bc_archive(args.bc)
    edge_check = check_temperature_edges(axes, fields)
    reports, arrays = [], []
    for path in args.track:
        report, data = convert_track(path, axes, fields, args.eep_range)
        reports.append(report)
        arrays.append(data)
    compositions = [r["composition"] for r in reports]
    if any(c != compositions[0] for c in compositions[1:]):
        raise ValueError("This comparison requires identical initial composition and rotation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, (path, data) in enumerate(zip(args.track, arrays)):
        filename = f"{i:02d}_{path.stem}.npz"
        np.savez_compressed(args.output_dir / filename, **data)
        reports[i]["output_npz"] = filename
    summary = {
        "scope": "Native track age/EEP and photometry audit; no mass/metallicity interpolation or inference",
        "focus_eep_range": args.eep_range, "Av": 0, "Rv": 3.1,
        "bc_input": file_record(args.bc), "script": file_record(Path(__file__)),
        "bc_implementation": file_record(Path(__file__).with_name("validate_mist_bolometric_corrections.py")),
        "bands": BANDS, "tracks": reports,
        "temperature_edge_check": edge_check,
        "derivative_convention": "Delta age / Delta EEP is the derivative of linear-age interpolation inside each cell; logage-linear derivatives are separately evaluated at cell midpoints",
        "sources": [SOURCE_ROOT + "iso_eep_color.f90", SOURCE_ROOT + "BC_tables.f90",
                    "https://mist.science/model_grids.html"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(args.output_dir, arrays, args.eep_range, compositions[0])
    print("mass  native age64 bad steps  float32 age/logage zero steps  focus BC nodes")
    for report in reports:
        full = report["age_full"]
        print(f"{report['initial_mass']:.1f}   {full['age64_nonpositive_steps']:>9}"
              f"                   {full['float32_age']['zero_intervals']:>4}/{full['float32_logage']['zero_intervals']:<4}"
              f"                    {report['bc_coverage']['valid_focus_nodes']}/{report['age_focus']['n_nodes']}")
    print(f"Saved results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
