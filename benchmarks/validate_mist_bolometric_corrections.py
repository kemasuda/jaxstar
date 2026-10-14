"""Compare packaged MIST v1.2 BC tables with published UBVRIplus CMDs.

This is an offline characterization experiment, not a production BC backend.
It uses CMD stellar parameters directly, so it measures no track interpolation
error. Inputs are read-only; no downloads, extraction or MIST-grid generation
are performed. See benchmarks/README.md for the command and interpretation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator


BANDS = (
    "Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I",
    "2MASS_J", "2MASS_H", "2MASS_Ks",
    "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev",
    "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3",
)
SOURCE_ROOT = (
    "https://github.com/aarondotter/iso/blob/"
    "ed6ef0588664de62df39564fb02e3e0e32d5b2eb/src/"
)


def file_record(path):
    """Record the exact local input rather than assume a download version."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def read_cmd(path, mass_range, logage_range, eep_range):
    """Read named columns and apply predeclared physical selection bounds."""
    with path.open() as stream:
        header = []
        for line in stream:
            header.append(line.strip())
            if line.startswith("# EEP "):
                names = line[1:].split()
                break
        else:
            raise ValueError(f"CMD column header not found: {path}")
    if not any("MIST version number" in s and s.split("=")[-1].strip() == "1.2"
               for s in header):
        raise ValueError("This experiment requires a MIST v1.2 CMD")
    if not any("extinction: Av" in s and float(s.split("=")[-1]) == 0
               for s in header):
        raise ValueError("This experiment requires Av=0 CMD data")
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError("CMD column count disagrees with the header")
    data = {name: values[:, i] for i, name in enumerate(names)}
    mask = np.ones(len(values), dtype=bool)
    for key, bounds in (("initial_mass", mass_range),
                        ("log10_isochrone_age_yr", logage_range),
                        ("EEP", eep_range)):
        mask &= (data[key] >= bounds[0]) & (data[key] <= bounds[1])
    if not np.any(mask):
        raise ValueError(f"No CMD rows pass the selection: {path}")
    return {key: value[mask] for key, value in data.items()}, len(values), header


def read_bc_archive(path):
    """Read the Av=0, Rv=3.1 slices without extracting an archive to disk."""
    planes = []
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".UBVRIplus"):
                continue
            with archive.extractfile(member) as stream:
                for raw in stream:
                    line = raw.decode("ascii")
                    if line.startswith("#") and line[1:].split()[:5] == [
                        "Teff", "logg", "[Fe/H]", "Av", "Rv"
                    ]:
                        names = line[1:].split()
                        break
                else:
                    raise ValueError(f"BC column header not found: {member.name}")
                values = np.loadtxt(stream, ndmin=2)
            if values.shape[1] != len(names):
                raise ValueError(f"BC column count mismatch: {member.name}")
            values = values[(values[:, 3] == 0) & (values[:, 4] == 3.1)]
            temperatures = np.unique(values[:, 0])
            gravities = np.unique(values[:, 1])
            metallicities = np.unique(values[:, 2])
            if len(metallicities) != 1:
                raise ValueError("Expected one metallicity per BC file")
            order = np.lexsort((values[:, 1], values[:, 0]))
            ordered = values[order]
            expected = np.stack(np.meshgrid(temperatures, gravities, indexing="ij"), -1)
            if not np.array_equal(ordered[:, :2], expected.reshape(-1, 2)):
                raise ValueError("BC temperature/gravity grid has missing or duplicate nodes")
            fields = ordered[:, [names.index(band) for band in BANDS]]
            if not np.all(np.isfinite(fields)):
                raise ValueError("Nonfinite BC table values")
            planes.append((metallicities[0], temperatures, gravities,
                           fields.reshape(len(temperatures), len(gravities), -1)))
    if len(planes) < 4:
        raise ValueError("At least four BC metallicities are needed")
    planes.sort(key=lambda item: item[0])
    z = np.array([p[0] for p in planes])
    t, g = planes[0][1:3]
    if not np.all(np.diff(z) > 0):
        raise ValueError("Duplicate BC metallicities")
    if any(not np.array_equal(t, p[1]) or not np.array_equal(g, p[2]) for p in planes):
        raise ValueError("BC atmosphere axes differ between metallicities")
    return (z, np.log10(t), g), np.stack([p[3] for p in planes])


def steffen_middle(x, y, query):
    """Four-point monotone Hermite interpolation on the middle interval.

    x: (N, 4); y: (N, 4, bands); query: (N,). Slopes use Steffen's
    limiter, as in MESA interp_4pt_pm_sg / interp_4_to_1_sg. This is not
    an unconstrained four-node polynomial or scipy's PCHIP prescription.
    """
    h = np.diff(x, axis=1)[..., None]
    slopes = np.diff(y, axis=1) / h
    parabola = (slopes[:, :-1] * h[:, 1:] + slopes[:, 1:] * h[:, :-1]) / (
        h[:, :-1] + h[:, 1:]
    )
    derivatives = (np.sign(slopes[:, :-1]) + np.sign(slopes[:, 1:])) * np.minimum(
        np.minimum(np.abs(slopes[:, :-1]), np.abs(slopes[:, 1:])),
        0.5 * np.abs(parabola),
    )
    dx = (query - x[:, 1])[:, None]
    quadratic = (3 * slopes[:, 1] - 2 * derivatives[:, 0] - derivatives[:, 1]) / h[:, 1]
    cubic = (derivatives[:, 0] + derivatives[:, 1] - 2 * slopes[:, 1]) / h[:, 1] ** 2
    return y[:, 1] + dx * (derivatives[:, 0] + dx * (quadratic + dx * cubic))


def source_recipe_bc(axes, fields, queries):
    """Interior Av=0 recipe from the public iso/MESA source, using float64.

    Interpolate logT first with Steffen, logg linearly, then metallicity
    with Steffen. Reject boundary stencils rather than silently extrapolate
    or claim that this small experiment reproduces all official edge rules.
    """
    z, t, g = axes
    indices = [np.searchsorted(a, q, side="right") - 1 for a, q in zip(axes, queries.T)]
    iz, it, ig = indices
    valid = (np.all(np.isfinite(queries), axis=1)
             & (iz >= 1) & (iz < len(z) - 2)
             & (it >= 1) & (it < len(t) - 2)
             & (ig >= 0) & (ig < len(g) - 1))
    if not np.all(valid):
        raise ValueError(f"{np.count_nonzero(~valid)} queries outside supported interior stencils")
    offsets = np.arange(-1, 3)
    tidx = it[:, None] + offsets

    def at_metallicity_and_gravity(zi, gi):
        values = fields[zi[:, None], tidx, gi[:, None]]
        return steffen_middle(t[tidx], values, queries[:, 1])

    weight = ((queries[:, 2] - g[ig]) / (g[ig + 1] - g[ig]))[:, None]
    planes = []
    for dz in offsets:
        lo = at_metallicity_and_gravity(iz + dz, ig)
        hi = at_metallicity_and_gravity(iz + dz, ig + 1)
        planes.append((1 - weight) * lo + weight * hi)
    return steffen_middle(z[iz[:, None] + offsets], np.stack(planes, axis=1), queries[:, 0])


def statistics(residual):
    if not np.all(np.isfinite(residual)):
        raise ValueError("Nonfinite residuals; refusing to silently drop rows")
    return {band: {
        "median_mag": float(np.median(r)),
        "rms_mag": float(np.sqrt(np.mean(r ** 2))),
        "p95_abs_mag": float(np.quantile(np.abs(r), 0.95)),
        "max_abs_mag": float(np.max(np.abs(r))),
    } for band, r in zip(BANDS, residual.T)}


def match_surface_hydrogen(path, cmd):
    """Match theoretical isochrone rows exactly in age/EEP, verifying physics.

    CMD [Fe/H] is NOT the metallicity passed to the BC interpolator. The
    public color code outputs [Fe/H] = log(Z/Zsun) - log(X/Xsun), but queries
    the BC table at log(Z/Zsun). Xsun=0.7381 in that code. A matching theory
    file supplies X; it must describe the same stellar models as the CMD.
    """
    with path.open() as stream:
        for line in stream:
            if line.startswith("# EEP "):
                names = line[1:].split()
                break
        else:
            raise ValueError(f"Theory column header not found: {path}")
    values = np.loadtxt(path, ndmin=2)
    theory = {key: values[:, i] for i, key in enumerate(names)}
    row_keys = [(round(age, 8), int(eep)) for age, eep in
                zip(theory["log10_isochrone_age_yr"], theory["EEP"])]
    lookup = {key: i for i, key in enumerate(row_keys)}
    if len(lookup) != len(row_keys):
        raise ValueError("Duplicate age/EEP keys in theoretical isochrones")
    indices = [lookup[(round(age, 8), int(eep))] for age, eep in
               zip(cmd["log10_isochrone_age_yr"], cmd["EEP"])]
    # These checks prevent wrong rotation/version/metallicity files being used.
    discrepancies = {}
    for key in ("log10_isochrone_age_yr", "initial_mass", "star_mass", "log_Teff", "log_g", "log_L"):
        delta = np.max(np.abs(theory[key][indices] - cmd[key]))
        discrepancies[key] = float(delta)
        if not np.allclose(theory[key][indices], cmd[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"CMD and theory disagree in {key}: {delta}")
    hydrogen = theory["surface_h1"][indices]
    if not np.all(np.isfinite(hydrogen) & (hydrogen > 0) & (hydrogen <= 1)):
        raise ValueError("Invalid surface hydrogen mass fractions")
    return hydrogen, discrepancies


def save_plot(path, eep, linear, recipe):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3", "2MASS_J", "2MASS_H", "2MASS_Ks")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, band in zip(axes.flat, bands):
        j = BANDS.index(band)
        ax.scatter(eep, 1000 * linear[:, j], s=2, alpha=0.22, color="#9aa4b2", label="Linear")
        ax.scatter(eep, 1000 * recipe[:, j], s=2, alpha=0.25, color="#165c9e", label="Public source recipe")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(band)
        ax.set_ylabel("Calculated − published [mmag]")
        ax.grid(alpha=0.15)
    for ax in axes[-1]:
        ax.set_xlabel("EEP")
    axes[0, 0].legend(markerscale=3, fontsize=9)
    fig.suptitle("Packaged MIST v1.2 BCs against CMDs at identical stellar parameters")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=Path, required=True, help="Packaged UBVRIplus BC .txz")
    parser.add_argument("--cmd", type=Path, action="append", required=True, help="CMD path; repeat for multiple files")
    parser.add_argument("--theory", type=Path, action="append", required=True,
                        help="Matching basic/full .iso; same order as --cmd")
    parser.add_argument("--mass-range", type=float, nargs=2, default=(0.7, 1.5))
    parser.add_argument("--logage-range", type=float, nargs=2, default=(8.0, 10.14))
    parser.add_argument("--eep-range", type=float, nargs=2, default=(202, 550))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/mist-bc-validation"))
    args = parser.parse_args()
    if len(args.theory) != len(args.cmd):
        parser.error("Supply one --theory per --cmd, in the same order")
    axes, fields = read_bc_archive(args.bc)
    linear_interpolator = RegularGridInterpolator(axes, fields, bounds_error=True)

    # Check real BC nodes independently of the CMD comparison.
    mesh = np.meshgrid(axes[0][1:-2:3], axes[1][1:-2:5], axes[2][1:-1:4], indexing="ij")
    node_queries = np.column_stack([v.ravel() for v in mesh])
    node_error = np.max(np.abs(source_recipe_bc(axes, fields, node_queries) - linear_interpolator(node_queries)))
    if node_error > 1e-12:
        raise AssertionError(f"BC grid-node recovery failed: {node_error}")

    records, datasets = [], []
    for path, theory_path in zip(args.cmd, args.theory):
        data, total, header = read_cmd(path, args.mass_range, args.logage_range, args.eep_range)
        data["surface_h1"], differences = match_surface_hydrogen(theory_path, data)
        datasets.append(data)
        records.append({**file_record(path), "total_rows": total, "selected_rows": len(data["EEP"]), "header": header,
                        "theory": file_record(theory_path), "theory_match_max_abs": differences})
    data = {key: np.concatenate([d[key] for d in datasets]) for key in datasets[0]}
    bc_metallicity = data["[Fe/H]"] + np.log10(data["surface_h1"] / 0.7381)
    queries = np.column_stack([bc_metallicity, data["log_Teff"], data["log_g"]])
    reference = np.column_stack([data[band] for band in BANDS])
    bolometric = 4.74 - 2.5 * data["log_L"][:, None]
    predictions = {
        "linear": bolometric - linear_interpolator(queries),
        "source_recipe": bolometric - source_recipe_bc(axes, fields, queries),
    }
    initial_queries = queries.copy()
    initial_queries[:, 0] = data["[Fe/H]_init"]
    predictions["source_recipe_initial_feh_control"] = bolometric - source_recipe_bc(axes, fields, initial_queries)
    surface_queries = queries.copy()
    surface_queries[:, 0] = data["[Fe/H]"]
    predictions["source_recipe_cmd_feh_control"] = bolometric - source_recipe_bc(axes, fields, surface_queries)
    residuals = {key: value - reference for key, value in predictions.items()}
    summary = {
        "scope": "CMD atmosphere-to-photometry only; no track interpolation or inference",
        "formula": "M_band = 4.74 - 2.5*log_L - BC_band; Av=0, Rv=3.1",
        "bc_metallicity": "CMD [Fe/H] + log10(surface_h1/0.7381); equals the color code's log(Zsurf/0.0134), including its composition cap",
        "bc_input": file_record(args.bc), "cmd_inputs": records,
        "selection": {"initial_mass": args.mass_range, "log10_age_yr": args.logage_range, "EEP": args.eep_range},
        "n_rows": len(queries), "bands": BANDS,
        "query_ranges": {name: [float(v.min()), float(v.max())] for name, v in zip(("BC_metallicity", "logTeff", "logg"), queries.T)},
        "bc_grid_shape": fields.shape,
        "bc_grid_axes": {name: a.tolist() for name, a in zip(("[Fe/H]", "logTeff", "logg"), axes)},
        "node_check": {"n": len(node_queries), "max_abs_mag": float(node_error)},
        "results": {key: statistics(value) for key, value in residuals.items()},
        "sources": [SOURCE_ROOT + "BC_tables.f90", SOURCE_ROOT + "iso_eep_color.f90",
                    "https://github.com/MESAHub/mesa/blob/main/interp_1d/public/interp_1d_lib_sg.f90",
                    "https://mist.science/model_grids.html"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "comparison.npz", queries=queries, eep=data["EEP"],
                        initial_mass=data["initial_mass"], initial_feh=data["[Fe/H]_init"],
                        logage=data["log10_isochrone_age_yr"], logL=data["log_L"],
                        cmd_feh=data["[Fe/H]"], surface_h1=data["surface_h1"],
                        bands=BANDS, reference=reference, **predictions)
    save_plot(args.output_dir / "residuals.png", data["EEP"], residuals["linear"], residuals["source_recipe"])
    print(f"{len(queries)} CMD rows; BC grid {fields.shape}; {len(node_queries)} node checks")
    print("band                       linear p95/max    source recipe p95/max [mmag]")
    for band in BANDS:
        a = summary["results"]["linear"][band]
        b = summary["results"]["source_recipe"][band]
        print(f"{band:26s} {1000*a['p95_abs_mag']:8.4f}/{1000*a['max_abs_mag']:8.4f}"
              f"  {1000*b['p95_abs_mag']:8.4f}/{1000*b['max_abs_mag']:8.4f}")
    print(f"Saved results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
