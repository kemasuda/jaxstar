"""Compare v2.5 BC conversion with paired official web-interpolator outputs.

Inputs are local complete UBVRIplus.txz and ZIPs from the MIST track form,
photometry option (each contains both .track.eep and .track.eep.cmd). No network
access is performed. See mist_v25/README.md for reference requests and results.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from jaxstar.mistfit.mistgrid import _tracks_v25 as v25


def table(text, first_column):
    names = next(line.lstrip("#").split() for line in text.splitlines()
                 if line.lstrip("#").split()[:1] == [first_column])
    return dict(zip(names, np.loadtxt(io.StringIO(text), ndmin=2).T))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(bc_path, reference_paths, iso_paths=(), fixture_path=None):
    axes, fields = v25.read_bc_archive(bc_path)
    report = {"bc_sha256": digest(bc_path), "bands": list(v25.BANDS), "tracks": [], "isochrones": []}
    fixture = {key: [] for key in ("raw", "expected_magnitudes", "bc_z", "bc_t", "bc_g", "bc_values")}
    for path in reference_paths:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".track.eep"):
                    continue
                data, eep, info = v25.read_track(io.BytesIO(archive.read(member)))
                if info["composition"]["[a/Fe]"] != 0.:
                    raise ValueError("Only solar-scaled chemistry is validated")
                cmd = table(archive.read(member+".cmd").decode("ascii"), "star_age")
                for raw, reference in (("star_age", "star_age"), ("log_Teff", "log_Teff"),
                                       ("log_g", "log_g"), ("log_L", "log_L")):
                    np.testing.assert_allclose(data[raw], cmd[reference], rtol=1e-12, atol=1e-12)
                feh = info["composition"]["[Fe/H]"]
                metallicity, _ = v25.surface_composition(data, feh)
                queries = np.column_stack((metallicity, data["log_Teff"], data["log_g"]))
                supported = v25.bc_coverage(axes, queries)[0]
                expected = np.column_stack([cmd[b] for b in v25.BANDS])
                predictions = v25.MBOL_SUN - 2.5*data["log_L"][supported, None] - v25.track_bc(axes, fields, queries[supported])
                residual = predictions-expected[supported]
                max_error = np.max(np.abs(residual), axis=1)
                report["tracks"].append({"archive": str(path), "sha256": digest(path), "member": member,
                    "mass": info["initial_mass"], **info["composition"], "native_nodes": len(eep),
                    "supported_nodes": int(supported.sum()), "rms_mag": float(np.sqrt(np.mean(residual**2))),
                    "max_abs_mag": float(max_error.max()), "worst_eep": int(eep[supported][max_error.argmax()])})
                # Evenly spaced native samples plus the largest discrepancy;
                # freeze official values and only their local BC stencils.
                indices = np.flatnonzero(supported)
                chosen = np.unique(np.r_[indices[np.linspace(0, len(indices)-1, 24, dtype=int)],
                                         indices[max_error.argmax()]])
                for i in chosen:
                    fixture["raw"].append([feh, data["log_surf_cell_z"][i], data["surface_h1"][i],
                                           data["log_Teff"][i], data["log_g"][i], data["log_L"][i]])
                    fixture["expected_magnitudes"].append(expected[i])
                    cells = [np.arange(n)+np.clip(np.searchsorted(axis, q, side="right")-offset, 0, len(axis)-n)
                             for axis, q, n, offset in zip(axes, queries[i], (4, 4, 2), (2, 2, 1))]
                    for key, axis, cell in zip(("bc_z", "bc_t", "bc_g"), axes, cells):
                        fixture[key].append(axis[cell])
                    fixture["bc_values"].append(fields[np.ix_(*cells)])
    iso_raw, iso_expected = [], []
    for path in iso_paths:
        with zipfile.ZipFile(path) as archive:
            member = next(n for n in archive.namelist() if n.endswith(".iso"))
            raw = table(archive.read(member).decode("ascii"), "EEP")
            cmd = table(archive.read(member+".UBVRIplus").decode("ascii"), "EEP")
            np.testing.assert_array_equal(raw["EEP"], cmd["EEP"])
            # Exclude the H-depleted tail where single precision in the
            # official CMD generator loses the tiny surface metal fraction.
            selected = (raw["EEP"] >= 202) & (raw["EEP"] <= 550)
            metal, feh = v25.surface_composition(raw, cmd["[Fe/H]_init"])
            error = feh[selected]-cmd["[Fe/H]"][selected]
            report["isochrones"].append({"archive": str(path), "sha256": digest(path),
                "eep_range": [202, 550], "nodes": int(selected.sum()), "max_abs_feh": float(np.max(abs(error)))})
            indices = np.flatnonzero(selected)[::15]
            iso_raw.extend(np.column_stack((cmd["[Fe/H]_init"][indices], raw["log_surf_cell_z"][indices],
                                            raw["surface_h1"][indices])))
            iso_expected.extend(cmd["[Fe/H]"][indices])
    if fixture_path is not None:
        np.savez_compressed(fixture_path, **{k: np.asarray(v) for k, v in fixture.items()},
                            iso_raw=np.asarray(iso_raw), iso_expected=np.asarray(iso_expected))
        report["fixture_sha256"] = digest(fixture_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=Path, required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--iso-reference", type=Path, nargs="*", default=[])
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.bc, args.reference, args.iso_reference, args.fixture)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if (any(r["max_abs_mag"] > 5e-4 for r in report["tracks"])
            or any(r["max_abs_feh"] > 5e-6 for r in report["isochrones"])):
        raise SystemExit("Official reference comparison exceeded tolerance; inspect report")


if __name__ == "__main__":
    main()
