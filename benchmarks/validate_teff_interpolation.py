"""Leave one native Teff node out; compare linear and cubic spectral prediction.

Run with PYTHONPATH=src and JAX x64. Reads existing spectra only: no downloads,
synthesis, wavelength resampling, broadening, or inference. See
benchmarks/teff_interpolation.md for examples and interpretation.
"""

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import jax
import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import load_spectral_grid, load_coelho, load_bosz, load_tlusty
from jaxstar.specfit._interpolation import interpolate_flux


def summarize(rows):
    """Equal weight per case; pooled RMS equals pixel RMS for equal payloads."""
    out = {"cases": len(rows)}
    if not rows:
        return out
    for metric in ("rms", "max_abs", "weighted_rms"):
        linear = np.array([r[f"linear_{metric}"] for r in rows])
        cubic = np.array([r[f"cubic_{metric}"] for r in rows])
        nonzero = linear > 0
        out[metric] = dict(
            linear_mean=float(linear.mean()), cubic_mean=float(cubic.mean()),
            linear_max=float(linear.max()), cubic_max=float(cubic.max()),
            improved=int(np.sum(cubic < linear)), worse=int(np.sum(cubic > linear)),
            median_ratio=float(np.median(cubic[nonzero] / linear[nonzero])) if nonzero.any() else None)
    out["pooled_rms"] = {mode: float(np.sqrt(np.mean([r[f"{mode}_rms"]**2 for r in rows])))
                         for mode in ("linear", "cubic")}
    out["max_overshoot"] = max(r["cubic_overshoot"] for r in rows)
    return out


def metrics(prediction, truth, weights):
    residual = prediction.astype(np.float64) - truth
    return dict(rms=float(np.sqrt(np.mean(residual**2))),
                max_abs=float(np.max(np.abs(residual))),
                weighted_rms=float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights))))


def validate(spectra, coordinates, excluded_nodes=()):
    grid = spectra.grid
    field = grid.field("flux")
    temperatures = np.asarray(grid.axis("teff"))
    if len(temperatures) < 4:
        raise ValueError("held-out validation requires at least four native Teff nodes")
    names = tuple(n for n in grid.axis_names if n != "teff")
    if set(coordinates) != set(names):
        raise ValueError(f"coordinates must specify exactly {names}, using native values")
    excluded = [r["node"] for r in spectra.provenance.get("replacements", [])] + list(excluded_nodes)
    values = np.asarray(field.values)
    evaluate = {mode: jax.jit(lambda g, t, mode=mode: interpolate_flux(g, {"teff": t}, teff_interp=mode))
                for mode in ("linear", "cubic")}
    rows, skipped = [], []
    for location in product(*(coordinates[n] for n in names)):
        point = dict(zip(names, location))
        indices = {}
        for name, value in point.items():
            found = np.flatnonzero(np.isclose(np.asarray(grid.axis(name)), value, atol=1e-10, rtol=0))
            if len(found) != 1:
                raise ValueError(f"{name}={value} is not a unique native node")
            indices[name] = int(found[0])
        line = values[tuple(slice(None) if n == "teff" else indices[n] for n in field.dims)]
        replaced = {int(np.argmin(abs(temperatures - node["teff"]))) for node in excluded
                    if all(np.isclose(node.get(n, np.nan), v, atol=1e-10, rtol=0) for n, v in point.items())
                    and np.any(np.isclose(temperatures, node["teff"], atol=1e-10, rtol=0))}
        for omitted in range(1, len(temperatures) - 1):
            keep = np.delete(np.arange(len(temperatures)), omitted)
            # Reject replacements in the ground truth or either method's local
            # training stencil, so imputed spectra cannot masquerade as native.
            stencil = keep[np.clip(np.arange(omitted-2, omitted+2), 0, len(keep)-1)]
            if replaced.intersection({omitted, *stencil}):
                skipped.append(dict(**point, teff=float(temperatures[omitted]), reason="replacement in truth/stencil"))
                continue
            reduced = RectilinearGrid(axes={"teff": temperatures[keep]}, fields={
                "flux": Field(line[keep], ("teff",), payload_dims=("region", "pixel"))})
            truth = line[omitted].astype(np.float64)
            if not np.all(np.isfinite(line[np.r_[omitted, stencil]])):
                skipped.append(dict(**point, teff=float(temperatures[omitted]), reason="nonfinite truth/stencil"))
                continue
            prediction = {mode: np.asarray(fn(reduced, temperatures[omitted])) for mode, fn in evaluate.items()}
            if not all(np.all(np.isfinite(pred)) for pred in prediction.values()):
                raise ValueError("interpolation produced nonfinite predictions from finite native spectra")
            for r, region in enumerate(spectra.regions):
                row = dict(**point, teff=float(temperatures[omitted]), region=region)
                # Explicit diagnostic noise model, not measured observational
                # errors: continuum S/N=100 photon noise, flux floor=0.05.
                weights = 1 / (.01**2 * np.maximum(truth[r], .05))
                for mode, pred in prediction.items():
                    row.update({f"{mode}_{key}": value for key, value in metrics(pred[r], truth[r], weights).items()})
                lo = np.minimum(line[omitted-1, r], line[omitted+1, r])
                hi = np.maximum(line[omitted-1, r], line[omitted+1, r])
                row["cubic_overshoot"] = float(np.maximum(np.maximum(lo-prediction["cubic"][r], prediction["cubic"][r]-hi), 0).max())
                rows.append(row)
    return rows, skipped


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--format", choices=("common", "coelho", "bosz", "tlusty"), default="common")
    parser.add_argument("--coordinates", required=True, help="JSON mapping of non-Teff axes to native coordinate lists")
    parser.add_argument("--exclude-nodes", type=Path, help="JSON list of known non-native atmosphere nodes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    coordinates = json.loads(args.coordinates)
    excluded = [] if args.exclude_nodes is None else json.loads(args.exclude_nodes.read_text())
    rows, skipped, sources = [], [], []
    for path in args.paths:
        loader = {"common": load_spectral_grid, "coelho": load_coelho,
                  "bosz": load_bosz, "tlusty": load_tlusty}[args.format]
        spectra = loader(path)
        current, rejected = validate(spectra, coordinates, excluded)
        for row in current:
            row["source"] = path.name
        rows.extend(current)
        skipped.extend(dict(source=path.name, **row) for row in rejected)
        wave = np.asarray(spectra.wavelength)
        sources.append(dict(path=str(path), sha256=file_hash(path), library=spectra.library,
                            flux_dtype=str(spectra.grid.field("flux").values.dtype),
                            wavelength_ranges=np.stack([wave[:, 0], wave[:, -1]], axis=-1).tolist(),
                            pixels=wave.shape[-1], known_replacements=len(spectra.provenance.get("replacements", []))))
    args.output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no valid native holdouts")
    with (args.output / "cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(summary=summarize(rows), sources=sources, coordinates=coordinates, skipped=skipped,
                  excluded_nodes=excluded, jax_version=jax.__version__, x64=True,
                  weighting="1 / (0.01^2 * max(native normalized flux, 0.05)); diagnostic photon-noise proxy",
                  by_teff={str(t): summarize([r for r in rows if r["teff"] == t]) for t in sorted({r["teff"] for r in rows})},
                  by_location={str(p): summarize([r for r in rows if tuple(r[n] for n in coordinates) == p])
                               for p in sorted({tuple(r[n] for n in coordinates) for r in rows})})
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
