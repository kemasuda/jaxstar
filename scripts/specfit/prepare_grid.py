"""Build a reusable BOSZ or Coelho NPZ from a recipe and local raw spectra."""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


AXES = {
    "bosz": ("teff", "logg", "mh", "alpha", "carbon", "vmic"),
    "coelho": ("teff", "logg", "feh", "alpha"),
}
REQUIRED = {"library", "axes", "wavelength_ranges_file", "wav_margin",
            "sampling", "normalized"}
OPTIONAL = {"description", "regions", "wavelength_medium", "missing_nodes"}


def resolve_axis(name, values):
    """Expand an inclusive min/max/step range or validate explicit nodes."""
    if isinstance(values, dict):
        if set(values) != {"min", "max", "step"}:
            raise ValueError(f"{name} range must contain exactly min, max and step")
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in values.values()):
            raise ValueError(f"{name} range values must be finite numbers")
        lo, hi, step = (Decimal(str(values[key])) for key in ("min", "max", "step"))
        if step <= 0 or hi < lo:
            raise ValueError(f"{name} range requires step > 0 and max >= min")
        # Decimal arithmetic avoids losing endpoints for steps such as 0.1.
        count = (hi - lo) / step
        if count != count.to_integral_value():
            raise ValueError(f"{name} range: max - min must be an integer multiple of step")
        values = [float(lo + i * step) for i in range(int(count) + 1)]
    array = np.asarray(values)
    if (array.ndim != 1 or not array.size or array.dtype.kind not in "iuf"
            or not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0)):
        raise ValueError(f"{name} must be a nonempty, finite, increasing numeric array")
    return array.tolist()


def read_recipe(path):
    """Resolve shared CSV ranges relative to the recipe, preserving region IDs."""
    recipe = json.loads(path.read_text())
    if not isinstance(recipe, dict) or not REQUIRED <= recipe.keys():
        raise ValueError(f"recipe must contain {sorted(REQUIRED)}")
    if recipe.keys() - REQUIRED - OPTIONAL:
        raise ValueError(f"unknown recipe keys: {sorted(recipe.keys() - REQUIRED - OPTIONAL)}")
    library = recipe["library"]
    if library not in AXES:
        raise ValueError("library must be 'bosz' or 'coelho'")
    missing_nodes = recipe.get("missing_nodes", "error")
    if missing_nodes not in ("error", "logg"):
        raise ValueError("missing_nodes must be 'error' or 'logg'")
    if library != "bosz" and missing_nodes != "error":
        raise ValueError("missing_nodes='logg' is supported only for BOSZ")
    axes = recipe["axes"]
    if not isinstance(axes, dict) or set(axes) != set(AXES[library]):
        raise ValueError(f"{library} axes must contain exactly {AXES[library]}")
    axes = {name: resolve_axis(name, values) for name, values in axes.items()}
    if type(recipe["normalized"]) is not bool:
        raise ValueError("normalized must be true or false")
    margin = recipe["wav_margin"]
    if (type(margin) not in (int, float) or not np.isfinite(margin) or margin < 0):
        raise ValueError("wav_margin must be finite nonnegative Angstrom")
    sampling = recipe["sampling"]
    if not isinstance(sampling, dict) or set(sampling) not in ({"pixels"}, {"velocity_step"}):
        raise ValueError("sampling must specify exactly one of pixels or velocity_step")
    value = next(iter(sampling.values()))
    if "pixels" in sampling:
        if type(value) is not int or value < 2:
            raise ValueError("pixels must be an integer >= 2")
    elif type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError("velocity_step must be positive finite km/s")

    csv_path = (path.parent / recipe["wavelength_ranges_file"]).resolve()
    rows = {}
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["region", "wavmin", "wavmax", "unit", "medium"]:
            raise ValueError("CSV columns must be region,wavmin,wavmax,unit,medium")
        for row in reader:
            region = int(row["region"])
            lo, hi = float(row["wavmin"]), float(row["wavmax"])
            if region in rows or region < 0:
                raise ValueError("CSV region IDs must be unique nonnegative integers")
            if row["unit"] != "angstrom" or row["medium"] not in {"air", "vacuum", "unknown"}:
                raise ValueError("CSV requires angstrom units and air/vacuum/unknown medium")
            if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
                raise ValueError(f"region {region}: invalid wavelength bounds")
            rows[region] = (lo, hi, row["medium"])
    regions = recipe.get("regions", list(rows))
    if (not isinstance(regions, list) or not regions
            or any(type(r) is not int or r not in rows for r in regions)
            or len(set(regions)) != len(regions)):
        raise ValueError("regions must be a nonempty list of distinct CSV region IDs")
    media = {rows[r][2] for r in regions}
    if len(media) != 1:
        raise ValueError("selected regions must share a wavelength medium")
    csv_medium = media.pop()
    medium = recipe.get("wavelength_medium", csv_medium)
    if medium not in {"air", "vacuum"}:
        raise ValueError("CSV medium is unknown; declare wavelength_medium in the recipe")
    if csv_medium != "unknown" and medium != csv_medium:
        raise ValueError("wavelength_medium conflicts with CSV; no range conversion is implicit")
    bounds = [[rows[r][0] - margin, rows[r][1] + margin] for r in regions]
    if any(lo <= 0 for lo, _ in bounds):
        raise ValueError("wav_margin produces nonpositive wavelength bounds")
    kwargs = dict(axes=axes, wavelength_ranges=bounds, regions=regions,
                  wavelength_medium=medium, normalized=recipe["normalized"],
                  sampling="log", **sampling)
    if library == "bosz" and "missing_nodes" in recipe:
        kwargs["missing_nodes"] = missing_nodes
    return recipe, csv_path, kwargs


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def check_inputs(library, data_dir, kwargs, *, stream=None):
    """Print every planned replacement before allowing any spectrum reads."""
    from jaxstar.specfit import check_spectral_inputs

    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    print("Checking all input files before reading spectra", file=stream, flush=True)
    report = check_spectral_inputs(library, data_dir, axes=kwargs["axes"],
                                   missing_nodes=kwargs.get("missing_nodes", "error"))
    for entry in report["replacements"]:
        gravities = [source["node"]["logg"] for source in entry["sources"]]
        if entry["method"] == "linear_logg":
            action = f"will interpolate linearly in logg from {gravities[0]:g} and {gravities[1]:g}"
        else:
            action = (f"upper logg neighbor is also missing; will use logg={gravities[0]:g} "
                      "at the requested upper edge")
        print(f"Missing {entry['missing_path']}: {action}.", file=stream, flush=True)
    if not report["ready"]:
        details = []
        for entry in report["unresolved"]:
            details.append(f"{entry['missing_path']}: {entry['reason']}")
            details.extend(f"  Missing neighbor: {path}" for path in entry.get("missing_neighbors", ()))
        raise FileNotFoundError(
            f"Input check failed for {len(report['unresolved'])} node(s); no spectra were read.\n"
            + "\n".join(details))
    print(f"Input check passed: {report['requested_nodes']} requested nodes, "
          f"{len(report['sources'])} source files, {len(report['replacements'])} replacements.",
          file=stream, flush=True)
    return report


def build(recipe_path, data_dir, output_dir):
    """Publish a new directory only after the grid and provenance are complete."""
    recipe, csv_path, kwargs = read_recipe(recipe_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"{output_dir} already exists; choose a new output directory")
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    import jax
    import jaxstar
    from jaxstar.grid import RectilinearGrid
    from jaxstar.specfit import prepare_bosz, prepare_coelho, save_spectral_grid

    source_plan = check_inputs(recipe["library"], data_dir, kwargs)
    jax.config.update("jax_enable_x64", True)
    prepare = {"bosz": prepare_bosz, "coelho": prepare_coelho}[recipe["library"]]
    print(f"Preparing {recipe['library']}: {len(kwargs['regions'])} regions, "
          f"{np.prod([len(a) for a in kwargs['axes'].values()]):d} atmosphere nodes", flush=True)
    spectra = prepare(data_dir, **kwargs)
    # Raw libraries may be revised under the same filenames. Record their
    # compressed/native bytes, not just the paths already stored in the NPZ.
    print("Hashing raw inputs and saving the prepared grid", flush=True)
    sources = {str(Path(p).relative_to(data_dir)): file_record(Path(p))
               for p in dict.fromkeys(spectra.sources)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-grid-", dir=output_dir.parent) as staging:
        stage = Path(staging)
        save_spectral_grid(stage / "grid.npz", spectra)
        # The snapshot is itself runnable after moving the output directory.
        write_json(stage / "recipe.json", {**recipe, "wavelength_ranges_file": "wavelength_ranges.csv"})
        (stage / "wavelength_ranges.csv").write_bytes(csv_path.read_bytes())
        wave = np.asarray(spectra.wavelength)
        implementations = {"prepare_grid.py": Path(__file__),
                           "libraries.py": Path(inspect.getfile(prepare)),
                           "storage.py": Path(inspect.getfile(save_spectral_grid)),
                           "grid/core.py": Path(inspect.getfile(RectilinearGrid))}
        implementations["sampling.py"] = implementations["libraries.py"].with_name("sampling.py")
        write_json(stage / "manifest.json", dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            library=recipe["library"], data_dir=str(data_dir),
            versions=dict(python=sys.version, jaxstar=jaxstar.__version__, jax=jax.__version__, numpy=np.__version__),
            implementations={name: file_record(path) for name, path in implementations.items()},
            input_recipe=file_record(recipe_path), input_wavelength_ranges=file_record(csv_path),
            resolved_preparation=kwargs, sources=sources, source_plan=source_plan,
            flux_shape=list(spectra.grid.field("flux").values.shape),
            actual_wavelength_ranges=wave[:, [0, -1]].tolist(),
            actual_velocity_step_kms=(299792.458 * np.log(wave[:, -1] / wave[:, 0]) / (wave.shape[1]-1)).tolist(),
            outputs={name: file_record(stage / name)
                     for name in ("grid.npz", "recipe.json", "wavelength_ranges.csv")},
        ))
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"{output_dir} appeared during preparation")
        stage.rename(output_dir)
    print(f"Saved {output_dir / 'grid.npz'}", flush=True)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, help="JSON recipe; CSV paths are relative to this file")
    parser.add_argument("--data-dir", type=Path, help="local BOSZ rorig/ or Coelho FITS directory")
    parser.add_argument("--output-dir", type=Path, help="new directory for grid.npz and provenance")
    parser.add_argument("--dry-run", action="store_true", help="resolve settings; also check input paths if --data-dir is supplied; do not read spectra or write files")
    args = parser.parse_args(argv)
    try:
        recipe_path = args.recipe.expanduser().resolve()
        if args.dry_run:
            recipe, csv_path, kwargs = read_recipe(recipe_path)
            summary = dict(library=recipe["library"], wavelength_ranges_file=str(csv_path), preparation=kwargs)
            if args.data_dir is not None:
                report = check_inputs(recipe["library"], args.data_dir.expanduser().resolve(), kwargs,
                                      stream=sys.stderr)
                summary["input_check"] = {key: value for key, value in report.items() if key != "sources"}
                summary["input_check"]["source_file_count"] = len(report["sources"])
            print(json.dumps(summary, indent=2, allow_nan=False))
            return
        if args.data_dir is None or args.output_dir is None:
            parser.error("--data-dir and --output-dir are required unless --dry-run is used")
        build(recipe_path, args.data_dir.expanduser().resolve(), args.output_dir.expanduser().absolute())
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
