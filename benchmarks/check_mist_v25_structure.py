"""Compare a baseline/66-column v2.5 grid and postprocess saved NUTS draws.

Run from the repository root with PYTHONPATH=src. No sampling or downloads.
The output directory must not exist; saved inference bundles are read only.
See mist_v25/README.md for the command and interpretation of the diagnostics.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_grid(path):
    import jax
    import psutil
    from jaxstar.mistfit import MistModel

    jax.config.update("jax_enable_x64", True)
    jax.devices()  # exclude first device initialization from the load measurement
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    model = MistModel.from_tracks(path)
    jax.block_until_ready(model)
    seconds = time.perf_counter()-start
    return {"load_seconds": seconds,
            "rss_delta_bytes": process.memory_info().rss-rss_before,
            "model_array_bytes": sum(x.nbytes for x in jax.tree_util.tree_leaves(model)),
            "field_count": len(model.grid.field_names),
            "npz_bytes": Path(path).stat().st_size}


def compare_sources(data, metadata, columns):
    """Independent header/loadtxt read: every selected native field/node."""
    masses = {round(float(m), 8): i for i, m in enumerate(data["initial_mass"])}
    metallicities = {float(z): j for j, z in enumerate(data["initial_feh"])}
    first_eep, last_eep = data["eep"][[0, -1]]
    tracks = nodes = 0
    for source in metadata["inputs"]:
        if source["source"]["kind"] != "tracks":
            continue
        # Original source paths are used only for reading, never echoed in the report.
        with tarfile.open(source["path"], "r|xz") as archive:
            for member in archive:
                match = re.fullmatch(r"(\d+)M\.track\.eep", Path(member.name).name)
                if not member.isfile() or not match:
                    continue
                mass = int(match[1])/100
                if mass not in masses:
                    continue
                with archive.extractfile(member) as stream:
                    text = stream.read().decode("ascii")
                lines = text.splitlines()
                names = next(line.lstrip("#").split() for line in lines
                             if line.lstrip("#").split()[:1] == ["star_age"])
                assert set(columns) <= set(names)
                # In the actual v2.5 schema the other 102 fields are isotope abundances.
                excluded = set(names)-set(columns)
                assert len(names) == 168 and len(excluded) == 102
                assert all(re.fullmatch(r"(?:surface|center)_[a-z]+\d+", n) for n in excluded)
                start = int(next(line.split(":")[1].split()[0] for line in lines if "EEPs:" in line))
                cidx = next(i for i, line in enumerate(lines) if line.lstrip("#").split()[:1] == ["Yinit"])
                composition = dict(zip(lines[cidx].lstrip("#").split(),
                                       map(float, lines[cidx+1].lstrip("#").split())))
                j = metallicities[composition["[Fe/H]"]]
                raw = np.loadtxt(io.StringIO(text), ndmin=2)
                eep = start+np.arange(len(raw))
                selected = (eep >= first_eep) & (eep <= last_eep)
                indices = (eep[selected]-first_eep).astype(int)
                for column in columns:
                    expected = raw[selected, names.index(column)]
                    expected = np.where(np.isfinite(expected), expected, np.nan)
                    np.testing.assert_array_equal(data[column][masses[mass], j, indices], expected)
                tracks += 1
                nodes += len(indices)
    assert nodes == int(data["track_present"].sum())
    return {"tracks": tracks, "nodes": nodes, "columns": len(columns), "all_values_equal": True}


def summarize(values):
    finite = np.isfinite(values)
    return {"samples": int(values.size), "nonfinite": int((~finite).sum()),
            "nonpositive_finite": int((finite & (values <= 0)).sum()),
            "q16_q50_q84": np.quantile(values[finite], [.16, .5, .84]).tolist() if finite.any() else None}


def check(base_path, expanded_path, posterior_dir, output):
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density
    from jaxstar.mistfit import MistModel, MistObservations, load_result
    from jaxstar.mistfit.inference import model_track
    from jaxstar.mistfit.mistgrid._tracks_v25 import STRUCTURE_COLUMNS

    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    report = {"baseline": {"path": str(base_path), "sha256": digest(base_path)},
              "expanded": {"path": str(expanded_path), "sha256": digest(expanded_path)},
              "posterior": {"path": str(posterior_dir),
                            "sha256": digest(posterior_dir / "posterior.nc")}}
    with np.load(base_path, allow_pickle=False) as src:
        base = {k: src[k] for k in src.files}
    with np.load(expanded_path, allow_pickle=False) as src:
        expanded = {k: src[k] for k in src.files}
    metadata = json.loads(expanded["metadata_json"].item())
    for key in base.keys()-{"metadata_json"}:
        np.testing.assert_array_equal(base[key], expanded[key])
    report["baseline_arrays_equal"] = True
    report["shape"] = list(base["logage"].shape)
    report["added_columns"] = sorted(expanded.keys()-base.keys())
    report["native_reference"] = compare_sources(expanded, metadata, STRUCTURE_COLUMNS)
    print("Native source and baseline arrays agree.", flush=True)

    for name, path in (("baseline", base_path), ("expanded", expanded_path)):
        measurements = [json.loads(subprocess.check_output(
            [sys.executable, __file__, "--measure-grid", str(path)], text=True)) for _ in range(3)]
        report[name]["load_measurements"] = measurements
        report[name]["median_load_seconds"] = float(np.median([m["load_seconds"] for m in measurements]))
        report[name]["model_array_bytes"] = measurements[0]["model_array_bytes"]
        report[name]["npz_bytes"] = measurements[0]["npz_bytes"]
    report["measurement_notes"] = (
        "Three fresh CPU processes per grid; JAX devices initialized before timing; "
        "filesystem cache warm. Array bytes include JAX fields, axes and validity masks, "
        "not temporary arrays/compiler/runtime allocations. RSS deltas are process-specific.")

    old, new = (MistModel.from_tracks(p) for p in (base_path, expanded_path))
    idata = load_result(posterior_dir)
    posterior = idata.posterior
    parameters = {k: np.asarray(posterior[v].values) for k, v in
                  (("mass_init", "mass"), ("feh_init", "feh_init"), ("eep", "eep"), ("distance", "distance"))}
    predict = jax.jit(lambda model, p: model(p, keys=old.output_keys))
    old_predictions, new_predictions = predict(old, parameters), predict(new, parameters)
    for key in old.output_keys:
        np.testing.assert_array_equal(old_predictions[key], new_predictions[key])
    report["predictions"] = {"draws": int(parameters["eep"].size), "keys": list(old.output_keys),
                             "all_values_equal": True}
    record = json.loads((posterior_dir / "run.json").read_text())["run"]["model"]
    assert any(s["sha256"] == report["baseline"]["sha256"] for s in record["physical_model"]["sources"])
    obs = record["observations"]
    observations = MistObservations.from_mapping({k: (v, e) for k, v, e in
                                                 zip(obs["keys"], obs["values"], obs["errors"])})
    options = record["arguments"]

    def joint(model, point):
        return log_density(model_track, (observations, model), options, point)[0]

    def likelihood(model, point):
        result = model({"mass_init": point["mass"], "feh_init": point["feh_init"],
                        "eep": point["eep"], "distance": point["distance"]}, keys=observations.keys)
        residual = (jnp.array([result[k] for k in observations.keys])-jnp.array(observations.values))/jnp.array(observations.errors)
        return -.5*jnp.sum(residual**2+jnp.log(2*jnp.pi*jnp.array(observations.errors)**2))

    points = {k: parameters[v].ravel()[np.linspace(0, parameters[v].size-1, 32, dtype=int)]
              for k, v in (("mass", "mass_init"), ("feh_init", "feh_init"), ("eep", "eep"), ("distance", "distance"))}
    for name, function in (("log_likelihood", likelihood), ("constrained_log_joint", joint)):
        evaluate = jax.jit(jax.vmap(jax.value_and_grad(function, argnums=1), in_axes=(None, 0)))
        a, b = evaluate(old, points), evaluate(new, points)
        for left, right in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            assert np.isfinite(right).all()
            np.testing.assert_array_equal(left, right)
        report[name] = {"points": 32, "values_and_gradients_equal": True, "all_finite": True}
    print("All saved-draw predictions and 32 likelihood/gradient comparisons agree.", flush=True)

    raw = jax.jit(lambda model, p: model(p, keys=STRUCTURE_COLUMNS))(new, parameters)
    values = {k: np.asarray(raw[k]) for k in STRUCTURE_COLUMNS}
    # MistModel uses -inf for unsupported predictions. Postprocessing files use NaN.
    values = {k: np.where(np.isfinite(v), v, np.nan) for k, v in values.items()}
    report["native_posterior"] = {k: summarize(v) for k, v in values.items()}
    radius = np.asarray(new_predictions["radius"])
    depth = values["conv_env_top_radius"]-values["conv_env_bot_radius"]
    mass = values["conv_env_top_mass"]-values["conv_env_bot_mass"]
    # Zero boundaries/times may encode no surface convection zone in the source.
    # Keep raw columns intact; flag such draws in the derived summaries.
    has_envelope = np.isfinite(depth) & np.isfinite(mass) & (depth > 0) & (mass > 0)
    derived = {
        "conv_env_depth_rsun": np.where(has_envelope, depth, np.nan),
        "conv_env_depth_fraction": np.where(has_envelope, depth/radius, np.nan),
        "conv_env_mass_msun": np.where(has_envelope, mass, np.nan),
        "conv_env_mass_fraction": np.where(has_envelope, mass/values["star_mass"], np.nan),
    }
    for suffix in ("l_t", "l_b", "g"):
        value = values["conv_env_turnover_time_"+suffix]
        derived["turnover_"+suffix+"_days"] = np.where(has_envelope & (value > 0), value/86400., np.nan)
    derived["moment_of_inertia_cgs"] = values["total_moment_of_inertia"]
    derived["apsidal_k2"] = values["apsidal_constant_k2"]
    report["derived_posterior"] = {k: summarize(v) for k, v in derived.items()}
    report["posterior"]["shape"] = list(parameters["eep"].shape)
    report["posterior"]["without_surface_envelope"] = int((~has_envelope).sum())
    report["source_nonfinite_nodes"] = {k: int((expanded["track_present"] & ~np.isfinite(expanded[k])).sum())
                                         for k in STRUCTURE_COLUMNS}
    np.savez_compressed(output / "posterior_structure.npz", **values, **derived,
                        mass_init=parameters["mass_init"], feh_init=parameters["feh_init"],
                        eep=parameters["eep"], age_gyr=np.asarray(new_predictions["age"]))

    import matplotlib.pyplot as plt
    labels = ("Envelope thickness [$R_\\odot$]", "Envelope thickness / radius",
              "Envelope mass [$M_\\odot$]", "Envelope mass / stellar mass",
              "Local turnover, 1 mixing length [days]", "Local turnover, 0.5 mixing length [days]",
              "Global turnover [days]", "Total moment of inertia [g cm$^2$]", "Apsidal constant $k_2$")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, (key, array), label in zip(axes.flat, derived.items(), labels):
        finite = array[np.isfinite(array)]
        if finite.size:
            ax.hist(finite, bins=50, histtype="step", color="C0")
            q = np.quantile(finite, [.16, .5, .84])
            ax.axvline(q[1], color="C0", linestyle="--")
            ax.set_title(f"{q[1]:.4g} [{q[0]:.4g}, {q[2]:.4g}]", fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Posterior draws")
    fig.suptitle(f"MIST v2.5: {parameters['eep'].size:,} saved NUTS draws; no resampling")
    fig.tight_layout()
    fig.savefig(output / "posterior_structure.png", dpi=150)
    plt.close(fig)
    report["artifacts"] = ["posterior_structure.npz", "posterior_structure.png"]
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["derived_posterior"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-grid", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--expanded", type=Path)
    parser.add_argument("--posterior", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.measure_grid:
        print(json.dumps(measure_grid(args.measure_grid)))
    else:
        if not all((args.baseline, args.expanded, args.posterior, args.output)):
            parser.error("--baseline, --expanded, --posterior and --output are required")
        check(args.baseline, args.expanded, args.posterior, args.output)


if __name__ == "__main__":
    main()
