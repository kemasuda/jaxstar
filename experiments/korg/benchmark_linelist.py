"""Time GALAH and GES at one atmosphere through Python/juliacall.

The existing mini grid remains a historical vmic=1 experiment. This benchmark
uses jaxstar's empirical vmic and Korg's standard alpha group instead.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from mini_grid import HERE, ROOT, merge_intervals, preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/benchmark_ges")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step", type=float, default=.0025)
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.step) or args.step <= 0:
        parser.error("--repeats and --step must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another --output-dir")

    import jax
    jax.config.update("jax_enable_x64", True)
    from jaxstar.specfit.numpyro_model import empirical_vmic
    from jaxstar.specfit.sampling import _target_wavelengths

    # Reuse only the wavelength/region/sampling settings of the production
    # recipe. Its BOSZ atmosphere axes do not define this one-point benchmark.
    recipe_path = ROOT / "scripts/specfit/recipes/bosz_gaoes.json"
    _, csv_path, settings = preparation.read_recipe(recipe_path)
    wavelength = _target_wavelengths(settings["wavelength_ranges"], "log",
                                     settings.get("pixels"), settings.get("velocity_step"))
    if settings["wavelength_medium"] != "air":
        raise ValueError("This benchmark expects the existing GAOES air CSV")
    if args.step > np.min(np.diff(wavelength, axis=1)):
        parser.error("--step must be no coarser than the output sampling")
    small_indices = [settings["regions"].index(r) for r in (2, 3)]
    point = dict(teff=6000., logg=4.25, feh=0., alpha=0.)
    vmic = float(empirical_vmic(point["teff"], point["logg"], point["feh"]))
    if not np.isfinite(vmic) or vmic <= 0:
        raise ValueError(f"Invalid empirical vmic: {vmic}")
    output.mkdir(parents=True)
    report = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "point": point, "vmic_km_s": vmic,
        "vmic_prescription": "jaxstar.specfit.numpyro_model.empirical_vmic(teff, logg, feh)",
        "vmic_function_source": inspect.getsource(empirical_vmic),
        "solar_abundances": "bergemann_2025_solar_abundances",
        "alpha_elements": "Korg.default_alpha_elements",
        "normalization": "solution.flux / solution.cntm",
        "wavelength_medium": "air", "synthesis_medium": "vacuum",
        "step_A": args.step, "line_buffer_A": 10.,
        "hydrogen_lines": True, "use_MHD_for_hydrogen_lines": True,
        "additional_broadening": "none", "include_molecules": True,
        "timing_definition": "Per-call abundance/atmosphere calculation, synthesis, array transfer, normalization and resampling; excludes linelist loading and disk I/O",
        "warm_measurement": "Same point repeated without reusing a synthesis or chemical-equilibrium solution",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "jax": jax.__version__, "juliacall": version("juliacall")},
        "sources": {"wavelength_recipe": preparation.file_record(recipe_path),
                    "wavelength_csv": preparation.file_record(csv_path),
                    "benchmark_script": preparation.file_record(Path(__file__)),
                    "vmic_module": preparation.file_record(Path(inspect.getfile(empirical_vmic)))},
        "loads": {}, "cases": {},
    }

    def checkpoint():
        preparation.write_json(output / "timing.json", report)

    checkpoint()
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(HERE / ".juliacall"))
    os.environ.setdefault("JULIA_NUM_THREADS", "1")
    start = perf_counter()
    import juliacall
    jl = juliacall.newmodule("JaxstarKorgTiming")
    jl.seval("using Korg, Pkg")
    korg = jl.Korg
    report["julia_import_seconds"] = perf_counter() - start
    report["versions"].update(julia=str(jl.seval("string(VERSION)")),
                                korg=str(jl.seval("string(pkgversion(Korg))")))
    report["julia_threads"] = int(jl.seval("Threads.nthreads()"))
    if report["versions"]["korg"] != "1.3.1":
        raise RuntimeError("Benchmark expects Korg 1.3.1")
    solar = korg.bergemann_2025_solar_abundances
    solar_numpy = np.array(solar)
    alphas = np.array(korg.default_alpha_elements, dtype=int)
    np.testing.assert_array_equal(alphas, [8, 10, 12, 14, 16, 18, 20, 22])
    report["alpha_atomic_numbers"] = alphas.tolist()
    target_vacuum = np.array(jl.broadcast(korg.air_to_vacuum, wavelength))
    package_root = Path(str(jl.pathof(korg))).parents[1]

    def run_case(label, lines, indices):
        target = target_vacuum[indices]
        intervals = merge_intervals(target[:, [0, -1]] + [-args.step, args.step])
        windows = jl.seval("Tuple{Float64,Float64,Float64}[]")
        for lo, hi in intervals:
            jl.push_b(windows, (lo, hi, args.step))
        case = {"regions": [settings["regions"][i] for i in indices],
                "air_bounds_A": wavelength[indices][:, [0, -1]].tolist(),
                "merged_vacuum_intervals_A": intervals,
                "unique_span_A": sum(hi - lo for lo, hi in intervals),
                "times_seconds": []}
        report["cases"][label] = case
        previous = None
        for repeat in range(args.repeats + 1):
            print(f"{label}: {'first call' if repeat == 0 else f'warm {repeat}'}", flush=True)
            start = perf_counter()
            abundances = korg.format_A_X(point["feh"], point["feh"] + point["alpha"],
                                         solar_abundances=solar)
            actual = np.array(abundances) - solar_numpy
            np.testing.assert_allclose(actual[25], point["feh"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual[alphas - 1] - actual[25], point["alpha"],
                                       atol=1e-12, rtol=0)
            atm = korg.interpolate_marcs(point["teff"], point["logg"], abundances,
                                         perturb_at_grid_values=False)
            sol = korg.synthesize(atm, lines, abundances, windows, vmic=vmic,
                                  return_cntm=True, line_buffer=10.,
                                  hydrogen_lines=True, use_MHD_for_hydrogen_lines=True)
            native_wavelength = np.array(sol.wavelengths)
            native_flux, continuum = np.array(sol.flux), np.array(sol.cntm)
            if (np.any(~np.isfinite(native_flux)) or np.any(~np.isfinite(continuum))
                    or np.any(continuum <= 0) or np.any(~np.isfinite(native_wavelength))
                    or np.any(np.diff(native_wavelength) <= 0)):
                raise ValueError(f"Invalid spectrum in {label}")
            normalized = native_flux / continuum
            rows = []
            for row in target:
                lo, hi = next((lo, hi) for lo, hi in intervals
                              if lo <= row[0] and hi >= row[-1])
                keep = (native_wavelength >= lo) & (native_wavelength <= hi)
                w, f = native_wavelength[keep], normalized[keep]
                if len(w) < 2 or w[0] > row[0] or w[-1] < row[-1]:
                    raise ValueError("Synthesis does not bracket the target wavelengths")
                rows.append(np.interp(row, w, f))
            flux = np.asarray(rows)
            elapsed = perf_counter() - start
            case["times_seconds"].append(elapsed)
            print(f"  {elapsed:.3f} s", flush=True)
            if previous is not None:
                np.testing.assert_allclose(flux, previous, rtol=0, atol=1e-12)
            previous = flux
            checkpoint()
        case.update(first_seconds=case["times_seconds"][0],
                    warm_median_seconds=float(np.median(case["times_seconds"][1:])),
                    native_pixels=len(native_wavelength),
                    normalized_flux_min_per_region=np.min(flux, axis=1).tolist(),
                    normalized_flux_max_per_region=np.max(flux, axis=1).tolist(),
                    repeated_spectra_agree=True)
        np.savez_compressed(output / f"{label}.npz", wavelength=wavelength[indices],
                            normalized_flux=flux, regions=case["regions"])
        # Count the lines Korg retains, outside the timed calls.
        case["lines_in_synthesis_windows"] = int(jl.length(
            korg.filter_linelist(lines, korg.Wavelengths(windows), 10e-8)))
        checkpoint()

    for name in ("galah", "ges"):
        print(f"Loading {name} line list (molecules included)", flush=True)
        start = perf_counter()
        lines = (korg.get_GALAH_DR3_linelist() if name == "galah"
                 else korg.get_GES_linelist(include_molecules=True))
        elapsed = perf_counter() - start
        report["loads"][name] = {"seconds": elapsed, "line_count": int(jl.length(lines))}
        print(f"Loaded {name}: {int(jl.length(lines))} lines in {elapsed:.3f} s", flush=True)
        if name == "galah":
            path = package_root / "data/linelists/GALAH_DR3/galah_dr3_linelist.h5"
        else:
            directory = str(jl.seval('Pkg.Artifacts.artifact_path(Pkg.Artifacts.artifact_hash("Heiter_2021_GES_linelist", joinpath(dirname(dirname(pathof(Korg))), "Artifacts.toml")))'))
            path = Path(directory) / "Heiter_et_al_2021_2022_06_17/Heiter_et_al_2021.h5"
            report["loads"][name]["artifact_tree_hash"] = Path(directory).name
        report["loads"][name]["file"] = preparation.file_record(path)
        checkpoint()
        # Release loader temporaries before synthesis. Normal automatic GC is
        # included in the per-call timings; there is no forced GC between calls.
        jl.seval("GC.gc()")
        run_case(f"{name}_two_regions", lines, small_indices)
        if name == "ges":
            run_case("ges_all_gaoes", lines, list(range(len(wavelength))))
        lines = None
        jl.seval("GC.gc()")

    report["ges_over_galah_two_regions"] = (
        report["cases"]["ges_two_regions"]["warm_median_seconds"]
        / report["cases"]["galah_two_regions"]["warm_median_seconds"])
    project = Path(str(jl.seval("Base.active_project()")))
    for name in ("Project.toml", "Manifest.toml"):
        (output / f"julia_{name}").write_bytes((project.parent / name).read_bytes())
    report["status"] = "complete"
    checkpoint()
    print(json.dumps({k: v["warm_median_seconds"] for k, v in report["cases"].items()}, indent=2))
    print(f"Saved {output / 'timing.json'}", flush=True)


if __name__ == "__main__":
    main()
