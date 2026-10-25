"""Compare Korg's internal Julia threading, using the production synthesis path.

One fresh Python/Julia process per thread count, run sequentially. This is not
parallel execution of grid nodes. Generated results stay under results/.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter, process_time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POINTS = {
    "solar_dwarf": dict(teff=5950., logg=4., feh=0., alpha=0.),
    "cool_rich_dwarf": dict(teff=4600., logg=4.5, feh=.5, alpha=.25),
    "warm_poor_subgiant": dict(teff=6700., logg=3.5, feh=-.5, alpha=-.25),
}


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_preparation():
    spec = importlib.util.spec_from_file_location("korg_thread_recipe", ROOT / "scripts/specfit/prepare_grid.py")
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    return preparation


def worker(args):
    sys.path.insert(0, str(ROOT / "src"))
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    from jaxstar.specfit import korg
    from jaxstar.specfit.numpyro_model import empirical_vmic
    from jaxstar.specfit.sampling import _target_wavelengths

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    preparation = load_preparation()
    recipe, csv_path, settings = preparation.read_recipe(args.recipe)
    if recipe["library"] != "korg":
        raise ValueError("Expected a Korg recipe")
    for point in POINTS.values():
        if any(value not in settings["axes"][name] for name, value in point.items()):
            raise ValueError(f"Benchmark point not on recipe grid: {point}")
    wavelength = _target_wavelengths(settings["wavelength_ranges"], settings["sampling"],
                                     settings.get("pixels"), settings.get("velocity_step"))
    report = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_threads": args.worker, "repeats": args.repeats,
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "logical_cpus": os.cpu_count()},
        "sources": {"recipe": preparation.file_record(args.recipe),
                    "wavelength_csv": preparation.file_record(csv_path),
                    "backend": preparation.file_record(Path(korg.__file__)),
                    "benchmark": preparation.file_record(Path(__file__))},
        "settings": settings, "cases": {},
        "environment": {name: os.environ.get(name) for name in (
            "PYTHON_JULIACALL_THREADS", "JULIA_NUM_THREADS", "PYTHON_JULIACALL_HANDLE_SIGNALS",
            "JULIA_NUM_GC_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "timing_definition": "abundance/MARCS interpolation + Korg synthesis + array transfer + normalization/resampling; excludes startup, linelist loading, disk I/O, and vmic evaluation",
        "memory_definition": "cumulative process peak RSS from getrusage(RUSAGE_SELF), including startup/JIT/linelist and earlier calls; bytes (Darwin native bytes, Linux KiB converted)",
        "forced_gc": "once after setup, before first synthesis; normal automatic GC during all timed calls",
    }

    def checkpoint():
        report["peak_rss_bytes"] = peak_rss_bytes()
        write_json(output / "timing.json", report)

    checkpoint()
    try:
        print(f"threads={args.worker}: initialize Julia/Korg", flush=True)
        start = perf_counter()
        runtime = korg._get_runtime()
        runtime.jl.seval("using LinearAlgebra; BLAS.set_num_threads(1)")
        actual = int(runtime.jl.seval("Threads.nthreads()"))
        if actual != args.worker:
            raise ValueError(f"Requested {args.worker} threads but Julia uses {actual}")
        report["actual_threads"] = actual
        report["blas_threads"] = int(runtime.jl.seval("BLAS.get_num_threads()"))
        report["julia_import_seconds"] = perf_counter() - start
        target = runtime.to_vacuum(wavelength, settings["wavelength_medium"])
        step, buffer = settings["synthesis_step"], settings["line_buffer"]
        if step > np.min(np.diff(target, axis=1)):
            raise ValueError("Native step coarser than output sampling")
        intervals = korg._merged_intervals(target[:, [0, -1]] + [-step, step])
        windows = runtime.windows(intervals, step)
        report["vacuum_intervals_A"] = intervals
        print(f"threads={actual}: preflight {len(POINTS)} atmospheres", flush=True)
        start = perf_counter()
        for point in POINTS.values():
            runtime.atmosphere(point)
        report["preflight_seconds"] = perf_counter() - start
        checkpoint()
        print(f"threads={actual}: load GES", flush=True)
        start = perf_counter()
        runtime.load_lines()
        report["linelist_load_seconds"] = perf_counter() - start
        report["runtime"] = runtime.provenance
        if any(lo - buffer < runtime.coverage[0] or hi + buffer > runtime.coverage[1]
               for lo, hi in intervals):
            raise ValueError("Requested wavelength range exceeds GES coverage")
        runtime.jl.seval("GC.gc()")
        checkpoint()

        for name, point in POINTS.items():
            vmic = float(empirical_vmic(point["teff"], point["logg"], point["feh"]))
            case = {"point": point, "vmic_km_s": vmic, "calls": []}
            report["cases"][name] = case
            previous, reference = None, None
            for iteration in range(args.repeats + 1):
                print(f"threads={actual}: {name}, {'first' if iteration == 0 else f'warm {iteration}'}", flush=True)
                cpu_start, start = process_time(), perf_counter()
                solution = runtime.synthesize(runtime.atmosphere(point), windows, vmic, buffer)
                flux = korg._resample(solution, target, intervals)
                wall, cpu = perf_counter() - start, process_time() - cpu_start
                entry = {"seconds": wall, "cpu_seconds": cpu, "cpu_over_wall": cpu / wall,
                         "peak_rss_bytes": peak_rss_bytes()}
                if previous is not None:
                    difference = flux - previous
                    entry["repeat_max_abs_difference"] = float(np.max(np.abs(difference)))
                if iteration == 1:
                    reference = flux.copy()
                previous = flux
                case["calls"].append(entry)
                case["native_pixels"] = len(solution[0])
                print(f"  {wall:.3f} s, CPU/wall={cpu / wall:.2f}, peak RSS={peak_rss_bytes() / 2**30:.2f} GiB", flush=True)
                checkpoint()
                del solution
            case["first_seconds"] = case["calls"][0]["seconds"]
            case["warm_median_seconds"] = float(np.median([c["seconds"] for c in case["calls"][1:]]))
            case["warm_median_cpu_over_wall"] = float(np.median([c["cpu_over_wall"] for c in case["calls"][1:]]))
            np.savez(output / f"{name}.npz", wavelength=wavelength, flux=reference,
                     point=list(point.values()), regions=settings["regions"])
            checkpoint()
        report["status"] = "complete"
        checkpoint()
        print(f"threads={actual}: complete", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        checkpoint()
        raise


def compare(output, thread_counts):
    import numpy as np
    reports = {n: json.loads((output / f"threads_{n}/timing.json").read_text()) for n in thread_counts}
    baseline = reports[1]
    summary = {"thread_order": thread_counts, "cases": {}, "process_peak_rss_GiB": {}}
    for n, report in reports.items():
        if report["status"] != "complete":
            raise ValueError(f"Incomplete result for {n} threads")
        assert report["sources"] == baseline["sources"]
        assert report["settings"] == baseline["settings"]
        assert report["repeats"] == baseline["repeats"]
        assert ({k: v for k, v in report["runtime"].items() if k != "julia_threads"}
                == {k: v for k, v in baseline["runtime"].items() if k != "julia_threads"})
        summary["process_peak_rss_GiB"][str(n)] = report["peak_rss_bytes"] / 2**30
    for name in POINTS:
        summary["cases"][name] = {}
        with np.load(output / f"threads_1/{name}.npz") as data:
            reference, wavelength = data["flux"], data["wavelength"]
        for n, report in reports.items():
            with np.load(output / f"threads_{n}/{name}.npz") as data:
                np.testing.assert_array_equal(data["wavelength"], wavelength)
                delta = data["flux"] - reference
            seconds = report["cases"][name]["warm_median_seconds"]
            summary["cases"][name][str(n)] = dict(
                seconds=seconds, speedup=baseline["cases"][name]["warm_median_seconds"] / seconds,
                max_abs_difference=float(np.max(np.abs(delta))), rms_difference=float(np.sqrt(np.mean(delta**2))))
    write_json(output / "comparison.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/benchmark_threads")
    parser.add_argument("--recipe", type=Path, default=ROOT / "scripts/specfit/recipes/korg_gaoes.json")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir, args.recipe = args.output_dir.resolve(), args.recipe.resolve()
    if args.repeats < 1 or any(n < 1 for n in args.threads):
        parser.error("repeats and thread counts must be positive")
    if args.worker is not None:
        worker(args)
        return
    if 1 not in args.threads or len(set(args.threads)) != len(args.threads):
        parser.error("threads must include 1, without duplicates")
    if args.output_dir.exists() and not args.resume:
        parser.error("output directory exists; choose a new one or use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for count in args.threads:
        output = args.output_dir / f"threads_{count}"
        status = output / "timing.json"
        if args.resume and status.is_file() and json.loads(status.read_text())["status"] == "complete":
            print(f"Reuse completed {count}-thread benchmark", flush=True)
            continue
        environment = dict(os.environ, PYTHON_JULIACALL_THREADS=str(count), JULIA_NUM_THREADS=str(count),
                           PYTHON_JULIACALL_HANDLE_SIGNALS="yes", JULIA_NUM_GC_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        environment.setdefault("PYTHON_JULIAPKG_PROJECT", str(HERE / ".juliacall"))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(count),
                   "--output-dir", str(output), "--recipe", str(args.recipe), "--repeats", str(args.repeats)]
        output.mkdir(exist_ok=True)
        with (output / "stdout.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code != 0:
            raise RuntimeError(f"{count}-thread worker exited {code}; see {output / 'stdout.log'}")
    compare(args.output_dir, args.threads)


if __name__ == "__main__":
    main()
