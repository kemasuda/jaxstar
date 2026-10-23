"""Minimal float64 value_and_grad timings (degree-4 continuum marginalized).

JAX_PLATFORMS=cpu PYTHONPATH=src python benchmarks/benchmark_likelihood.py --pixels 2048 --orders 2
JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_likelihood.py --pixels 4096 --orders 10
PYTHONPATH=src python benchmarks/benchmark_likelihood.py --sweep --label workstation

--sweep measures 2048/4096 pixels and 1/2/4/6/8/10 orders on CPU and, if
available, GPU. Saves JSON, CSV and absolute runtime plots with linear guides.
Default output: likelihood-scaling-<environment>/ beside this script.
--cpu-only skips GPU discovery; --plot-only redraws saved sweep results.

Gradients: depth, log(jitter), log(GP amplitude), log(GP scale),
log(sigma_constant), log(sigma_continuum). Gaussian has zero GP derivatives.
Synthetic data only; no spectral-grid evaluation or parameter fitting.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.specfit import (
    Observation, chebyshev_basis, gp_marginalized_continuum_log_likelihood,
    marginalized_continuum_log_likelihood, prepare_gp_observation,
)


def benchmark(pixels, orders):
    jax.config.update("jax_enable_x64", True)
    x = .02 * np.arange(pixels)
    wave = 5000. + 100 * np.arange(orders)[:, None] + x
    profile = np.broadcast_to(np.exp(-.5 * ((x - x.mean()) / .2)**2), wave.shape)
    flux = 1 - .35 * profile + np.random.default_rng(0).normal(0, .01, wave.shape)
    obs = Observation(wave, flux, np.full_like(wave, .01))
    theta = np.r_[.3, np.log([.005, .02, .5, .1, .03])]
    inputs = jax.device_put((theta, prepare_gp_observation(obs), profile,
                             chebyshev_basis(wave, degree=4)))
    jax.block_until_ready(inputs)
    results = []
    for method in ("gaussian", "full_gp", "vecchia1"):
        def log_likelihood(theta, data, profile, basis):
            jitter, amplitude, scale, constant, continuum = jnp.exp(theta[1:])
            model_flux = 1 - theta[0] * profile
            options = dict(jitter=jitter, sigma_constant=constant,
                           sigma_continuum=continuum, degree=4, basis=basis)
            if method == "gaussian":
                return marginalized_continuum_log_likelihood(
                    data.observation, model_flux, **options)
            return gp_marginalized_continuum_log_likelihood(
                data, model_flux, gp_amplitude=amplitude, gp_scale=scale,
                gp_solver="auto" if method == "full_gp" else "vecchia1", **options)

        start = perf_counter()
        compiled = jax.jit(jax.value_and_grad(log_likelihood)).lower(*inputs).compile()
        compile_seconds = perf_counter() - start
        for _ in range(3):
            jax.block_until_ready(compiled(*inputs))
        durations = []
        for _ in range(10):
            start = perf_counter()
            value, grad = jax.block_until_ready(compiled(*inputs))
            durations.append(perf_counter() - start)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise RuntimeError(f"{method}: nonfinite value or gradient")
        results.append(dict(
            backend=jax.default_backend(), pixels=pixels, orders=orders, method=method,
            compile_s=compile_seconds, median_ms=1000 * float(np.median(durations)),
            p25_ms=1000 * float(np.percentile(durations, 25)),
            p75_ms=1000 * float(np.percentile(durations, 75)),
            log_likelihood=float(value), gradient=np.asarray(grad).tolist(),
            samples_ms=(1000 * np.asarray(durations)).tolist()))
        del compiled
    import tinygp
    return dict(device=jax.devices()[0].device_kind, jax=jax.__version__,
                tinygp=tinygp.__version__, results=results)


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("gaussian", "full_gp", "vecchia1")
    titles = ("Gaussian", "Full GP (CPU: quasisep / GPU: direct)", "Vecchia1")
    devices = " | ".join(f"{key.upper()}: {value}" for key, value in report["devices"].items())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    for row, pixels in enumerate((2048, 4096)):
        for col, (method, title) in enumerate(zip(methods, titles)):
            ax = axes[row, col]
            for backend, color in (("cpu", "#2563eb"), ("gpu", "#e76f00")):
                data = sorted((r for r in report["results"] if r["backend"] == backend
                               and r["pixels"] == pixels and r["method"] == method),
                              key=lambda r: r["orders"])
                if not data:
                    continue
                x = np.array([r["orders"] for r in data])
                ax.plot(x, [r["median_ms"] for r in data], "o-",
                        color=color, label=backend.upper())
                ax.fill_between(x, [r["p25_ms"] for r in data],
                                [r["p75_ms"] for r in data], color=color, alpha=.15)
                baseline = next((r["median_ms"] for r in data if r["orders"] == 1), None)
                if baseline is not None:
                    ax.plot(x, baseline * x, "--", color=color, alpha=.6,
                            label=f"{backend.upper()} linear from 1 order")
            ax.set(title=f"{title}\n{pixels} pixels / order", xlabel="Orders",
                   ylabel="value_and_grad [ms]", xticks=[1, 2, 4, 6, 8, 10], yscale="log")
            ax.grid(which="both", alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle(f"Likelihood value_and_grad runtime — {report['label']}\n{devices}\n"
                 f"Python {report['python']} | JAX {report['jax']} | tinygp {report['tinygp']}",
                 fontsize=12)
    fig.text(.5, .015, "float64; degree-4 continuum marginalized; 3 warmups; median of 10 synchronized calls; "
             "shading: 25–75%; compilation excluded", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"runtime.{suffix}", dpi=180)
    plt.close(fig)


def sweep(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Match the repository's workaround for system/pip CUDA library conflicts.
    gpu_env = dict(env, JAX_PLATFORMS="", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    gpu_env.pop("LD_LIBRARY_PATH", None)
    backends = [("cpu", dict(env, JAX_PLATFORMS="cpu"))]
    if not args.cpu_only:
        probe = subprocess.run([sys.executable, "-c",
                                "import jax; assert jax.default_backend() == 'gpu', 'No GPU backend available'"],
                               env=gpu_env, capture_output=True, text=True)
        if probe.returncode == 0:
            backends.append(("gpu", gpu_env))
        else:
            (output / "gpu_probe.log").write_text(probe.stderr)
            print(f"GPU unavailable; using CPU only (details: {output / 'gpu_probe.log'}).", flush=True)
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    report = dict(label=args.label or f"{socket.gethostname().split('.')[0]} / {Path(sys.prefix).name}",
                  python=platform.python_version(), devices={"cpu": cpu}, results=[])
    for backend, child_env in backends:
        for pixels in (2048, 4096):
            for orders in (1, 2, 4, 6, 8, 10):
                print(f"{backend.upper()}: pixels={pixels}, orders={orders}", flush=True)
                child = subprocess.run([sys.executable, str(script), "--pixels", str(pixels),
                                        "--orders", str(orders), "--json"],
                                       env=child_env, capture_output=True, text=True)
                (output / f"{backend}_{pixels}_{orders}.log").write_text(child.stderr)
                child.check_returncode()
                result = json.loads(child.stdout)
                if any(r["backend"] != backend for r in result["results"]):
                    raise RuntimeError(f"Requested {backend}, but received a different backend")
                if backend == "gpu":
                    report["devices"]["gpu"] = result["device"]
                report.update(jax=result["jax"], tinygp=result["tinygp"])
                report["results"].extend(result["results"])
                (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print("  " + ", ".join(f"{r['method']}: {r['median_ms']:.3f} ms"
                                        for r in result["results"]), flush=True)
    columns = ("backend", "pixels", "orders", "method", "compile_s", "median_ms",
               "p25_ms", "p75_ms", "log_likelihood")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])
    plot(report, output)
    print(f"Saved results and plots to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=2048)
    parser.add_argument("--orders", type=int, default=2)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--sweep", action="store_true")
    action.add_argument("--plot-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="skip GPU discovery in a sweep")
    parser.add_argument("--label", help="environment name for plot titles and the default output directory")
    parser.add_argument("--output", type=Path,
                        help="output directory (default: likelihood-scaling-<environment>/ beside this script)")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pixels < 2 or args.orders < 1:
        parser.error("pixels >= 2 and orders >= 1 are required")
    if args.output is None:
        label = args.label or f"{socket.gethostname().split('.')[0]}-{Path(sys.prefix).name}"
        slug = re.sub(r"[^\w.-]+", "-", label.lower()).strip("-.")[:32] or "local"
        args.output = Path(__file__).resolve().parent / f"likelihood-scaling-{slug}"
    if args.sweep:
        sweep(args)
    elif args.plot_only:
        report = json.loads((args.output / "results.json").read_text())
        if args.label:
            report["label"] = args.label
        plot(report, args.output)
    else:
        report = benchmark(args.pixels, args.orders)
        if args.json:
            print(json.dumps(report))
            return
        print(f"{report['device']}, float64, pixels={args.pixels}, orders={args.orders}")
        print(f"{'method':<12} {'compile_s':>10} {'median_ms':>10} {'log_likelihood':>18}")
        for r in report["results"]:
            print(f"{r['method']:<12} {r['compile_s']:10.3f} {r['median_ms']:10.3f} {r['log_likelihood']:18.6f}")
            print(f"  grad = {np.asarray(r['gradient'])}")


if __name__ == "__main__":
    main()
