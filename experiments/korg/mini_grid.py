"""Build and validate a small Korg grid through Python/juliacall.

This experiment reuses prepare_grid's recipe parser and jaxstar's sampling and
storage. Only the per-node spectrum calculation is supplied by Korg.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "prepare_grid_script", ROOT / "scripts/specfit/prepare_grid.py"
)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)
SCHEMAS = {"korg": ("teff", "logg", "feh", "alpha", "vmic")}
# This is a private, separately loaded module instance. Extend its schema only
# in this process; the production script and its normal CLI remain unchanged.
preparation.AXES = {**preparation.AXES, **SCHEMAS}


def read_recipe(path):
    recipe, csv_path, settings = preparation.read_recipe(path)
    # Bound this initial experiment explicitly; these are not Korg's limits.
    axes = {name: settings["axes"][name] for name in SCHEMAS["korg"]}
    if (any(len(axes[name]) < 2 for name in ("teff", "logg", "feh"))
            or axes["alpha"] != [0.0] or axes["vmic"] != [1.0]):
        raise ValueError("Mini grid requires Teff/logg/FeH axes and fixed alpha=0, vmic=1")
    if not recipe["normalized"]:
        raise ValueError("This mini experiment requires normalized=true")
    if np.prod([len(a) for a in axes.values()]) > 64:
        raise ValueError("Keep the mini experiment at 64 nodes or fewer")
    if not (5000 <= min(axes["teff"]) <= max(axes["teff"]) <= 6500
            and 4 <= min(axes["logg"]) <= max(axes["logg"]) <= 4.5
            and -0.5 <= min(axes["feh"]) <= max(axes["feh"]) <= 0.5):
        raise ValueError("Mini domain: Teff 5000–6500, logg 4–4.5, FeH -0.5–0.5")
    bounds = np.asarray(settings["wavelength_ranges"])
    # GALAH has four separated bands, not continuous 4675–7930 Å coverage.
    # This conservative green-band interior leaves room for the line buffer
    # in both air and vacuum and excludes the large gaps in its line list.
    if np.any(bounds < 5660) or np.any(bounds > 5860):
        raise ValueError("Mini grid uses GALAH's green band; keep all bounds within 5660–5860 Å")
    settings["axes"] = axes
    return recipe, csv_path, settings


def merge_intervals(bounds):
    """Synthesize overlapping echelle regions together, in increasing order."""
    merged = []
    for lo, hi in sorted(map(tuple, bounds)):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([float(lo), float(hi)])
    return merged


class KorgSynthesis:
    """One Julia session and one line-list load for all nodes and validation."""

    def __init__(self, wavelength, medium, step):
        # JuliaPkg discovers the adjacent juliapkg.json. Keep its environment
        # local to this experiment unless the user explicitly selected one.
        os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(HERE / ".juliacall"))
        import juliacall

        self.jl = jl = juliacall.newmodule("JaxstarKorgMini")
        jl.seval("using Korg")
        self.korg = korg = jl.Korg
        if str(jl.seval("string(pkgversion(Korg))")) != "1.3.1":
            raise RuntimeError("This experiment is pinned to Korg 1.3.1")
        self.solar = korg.bergemann_2025_solar_abundances
        self.alphas = jl.seval("[8, 10, 12, 14, 16, 20, 22]")
        self.solar_numpy = np.array(self.solar, dtype=float)
        # Convert the evaluation coordinates, retaining the CSV's medium in
        # the saved NPZ. A flux/continuum ratio needs no flux-density Jacobian.
        self.target_vacuum = np.array(wavelength, dtype=float, copy=True)
        if medium == "air":
            self.target_vacuum = np.array(jl.broadcast(korg.air_to_vacuum, wavelength))
        elif medium != "vacuum":
            raise ValueError("Declare air or vacuum explicitly")
        self.step = step
        print("Loading GALAH DR3 line list once", flush=True)
        self.lines = korg.get_GALAH_DR3_linelist()
        self.timings = []
        self.metadata = {
            "engine": "Korg via juliacall", "korg_version": "1.3.1",
            "julia_version": str(jl.seval("string(VERSION)")),
            "juliacall_version": version("juliacall"),
            "linelist": "get_GALAH_DR3_linelist()",
            "linelist_scope": "Green band only; recipe bounds restricted to 5660–5860 Å",
            "solar_abundances": "bergemann_2025_solar_abundances",
            "solar_A_X": self.solar_numpy.tolist(),
            "alpha_elements": ["O", "Ne", "Mg", "Si", "S", "Ca", "Ti"],
            "axis_meanings": {"feh": "[Fe/H]", "alpha": "[alpha/Fe]", "vmic": "km/s"},
            "abundance_mapping": "format_A_X(feh, feh + alpha)",
            "atmosphere": "interpolate_marcs(teff, logg, A_X; perturb_at_grid_values=false)",
            "normalization": "synthesize.flux / synthesize.cntm",
            "synthesis_medium": "vacuum", "output_medium": medium,
            "air_conversion": "Korg.air_to_vacuum" if medium == "air" else "none",
            "step_A": step, "line_buffer_A": 10.0,
            "hydrogen_lines": True, "use_MHD_for_hydrogen_lines": True,
            "rotation_macroturbulence_instrumental_convolution": "none",
        }
        line_path = (Path(str(jl.pathof(korg))).parents[1]
                     / "data/linelists/GALAH_DR3/galah_dr3_linelist.h5")
        self.metadata["linelist_file"] = {"name": line_path.name,
                                           **preparation.file_record(line_path)}

    def __call__(self, point, *, step=None):
        start = perf_counter()
        step = self.step if step is None else step
        korg = self.korg
        abundances = korg.format_A_X(
            point["feh"], point["feh"] + point["alpha"],
            solar_abundances=self.solar, alpha_elements=self.alphas,
        )
        actual = np.array(abundances) - self.solar_numpy
        np.testing.assert_allclose(actual[25], point["feh"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[[7, 9, 11, 13, 15, 19, 21]] - actual[25],
                                   point["alpha"], rtol=0, atol=1e-12)
        atmosphere = korg.interpolate_marcs(
            point["teff"], point["logg"], abundances, perturb_at_grid_values=False
        )
        padded = self.target_vacuum[:, [0, -1]] + [-step, step]
        intervals = merge_intervals(padded)
        # Julia Vector{Tuple}, not a Python list: Korg's wavelength API expects
        # a Julia vector for multiple disjoint intervals.
        windows = self.jl.seval("Tuple{Float64,Float64,Float64}[]")
        for lo, hi in intervals:
            self.jl.push_b(windows, (lo, hi, step))
        solution = korg.synthesize(
            atmosphere, self.lines, abundances, windows,
            vmic=point["vmic"], return_cntm=True, line_buffer=10.0,
            hydrogen_lines=True, use_MHD_for_hydrogen_lines=True,
        )
        native_wave = np.array(solution.wavelengths)
        native_flux, continuum = np.array(solution.flux), np.array(solution.cntm)
        if (np.any(~np.isfinite(native_flux)) or np.any(~np.isfinite(continuum))
                or np.any(continuum <= 0) or np.any(~np.isfinite(native_wave))
                or np.any(np.diff(native_wave) <= 0)):
            raise ValueError(f"Invalid synthesis at {point}")
        normalized = native_flux / continuum
        flux = []
        for row in self.target_vacuum:
            # Check coverage in its own synthesis interval, including gaps.
            lo, hi = next((lo, hi) for lo, hi in intervals if lo <= row[0] and hi >= row[-1])
            keep = (native_wave >= lo) & (native_wave <= hi)
            w, f = native_wave[keep], normalized[keep]
            if len(w) < 2 or w[0] > row[0] or w[-1] < row[-1]:
                raise ValueError("Korg output does not bracket the target; no extrapolation")
            flux.append(np.interp(row, w, f))
        self.timings.append({"point": point, "step_A": step, "seconds": perf_counter() - start})
        return np.asarray(flux)


def metrics(delta):
    return {"rms": float(np.sqrt(np.mean(delta**2))),
            "max_abs": float(np.max(np.abs(delta)))}


def validate(stage, spectra, synthesis, bounds, margin):
    from jaxstar.specfit import Observation, SpecFit, load_spectral_grid

    restored = load_spectral_grid(stage / "grid.npz")
    names = restored.grid.axis_names
    np.testing.assert_array_equal(restored.wavelength, spectra.wavelength)
    np.testing.assert_array_equal(restored.grid.field("flux").values,
                                  spectra.grid.field("flux").values)
    axes = [np.asarray(restored.grid.axis(name)) for name in names]
    node_error = 0.0
    for index in np.ndindex(*(len(a) for a in axes)):
        point = dict(zip(names, (float(a[i]) for a, i in zip(axes, index))))
        got = np.asarray(restored.grid.interpolate(point)["flux"])
        expected = np.asarray(spectra.grid.field("flux").values[index])
        node_error = max(node_error, float(np.max(np.abs(got - expected))))
    if node_error > 1e-12:
        raise ValueError(f"Exact-node interpolation error: {node_error}")

    centers = [(a[1:] + a[:-1]) / 2 if len(a) > 1 else a for a in axes]
    points, direct, interpolated, reports = [], [], [], []
    for values in product(*centers):
        point = dict(zip(names, map(float, values)))
        print(f"Validate cell center {point}", flush=True)
        reference = synthesis(point)
        prediction = np.asarray(restored.grid.interpolate(point)["flux"])
        points.append(point)
        direct.append(reference)
        interpolated.append(prediction)
        reports.append({"point": point, **metrics(prediction - reference)})
    direct, interpolated = np.asarray(direct), np.asarray(interpolated)
    # One additional direct synthesis checks the native wavelength sampling.
    finer = synthesis(points[0], step=synthesis.step / 2)

    # Exercise the ordinary fitting-grid entry point with small synthetic rows.
    # These placeholders only check loading/region selection; no fit is run.
    obs_wave = np.array([np.geomspace(lo + margin, hi - margin, 64) for lo, hi in bounds])
    obs = Observation(obs_wave, np.ones_like(obs_wave), np.full_like(obs_wave, .01),
                      region=spectra.regions)
    fit = SpecFit.from_grid(stage / "grid.npz", obs, wav_margin=margin / 2, vmax=50.0)
    if fit.specmodel.spectra.library != "korg":
        raise ValueError("SpecFit did not preserve the Korg library label")
    np.testing.assert_array_equal(fit.specmodel.spectra.wavelength, spectra.wavelength)
    report = {
        "node_max_abs_error": node_error, "save_load_exact": True,
        "specfit_from_grid": True, "difference": "grid interpolation - direct Korg",
        "intrinsic_flux_errors": reports, "all_cell_centers": metrics(interpolated - direct),
        "sampling_check": {"point": points[0], "fine_step_A": synthesis.step / 2,
                           **metrics(direct[0] - finer)},
        "scope": "Intrinsic flux only; no inference or observational accuracy validation",
    }
    preparation.write_json(stage / "validation.json", report)
    np.savez(stage / "validation.npz", wavelength=np.asarray(spectra.wavelength),
             axis_names=names, coordinates=np.array([list(p.values()) for p in points]),
             direct=direct, interpolated=interpolated, finer_at_first_center=finer)
    plot_validation(stage, np.asarray(spectra.wavelength), spectra.regions,
                    points, direct, interpolated, spectra.wavelength_medium)
    return report


def plot_validation(stage, wavelength, regions, points, direct, interpolated, medium):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = interpolated - direct
    worst = int(np.argmax(np.max(np.abs(delta), axis=(1, 2))))
    fig, axes = plt.subplots(2, len(regions), squeeze=False, sharex="col",
                             figsize=(7 * len(regions), 6), height_ratios=(2, 1),
                             layout="constrained")
    for i, region in enumerate(regions):
        axes[0, i].plot(wavelength[i], direct[worst, i], lw=.8, label="Direct Korg")
        axes[0, i].plot(wavelength[i], interpolated[worst, i], lw=.8, label="Grid interpolation")
        axes[0, i].set(title=f"Region {region}", ylabel="Normalized flux")
        axes[0, i].legend()
        axes[1, i].plot(wavelength[i], delta[worst, i], lw=.8, color="tab:purple")
        axes[1, i].axhline(0, color="0.5", lw=.5)
        axes[1, i].set(xlabel=f"{medium.capitalize()} wavelength [Å]", ylabel="Interpolated − direct")
    p = points[worst]
    fig.suptitle(f"Worst cell-center maximum residual: Teff={p['teff']:g} K, "
                 f"log g={p['logg']:g}, [Fe/H]={p['feh']:g}\n"
                 "[α/Fe]=0, vmic=1 km/s · intrinsic normalized spectra")
    fig.savefig(stage / "validation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", nargs="?", type=Path, default=HERE / "recipes/korg_gaoes_mini.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/mini_gaoes")
    parser.add_argument("--step", type=float, default=.0025, help="Korg native vacuum sampling in Å")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the recipe without importing Julia")
    args = parser.parse_args()
    recipe, csv_path, settings = read_recipe(args.recipe.resolve())
    if not np.isfinite(args.step) or args.step <= 0:
        parser.error("--step must be finite and positive")
    import jax
    jax.config.update("jax_enable_x64", True)
    from jaxstar.specfit.sampling import _target_wavelengths
    # Private assembly stays confined to this experiment; no new public API.
    from jaxstar.specfit.libraries import _assemble
    from jaxstar.specfit import save_spectral_grid

    wavelength = _target_wavelengths(settings["wavelength_ranges"], "log",
                                     settings.get("pixels"), settings.get("velocity_step"))
    if args.step > np.min(np.diff(wavelength, axis=1)):
        parser.error("--step must be no coarser than the saved wavelength sampling")
    axes = settings["axes"]
    shape = tuple(len(a) for a in axes.values()) + wavelength.shape
    print(json.dumps({"preparation": settings, "flux_shape": shape,
                      "float64_bytes": int(np.prod(shape)) * 8}, indent=2), flush=True)
    if args.dry_run:
        return
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output-dir")
    start = perf_counter()
    synthesis = KorgSynthesis(wavelength, settings["wavelength_medium"], args.step)
    setup_seconds = perf_counter() - start
    flux = np.empty(shape, dtype=np.float64)
    for index in np.ndindex(*(len(a) for a in axes.values())):
        point = {name: float(a[i]) for (name, a), i in zip(axes.items(), index)}
        print(f"Synthesize node {point}", flush=True)
        flux[index] = synthesis(point)
    spectra = _assemble("korg", axes, wavelength, flux, settings["regions"],
                        settings["wavelength_medium"], "normalized", ())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".korg-mini-", dir=output.parent) as directory:
        stage = Path(directory)
        save_spectral_grid(stage / "grid.npz", spectra)
        preparation.write_json(stage / "recipe.json", {
            **recipe, "wavelength_ranges_file": "wavelength_ranges.csv"
        })
        (stage / "wavelength_ranges.csv").write_bytes(csv_path.read_bytes())
        report = validate(stage, spectra, synthesis, settings["wavelength_ranges"], recipe["wav_margin"])
        project = Path(str(synthesis.jl.seval("Base.active_project()")))
        for name in ("Project.toml", "Manifest.toml"):
            (stage / f"julia_{name}").write_bytes((project.parent / name).read_bytes())
        preparation.write_json(stage / "manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "jax": jax.__version__, "jaxstar": version("jaxstar")},
            "resolved_preparation": settings, "synthesis": synthesis.metadata,
            "setup_seconds": setup_seconds, "calculations": synthesis.timings,
            "implementation": {"mini_grid.py": preparation.file_record(Path(__file__)),
                               "prepare_grid.py": preparation.file_record(Path(preparation.__file__))},
            "outputs": {p.name: preparation.file_record(p) for p in stage.iterdir()},
        })
        if output.exists():
            raise FileExistsError(f"{output} appeared during generation")
        stage.rename(output)
    print(f"Saved {output / 'grid.npz'}", flush=True)
    print(json.dumps({"interpolation": report["all_cell_centers"],
                      "sampling": report["sampling_check"]}, indent=2))


if __name__ == "__main__":
    main()
