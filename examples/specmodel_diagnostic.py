"""Batch sample-data stages, legacy parity and parameter response plots.

For a step-by-step interactive tutorial, see tutorials/specmodel.ipynb.
This CLI retains the three-library, parameter-override and file-output workflow.

PYTHONPATH=src python examples/specmodel_diagnostic.py --reference-root ../jaxspec \
    --output-dir /private/tmp/jaxstar-specmodel-plots

Use --vsini/--vmacro/--resolving-power/--rv to vary one scalar, or comma-separated
per-region values. No fitting; no sample grids/reference artifacts are modified.
"""

import argparse
import copy
import json
from pathlib import Path
import tempfile

import jax
import numpy as np

from jaxstar.specfit import SpecModel, load_spectral_grid, save_spectral_grid
from _specmodel_reference import frozen_stages, make_case, verify_reference


def values(text):
    result = np.array([float(value) for value in text.split(",")])
    return float(result[0]) if len(result) == 1 else result


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _specmodel_plotting import display_wavelength, plot_comparison, plot_spectra
    root, output = Path(args.reference_root).resolve(), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    verify_reference(root)
    with tempfile.TemporaryDirectory(prefix="specmodel-diagnostic-") as directory:
        case = make_case(root, directory, args.library)
        artifact = save_spectral_grid(Path(directory) / "prepared_log.npz", case.spectra)
        model = SpecModel(load_spectral_grid(artifact), vmax=case.vmax)
        params = case.params
        for key in ("vsini", "vmacro", "resolving_power", "rv"):
            value = getattr(args, key)
            if value is not None:
                target = params["instrument"] if key == "resolving_power" else params["components"][0]
                if key not in ("resolving_power", "rv"):
                    target = target["broadening"]
                target[key] = value
        # Resolve model-grid-scale features in the display, particularly across
        # the wide TLUSTY blue arm. This is evaluation geometry, not regridding
        # the prepared library or choosing fitting resolution from detector N.
        wave = display_wavelength(model.spectra.wavelength, case.wave)
        stages = {name: np.asarray(jax.jit(lambda m, p, w: getattr(m, name)(p, w))(model, params, wave))
                  for name in ("intrinsic", "broadened", "full")}
        titles = [f"{args.library}: region {region}, segment {i}" for i, region in enumerate(model.spectra.regions)]
        limits = (min(flux.min() for flux in stages.values()) - .05, 1.15)
        fig, _ = plot_spectra(wave, stages, titles=titles,
                             observation=(case.wave, case.oracle["flux_obs"]), flux_limits=limits)
        fig.savefig(output / "stages.png", dpi=150)
        plt.close(fig)

        expected = {name: np.asarray(value) for name, value in frozen_stages(case, params, wave).items()}
        fig, _ = plot_comparison(wave, expected["full"], stages["full"])
        fig.savefig(output / "legacy_comparison.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        line_center = wave[0, np.argmin(stages["intrinsic"][0])]
        half_window = 50. if args.library == "tlusty" else 5.
        choices = {"vsini": (0., 6.3, 15.), "vmacro": (0., 3.1, 10.),
                   "resolving_power": (20000., 70000., 100000.), "rv": (-10., 0., 10.)}
        if args.library == "tlusty":
            choices = {"vsini": (0., 105., 200.), "vmacro": (0., 7., 20.),
                       "resolving_power": (1500., 2500., 5000.), "rv": (-150., 0., 150.)}
        for axis, (key, settings) in zip(axes.flat, choices.items()):
            for value in settings:
                altered = copy.deepcopy(params)
                target = altered["instrument"] if key == "resolving_power" else altered["components"][0]
                if key not in ("resolving_power", "rv"):
                    target = target["broadening"]
                target[key] = value
                axis.plot(wave[0], model(altered, wave)[0], label=str(value))
            axis.set(title=key, xlabel="Wavelength [Å]", ylabel="Flux",
                     xlim=(line_center - half_window, line_center + half_window))
            axis.ticklabel_format(useOffset=False, style="plain", axis="x")
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "parameter_response.png", dpi=150)
        plt.close(fig)
        metrics = {name: {"max_abs": float(np.max(np.abs(stages[name] - expected[name]))),
                          "rms": float(np.sqrt(np.mean((stages[name] - expected[name])**2)))} for name in stages}
        (output / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="../jaxspec")
    parser.add_argument("--output-dir", default="/private/tmp/jaxstar-specmodel-plots")
    parser.add_argument("--library", choices=("coelho", "bosz", "tlusty"), default="coelho")
    for key in ("vsini", "vmacro", "resolving-power", "rv"):
        parser.add_argument(f"--{key}", type=values)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args)


if __name__ == "__main__":
    main()
