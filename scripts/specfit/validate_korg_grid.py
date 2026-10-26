"""Explicit, optional real-Korg validation of a small prepared grid.

Runs direct syntheses at cell centers and at half the native wavelength step.
Use only with a small recipe: the number of centers grows with the grid size.
"""

import argparse
from itertools import product
from pathlib import Path

import jax
import numpy as np

from prepare_grid import read_recipe, write_json


def metrics(delta):
    return dict(rms=float(np.sqrt(np.mean(delta**2))), max_abs=float(np.max(np.abs(delta))))


def validate(bundle, output):
    from jaxstar.specfit import Observation, SpecFit, load_spectral_grid, prepare_korg

    if output.exists():
        raise FileExistsError(f"{output} already exists")
    jax.config.update("jax_enable_x64", True)
    recipe, _, settings = read_recipe(bundle / "recipe.json")
    if recipe["library"] != "korg":
        raise ValueError("Expected a Korg bundle")
    spectra = load_spectral_grid(bundle / "grid.npz")
    names = spectra.grid.axis_names
    axes = [np.asarray(spectra.grid.axis(name)) for name in names]
    flux = np.asarray(spectra.grid.field("flux").values)
    node_error = 0.
    for index in np.ndindex(*(len(a) for a in axes)):
        point = {name: float(a[i]) for name, a, i in zip(names, axes, index)}
        delta = np.asarray(spectra.grid.interpolate(point)["flux"]) - flux[index]
        node_error = max(node_error, float(np.max(np.abs(delta))))
    if node_error > 1e-12:
        raise ValueError(f"Exact-node interpolation error: {node_error}")

    centers = [(a[:-1] + a[1:]) / 2 if len(a) > 1 else a for a in axes]
    points, direct, interpolated = [], [], []
    for values in product(*centers):
        point = dict(zip(names, map(float, values)))
        print(f"Direct cell-center synthesis: {point}", flush=True)
        reference = prepare_korg(**{**settings, "axes": {k: [v] for k, v in point.items()}})
        points.append(point)
        direct.append(np.asarray(reference.grid.field("flux").values).reshape(spectra.wavelength.shape))
        interpolated.append(np.asarray(spectra.grid.interpolate(point)["flux"]))
    direct, interpolated = np.asarray(direct), np.asarray(interpolated)
    fine = prepare_korg(**{**settings, "axes": {k: [v] for k, v in points[0].items()},
                           "synthesis_step": settings["synthesis_step"] / 2})
    finer = np.asarray(fine.grid.field("flux").values).reshape(spectra.wavelength.shape)
    # Also exercise the actual Julia abundance/MARCS path away from alpha=0.
    alpha_point = {**points[0], "alpha": .2}
    alpha = prepare_korg(**{**settings, "axes": {k: [v] for k, v in alpha_point.items()}})
    if not np.all(np.isfinite(alpha.grid.field("flux").values)):
        raise ValueError("Nonfinite spectrum at [alpha/Fe]=0.2")

    bounds = np.asarray(settings["wavelength_ranges"])
    inset = np.minimum(recipe["wav_margin"], .1 * np.diff(bounds, axis=1)[:, 0])
    if np.any(inset <= 0):
        raise ValueError("Validation requires a positive wavelength margin")
    obs_wave = np.array([np.geomspace(lo + d, hi - d, 64) for (lo, hi), d in zip(bounds, inset)])
    obs = Observation(obs_wave, np.ones_like(obs_wave), np.full_like(obs_wave, .01), region=spectra.regions)
    fit = SpecFit.from_grid(bundle / "grid.npz", obs, wav_margin=float(inset.min() / 2), vmax=50.)
    assert fit.specmodel.spectra.library == "korg"
    np.testing.assert_array_equal(fit.specmodel.spectra.wavelength, spectra.wavelength)
    report = dict(node_max_abs_error=node_error, specfit_from_grid=True,
                  difference="grid interpolation minus direct Korg; intrinsic normalized flux",
                  cell_centers=[dict(point=p, **metrics(i - d))
                                for p, i, d in zip(points, interpolated, direct)],
                  all_cell_centers=metrics(interpolated - direct),
                  sampling_check=dict(point=points[0], fine_step_A=settings["synthesis_step"] / 2,
                                      **metrics(direct[0] - finer)),
                  nonzero_alpha_synthesis=alpha_point,
                  scope="Checks this recipe only; no parameter-inference accuracy claim")
    output.mkdir(parents=True)
    write_json(output / "validation.json", report)
    np.savez(output / "validation.npz", wavelength=np.asarray(spectra.wavelength),
             direct=direct, interpolated=interpolated, finer_at_first_center=finer,
             coordinates=np.array([list(p.values()) for p in points]), axis_names=names)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    worst = int(np.argmax(np.max(np.abs(interpolated - direct), axis=(1, 2))))
    fig, plots = plt.subplots(2, len(spectra.regions), squeeze=False, sharex="col",
                              figsize=(7 * len(spectra.regions), 6), height_ratios=[2, 1],
                              layout="constrained")
    for r, label in enumerate(spectra.regions):
        wave = np.asarray(spectra.wavelength[r])
        plots[0, r].plot(wave, direct[worst, r], lw=.8, label="Direct Korg")
        plots[0, r].plot(wave, interpolated[worst, r], lw=.8, label="Grid interpolation")
        plots[0, r].set(title=f"Region {label}", ylabel="Normalized flux")
        plots[0, r].legend()
        plots[1, r].plot(wave, interpolated[worst, r] - direct[worst, r], lw=.8)
        plots[1, r].axhline(0, color="grey", lw=.5)
        plots[1, r].set(xlabel=f"{spectra.wavelength_medium} wavelength [Å]", ylabel="Difference")
    fig.suptitle(f"GES / MARCS / empirical vmic; worst cell: {points[worst]}")
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    validate(args.bundle.resolve(), args.output_dir.resolve())
