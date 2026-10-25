"""Compare one exact Coelho node with an intrinsic, normalized Korg spectrum."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COELHO = ROOT / "examples/specfit/coelho/15239-15449_normed.npz"
POINT = {"teff": 6000.0, "logg": 4.0, "feh": 0.0, "alpha_fe": 0.0}
FITS_URL = (
    "https://wwwmpa.mpa-garching.mpg.de/PUBLICATIONS/DATA/SYNTHSTELLIB/"
    "solar_scaled/6000_40_p00p00.ms.fits.gz"
)


def air_to_vacuum_morton(wavelength):
    """Same formula as the repository's Coelho preparer; Angstrom in/out."""
    s2 = (1e4 / wavelength) ** 2
    n = (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2)
         + 0.0001599740894897 / (38.92568793293 - s2))
    return wavelength * n


def load_coelho():
    # Direct indexing retains the example's intrinsic, already-normalized flux.
    # Do not instantiate SpecFit: that also prepares/resamples the model grid.
    with np.load(COELHO, allow_pickle=False) as data:
        indices = []
        shape = []
        for key, value in zip(("tgrid", "ggrid", "fgrid", "agrid"), POINT.values()):
            axis = data[key]
            matches = np.flatnonzero(axis == value)
            if axis.ndim != 1 or matches.size != 1:
                raise ValueError(f"{key}: {value} must be an exact, unique grid node")
            indices.append(int(matches[0]))
            shape.append(axis.size)
        wavelength = data["wavgrid"].copy()
        flux_grid = data["flux"]
        if flux_grid.shape != (*shape, wavelength.size):
            raise ValueError("Coelho flux dimensions do not match its named axes")
        flux = flux_grid[tuple(indices)].copy()
    validate_spectrum(wavelength, flux)

    # This check is specific to the supplied segment. These are the actual
    # published FITS header values, NOT the paper's rounded 3000 + i*0.02.
    native_air = 2990.015161 + np.arange(750901) * 0.01999596
    native_vacuum = air_to_vacuum_morton(native_air)
    selected = native_vacuum[(native_vacuum > 15239) & (native_vacuum < 15449)]
    expected = np.linspace(selected[0], selected[-1], 5000)
    if wavelength.shape != expected.shape or not np.allclose(
        wavelength, expected, rtol=0, atol=1e-9
    ):
        raise ValueError("Wavelength provenance check failed; do not assume air/vacuum")
    return wavelength, flux, indices


def validate_spectrum(wavelength, flux):
    if (wavelength.ndim != 1 or wavelength.size < 2 or flux.shape != wavelength.shape
            or not np.all(np.isfinite(wavelength)) or not np.all(wavelength > 0)
            or not np.all(np.diff(wavelength) > 0) or not np.all(np.isfinite(flux))):
        raise ValueError("Spectrum must have finite flux and increasing positive wavelengths")


def verify_native_fits(path, wavelength, flux):
    """Optional independent check against the published, normalized FITS row."""
    from astropy.io import fits

    with fits.open(path) as hdus:
        header = hdus[0].header
        for key, value in zip(("TEFF", "LOG_G", "FEH", "AFE"), POINT.values()):
            if header[key] != value:
                raise ValueError(f"Native FITS {key} is not the selected grid point")
        native_air = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CD1_1"]
        native_flux = np.asarray(hdus[0].data[0], dtype=float)
    errors = {}
    for medium, native_wav in (("air", native_air),
                               ("vacuum", air_to_vacuum_morton(native_air))):
        selected = native_wav[(native_wav > 15239) & (native_wav < 15449)]
        reconstructed = np.linspace(selected[0], selected[-1], wavelength.size)
        delta = np.interp(wavelength, native_wav, native_flux) - flux
        errors[medium] = {
            "max_wavelength_error_A": float(np.max(np.abs(reconstructed - wavelength))),
            "flux_rms": float(np.sqrt(np.mean(delta**2))),
            "max_flux_error": float(np.max(np.abs(delta))),
        }
    if (errors["vacuum"]["max_wavelength_error_A"] > 1e-9
            or errors["vacuum"]["max_flux_error"] > 1e-7):
        raise ValueError(f"Native FITS does not reproduce the NPZ: {errors}")
    return {"source_url": FITS_URL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reconstruction_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia", help="Julia executable (default: PATH)")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--step", type=float, default=0.01, help="Korg sampling in Angstroms")
    parser.add_argument("--verify-native-fits", type=Path,
                        help="Also reproduce the NPZ from the published 6000_40_p00p00 FITS")
    parser.add_argument("--check-only", action="store_true", help="Check Coelho without Julia")
    args = parser.parse_args()
    wavelength, coelho, indices = load_coelho()
    if not np.isfinite(args.step) or not 0 < args.step <= np.min(np.diff(wavelength)):
        parser.error("--step must be positive and no coarser than the Coelho samples")
    metadata = {
        "parameters": POINT, "coelho_file": str(COELHO.relative_to(ROOT)),
        "coelho_sha256": hashlib.sha256(COELHO.read_bytes()).hexdigest(),
        "coelho_indices": indices, "wavelength_medium": "vacuum",
        "wavelength_bounds_A": wavelength[[0, -1]].tolist(), "pixel_count": wavelength.size,
        "wavelength_check": "Published FITS header + repository Morton conversion + linspace",
        "normalization": "Coelho as stored; Korg flux / cntm",
        "residual_definition": "Korg - Coelho", "interpolation": "linear; no extrapolation",
        "rotational_macroturbulent_instrumental_broadening": "none",
    }
    if args.verify_native_fits:
        metadata["native_fits_check"] = verify_native_fits(
            args.verify_native_fits, wavelength, coelho
        )
    print(f"Exact Coelho node: {POINT}; indices={indices}", flush=True)
    print(f"Verified vacuum wavelengths: {wavelength[0]:.8f}–{wavelength[-1]:.8f} Å; "
          f"{wavelength.size} samples", flush=True)
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        return
    julia = shutil.which(args.julia)
    if julia is None:
        parser.error("Julia was not found; see experiments/korg/README.md for setup")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    request = out / "request.toml"
    request.write_text("\n".join([
        *(f"{key} = {value}" for key, value in POINT.items()),
        f"wavelength_bounds_vacuum_A = {wavelength[[0, -1]].tolist()}",
        f"step_A = {args.step}", "",
    ]))
    subprocess.run([julia, "--startup-file=no", f"--project={HERE}",
                    str(HERE / "synthesize.jl"), str(request), str(out)], check=True)
    native = np.genfromtxt(out / "korg_native.csv", delimiter=",", names=True)
    kw, kf = native["wavelength_vacuum_A"], native["normalized_flux"]
    validate_spectrum(kw, kf)
    if kw[0] > wavelength[0] or kw[-1] < wavelength[-1]:
        raise ValueError("Korg does not bracket the Coelho grid; refusing extrapolation")
    korg = np.interp(wavelength, kw, kf)
    residual = korg - coelho
    metadata["difference_statistics"] = {
        "mean": float(np.mean(residual)), "rms": float(np.sqrt(np.mean(residual**2))),
        "max_abs": float(np.max(np.abs(residual))),
    }
    metadata["python_version"] = sys.version.split()[0]
    metadata["numpy_version"] = np.__version__
    np.savetxt(out / "comparison.csv", np.column_stack((wavelength, coelho, korg, residual)),
               delimiter=",", header="wavelength_vacuum_A,coelho,korg,korg_minus_coelho", comments="")
    (out / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Full interval plus a fixed zoom make individual intrinsic lines readable.
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex="col",
                             height_ratios=(2, 1), layout="constrained")
    for col, bounds in enumerate(((wavelength[0], wavelength[-1]), (15335, 15355))):
        axes[0, col].plot(wavelength, coelho, lw=0.8, label="Coelho")
        axes[0, col].plot(wavelength, korg, lw=0.8, alpha=0.85, label="Korg")
        axes[0, col].legend()
        axes[0, col].set(ylabel="Continuum-normalized flux", xlim=bounds)
        axes[1, col].plot(wavelength, residual, color="tab:purple", lw=0.8)
        axes[1, col].axhline(0, color="0.5", lw=0.6)
        axes[1, col].set(xlabel="Vacuum wavelength [Å]", ylabel="Korg − Coelho")
    fig.suptitle("6000 K · log g = 4.0 · [Fe/H] = 0.0 · [α/Fe] = 0.0\n"
                 "Intrinsic spectra · no rotation, macroturbulence or instrumental broadening")
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(metadata["difference_statistics"], indent=2))
    print(f"Comparison and provenance saved in {out}")


if __name__ == "__main__":
    main()
