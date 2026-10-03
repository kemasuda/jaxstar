"""Reusable IRD loading/setup groundwork, using existing frozen local inputs.

Only order 8 has a supplied matching prepared Coelho grid. Order 9 from the
legacy notebook needs its own grid before the full two-order port is possible.
No CCF, masks learned from residuals, GP, optimization or downloads are added.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import numpyro.distributions as dist
import pandas as pd

from jaxstar.specfit import (Observation, SpecModel, load_coelho, resample_spectral_grid,
    save_spectral_grid, load_spectral_grid, chebyshev_basis, model_single)


def load_ird_order8(reference_root, artifact_path, *, rv_initial, stride=4):
    """Convert the supplied order-8 grid once, then use the common loader.

    rv_initial is an explicitly supplied approximate effective RV in km/s.
    The illustrative RV prior is +/-5 km/s; this function does not find it.
    Masks retain the CSV exclusion convention and additionally exclude stored
    nonfinite flux/errors. Existing mask columns remain observation data.
    Legacy NPZ wavelength medium is unknown; no medium conversion is guessed.
    """
    if not isinstance(stride, int) or stride < 1 or not np.isfinite(rv_initial):
        raise ValueError("stride must be positive integer and rv_initial finite")
    root, artifact = Path(reference_root), Path(artifact_path)
    if not artifact.exists():
        path = root / "characterization/sample_grid_coelho/15239-15449_normed.npz"
        legacy = load_coelho(path, regions=(8,), flux_kind="normalized", wavelength_medium="unknown")
        prepared = resample_spectral_grid(legacy, velocity_step=1.)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        save_spectral_grid(artifact, prepared)
    model = SpecModel(load_spectral_grid(artifact), vmax=50.)
    if model.spectra.regions != (8,):
        raise ValueError("this setup requires the matching single order-8 artifact")
    frame = pd.read_csv(root / "characterization/sample_data_ird.csv")
    frame = frame.loc[frame.order == 8].iloc[32:2016:stride]
    wave = np.asarray(frame.lam) * 10.  # frozen notebook: nm -> Angstrom
    flux, error = np.asarray(frame.normed_flux), np.asarray(frame.flux_error)
    mask = np.asarray(frame.all_mask, dtype=bool) | ~np.isfinite(flux) | ~np.isfinite(error) | (error <= 0)
    obs = Observation(wave[None, :], flux[None, :], error[None, :], mask[None, :],
                      region=(8,), order=(8,), exposure="IRDA00042313_H")
    atmosphere = {name: dist.Uniform(float(model.spectra.grid.axis(name)[0]),
                                    float(model.spectra.grid.axis(name)[-1]))
                  for name in model.spectra.grid.axis_names}
    priors = dict(atmosphere=atmosphere,
        broadening={"vsini": dist.Uniform(0., 20.), "vmacro": dist.Uniform(0., 10.),
                    "q1": dist.Uniform(0., 1.), "q2": dist.Uniform(0., 1.)},
        rv=dist.Uniform(rv_initial-5., rv_initial+5.), resolving_power=70000.,
        sigma_constant=.1, sigma_continuum=.03, jitter=dist.HalfNormal(.01),
        degree=4, basis=chebyshev_basis(obs.wavelength))
    initial = {"teff": 5800., "logg": 4.3, "feh": -.15, "alpha": .1,
               "vsini": 5., "vmacro": 3., "q1": .36, "q2": .3,
               "rv": rv_initial, "jitter": .005}
    return SimpleNamespace(observation=obs, specmodel=model, priors=priors, initial=initial)


def main():
    from numpyro.infer.util import log_density
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="../jaxspec")
    parser.add_argument("--artifact", default="benchmark-results/ird-order8-log.npz")
    parser.add_argument("--rv-initial", type=float, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    case = load_ird_order8(args.reference_root, args.artifact, rv_initial=args.rv_initial)
    lp, _ = log_density(model_single, (case.observation, case.specmodel), case.priors, case.initial)
    assert np.isfinite(lp)
    print(f"Order 8 setup: {case.observation.shape}, usable={np.sum(case.observation.valid)}, log density={lp}")
    print("No inference performed; pass the same model_single + inputs to SVI/NUTS.")


if __name__ == "__main__":
    main()
