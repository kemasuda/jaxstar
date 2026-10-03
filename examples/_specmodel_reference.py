"""Read-only frozen-data support for tests and deterministic diagnostics.

This repository-only helper is not part of jaxstar's public API. It extracts
local atmosphere cells and prepares log grids offline; it never fits data or
regenerates reference outputs. Frozen modules are executed from source bytes
without importing sibling packages or writing their bytecode caches.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import load_bosz, load_coelho, load_tlusty, resample_spectral_grid


SCHEMAS = {
    "coelho": (("teff", "tgrid"), ("logg", "ggrid"), ("feh", "fgrid"), ("alpha", "agrid")),
    "bosz": (("teff", "tgrid"), ("logg", "ggrid"), ("mh", "mgrid"), ("alpha", "agrid"),
             ("carbon", "cgrid"), ("vmic", "vgrid")),
    "tlusty": (("teff", "tgrid"), ("logg", "ggrid"), ("logZ", "zgrid")),
}
POINTS = {"coelho": [5812.5, 4.2, -.15, .12], "bosz": [5812.5, 4.2, -.15, .10, -.10, 1.3],
          "tlusty": [31250., 3.85, -.12]}


def frozen_modules(root):
    namespace = "_jaxstar_readonly_frozen_specmodel"
    package = types.ModuleType(namespace)
    package.__path__ = []
    sys.modules[namespace] = package
    modules = {}
    for name in ("kernels", "utils", "specgrid", "specmodel"):
        path = root / "src/jaxspec" / f"{name}.py"
        module = types.ModuleType(f"{namespace}.{name}")
        module.__file__, module.__package__ = str(path), namespace
        sys.modules[module.__name__] = module
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        modules[name] = module
    return types.SimpleNamespace(**modules)


def verify_reference(root):
    manifest = json.loads((root / "characterization/reference_outputs/manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        digest = hashlib.sha256()
        with (root / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"frozen reference hash mismatch: {relative}")


def make_case(root, directory, name="coelho"):
    """Prepare the same log working samples once, keeping only local axis cells."""
    root, directory = Path(root).resolve(), Path(directory)
    modules = frozen_modules(root)
    point = POINTS[name]
    path = next((root / f"characterization/sample_grid_{name}").glob("*.npz"))
    with np.load(path, allow_pickle=False) as original:
        axes, indices = {}, []
        for (_, key), value in zip(SCHEMAS[name], point):
            axis = original[key]
            upper = int(np.searchsorted(axis, value, side="right"))
            selected = np.array([upper - 1, upper])
            axes[key] = axis[selected]
            indices.append(selected)
        native_wave = original["wavgrid"]
        payload = dict(axes, wavgrid=native_wave,
                       flux=original["flux"][np.ix_(*indices, np.arange(len(native_wave)))])
    extracted = directory / f"{name}_local.npz"
    np.savez(extracted, **payload)
    tag = "tlusty" if name == "tlusty" else "ird"
    with np.load(root / "characterization/reference_outputs/frozen_main.npz", allow_pickle=False) as reference:
        oracle = {key.removeprefix(f"forward_{tag}_"): reference[key].copy()
                  for key in reference.files if key.startswith(f"forward_{tag}_") and "sb2" not in key}
    wave = oracle["wav_obs"]
    count = wave.shape[0]
    loader = {"coelho": load_coelho, "bosz": load_bosz, "tlusty": load_tlusty}[name]
    legacy = loader([extracted] * count, regions=(8,) * count if tag == "ird" else (0,),
                    wavelength_medium="unknown", flux_kind="normalized")
    complete = resample_spectral_grid(legacy, pixels=len(native_wave))
    field = complete.grid.field("flux")
    trimmed = RectilinearGrid(axes={axis: complete.grid.axis(axis) for axis in complete.grid.axis_names},
        fields={"flux": Field(field.values[..., 1:-1], field.dims, payload_dims=field.payload_dims)})
    spectra = replace(complete, grid=trimmed, wavelength=complete.wavelength[:, 1:-1])
    atmosphere = dict(zip((axis for axis, _ in SCHEMAS[name]), point))
    params = {"components": ({"atmosphere": atmosphere,
              "broadening": {"vsini": 105. if tag == "tlusty" else 6.3,
                             "vmacro": 7. if tag == "tlusty" else 3.1, "u1": .5, "u2": .2},
              "rv": np.array([12.4, -7.2][:count])},),
              "instrument": {"resolving_power": np.array([2500., 2400.][:count]) if tag == "tlusty"
                              else np.array([70000., 67200.][:count])}}
    cls = {"coelho": modules.specgrid.SpecGrid, "bosz": modules.specgrid.SpecGridBosz,
           "tlusty": modules.specgrid.SpecGridTlusty}[name]
    old_grid = cls([extracted] * count)
    vmax = 1500. if tag == "tlusty" else 50.
    old = modules.specmodel.SpecModel(old_grid, wave, np.zeros_like(wave), np.ones_like(wave),
                                     np.zeros_like(wave, dtype=bool), vmax=vmax)
    return types.SimpleNamespace(name=name, spectra=spectra, params=params, wave=wave,
                                 old=old, modules=modules, oracle=oracle, vmax=vmax)


def frozen_stages(case, params=None, wave=None):
    params = case.params if params is None else params
    wave = case.wave if wave is None else wave
    component = params["components"][0]
    atmosphere = [component["atmosphere"][name] for name, _ in SCHEMAS[case.name]]
    intrinsic_grid = case.old.sg.values(*atmosphere, case.old.wavgrid)
    count = wave.shape[0]
    broad = {key: jnp.broadcast_to(jnp.asarray(value), (count,)) for key, value in component["broadening"].items()}
    resolution = jnp.broadcast_to(jnp.asarray(params["instrument"]["resolving_power"]), (count,))
    rv = jnp.broadcast_to(jnp.asarray(component["rv"]), (count,))
    function = jax.vmap(case.modules.utils.broaden_and_shift, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    def evaluate(velocity):
        return function(wave, case.old.wavgrid, intrinsic_grid, broad["vsini"], broad["vmacro"],
                        case.modules.utils.get_beta(resolution), velocity, case.old.varr, broad["u1"], broad["u2"])
    return {"intrinsic": jax.vmap(jnp.interp)(wave, case.old.wavgrid, intrinsic_grid),
            "broadened": evaluate(jnp.zeros(count)), "full": evaluate(rv)}


def continuum_for_oracle(wave):
    """Undo the fixed legacy continuum for saved physical-flux comparisons only."""
    count = wave.shape[0]
    return np.array([1.02, .97][:count])[:, None] + np.array([.04, -.03][:count])[:, None] * (
        wave - wave.mean(axis=1)[:, None]) / np.ptp(wave, axis=1)[:, None]
