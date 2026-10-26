"""Optional Korg synthesis backend for common spectral grids.

Julia is imported only when preparing a grid. Stored grids use the ordinary
Python/JAX loader. The Korg 1.3.1 optical configuration is deliberately explicit;
new abundance axes and other line lists require additional backend support.
"""

from collections.abc import Mapping
from functools import lru_cache
import hashlib
from importlib.metadata import version
import inspect
from itertools import product
import json
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import jax
import numpy as np

from .libraries import _assemble, _region_bounds
from .numpyro_model import empirical_vmic
from .sampling import _sampling_mode, _target_wavelengths


KORG_AXES = ("teff", "logg", "feh", "alpha")
KORG_VERSION = "1.3.1"


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".korg-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_korg_axes(axes):
    """Validate currently supported axes, independently of legacy NPZ schemas."""
    if not isinstance(axes, Mapping) or set(axes) != set(KORG_AXES):
        raise ValueError(f"Korg axes must contain exactly {KORG_AXES}; "
                         "vmic is empirical, and extra abundance axes are not yet supported")
    result = {}
    for name in KORG_AXES:
        values = np.asarray(axes[name])
        if (values.ndim != 1 or not values.size or values.dtype.kind not in "iuf"
                or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0)):
            raise ValueError(f"{name} must be a nonempty finite increasing numeric axis")
        result[name] = values.astype(np.float64)
    if np.any(result["teff"] <= 0):
        raise ValueError("teff must be positive kelvin")
    return result


def validate_korg_options(*, normalized=True, synthesis_step=.0025, line_buffer=10.):
    """Cheap validation, also used by the recipe's Julia-free dry run."""
    if normalized is not True:
        raise ValueError("Korg currently supports only normalized=true")
    for name, value in (("synthesis_step", synthesis_step), ("line_buffer", line_buffer)):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive finite Angstrom")


def _merged_intervals(bounds):
    merged = []
    for lo, hi in sorted(map(tuple, bounds)):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([float(lo), float(hi)])
    return merged


class _Runtime:
    def __init__(self):
        try:
            import juliacall
        except ImportError as exc:
            raise ImportError("Korg generation requires the optional dependency: "
                              "pip install 'jaxstar[korg]'; see scripts/specfit/README.md "
                              "for installing Korg.jl") from exc
        except Exception as exc:
            raise RuntimeError(f"JuliaCall initialization failed: {exc}") from exc
        self.jl = juliacall.newmodule("JaxstarKorg")
        try:
            self.jl.seval("using Korg, Pkg")
        except Exception as exc:
            raise RuntimeError("Cannot load Korg.jl in JuliaCall's environment. "
                               "Install Korg 1.3.1 as described in scripts/specfit/README.md. "
                               f"Julia reported: {exc}") from exc
        self.korg = self.jl.Korg
        installed = str(self.jl.seval("string(pkgversion(Korg))"))
        if installed != KORG_VERSION:
            raise RuntimeError(f"This backend requires Korg {KORG_VERSION}; found {installed}")
        self.solar = self.korg.bergemann_2025_solar_abundances
        self.alpha = np.array(self.korg.default_alpha_elements, dtype=int)
        self.lines = None

    def to_vacuum(self, wavelength, medium):
        return (np.array(self.jl.broadcast(self.korg.air_to_vacuum, wavelength))
                if medium == "air" else np.array(wavelength, copy=True))

    def atmosphere(self, point):
        # Korg expects [non-alpha metals/H] and [alpha/H], not [alpha/Fe].
        abundances = self.korg.format_A_X(
            point["feh"], point["feh"] + point["alpha"], solar_abundances=self.solar)
        shifts = np.array(abundances) - np.array(self.solar)
        if (not np.isclose(shifts[25], point["feh"], rtol=0, atol=1e-12)
                or not np.allclose(shifts[self.alpha - 1] - shifts[25], point["alpha"],
                                   rtol=0, atol=1e-12)):
            raise ValueError("Korg abundance mapping does not preserve [Fe/H] and [alpha/Fe]")
        atm = self.korg.interpolate_marcs(point["teff"], point["logg"], abundances,
                                          clamp_abundances=False,
                                          perturb_at_grid_values=False)
        return atm, abundances

    def load_lines(self):
        if self.lines is not None:
            return
        lines = self.korg.get_GES_linelist(include_molecules=True)
        root = Path(str(self.jl.pathof(self.korg))).parents[1]
        artifact = Path(str(self.jl.seval(
            'Pkg.Artifacts.artifact_path(Pkg.Artifacts.artifact_hash('
            '"Heiter_2021_GES_linelist", '
            'joinpath(dirname(dirname(pathof(Korg))), "Artifacts.toml")))')))
        self.line_path = artifact / "Heiter_et_al_2021_2022_06_17/Heiter_et_al_2021.h5"
        import h5py
        with h5py.File(self.line_path, "r") as data:
            wl = data["wl"]
            extrema = [(np.min(block), np.max(block))
                       for start in range(0, len(wl), 1000000)
                       for block in [wl[start:start + 1000000]]]
        self.coverage = self.to_vacuum(
            np.array([min(a for a, _ in extrema), max(b for _, b in extrema)]), "air")
        project = Path(str(self.jl.seval("Base.active_project()")))
        self.provenance = {
            "versions": {"korg": KORG_VERSION, "julia": str(self.jl.seval("string(VERSION)")),
                         "juliacall": version("juliacall"), "python": platform.python_version(),
                         "numpy": np.__version__, "jax": jax.__version__},
            "julia_threads": int(self.jl.seval("Threads.nthreads()")),
            "julia_environment": {name: _digest_file(project.parent / name)
                                  for name in ("Project.toml", "Manifest.toml")
                                  if (project.parent / name).is_file()},
            "korg_sources": {str(p.relative_to(root)): _digest_file(p)
                             for p in sorted((root / "src").rglob("*.jl"))},
            "korg_artifacts_toml_sha256": _digest_file(root / "Artifacts.toml"),
            "linelist": {"name": "GES", "include_molecules": True,
                         "artifact_tree_hash": artifact.name,
                         "sha256": _digest_file(self.line_path),
                         "line_count": int(self.jl.length(lines)),
                         "vacuum_coverage_A": self.coverage.tolist()},
            "solar_abundances": "bergemann_2025_solar_abundances",
            "solar_A_X": np.array(self.solar).tolist(),
            "alpha_atomic_numbers": self.alpha.tolist(),
            "atmosphere": "Korg.interpolate_marcs(Teff, logg, A_X)",
            "clamp_abundances": False, "perturb_at_grid_values": False,
            "geometry": "Korg default: spherical for logg < 3.5, planar otherwise",
            "hydrogen_lines": True, "use_MHD_for_hydrogen_lines": True,
            "normalization": "flux / cntm", "additional_broadening": "none",
            "air_to_vacuum": "Korg.air_to_vacuum",
        }
        # Publish initialized state only after coverage/provenance also succeeded.
        self.lines = lines

    def windows(self, intervals, step):
        windows = self.jl.seval("Tuple{Float64,Float64,Float64}[]")
        for lo, hi in intervals:
            self.jl.push_b(windows, (lo, hi, step))
        return windows

    def synthesize(self, atmosphere, windows, vmic, line_buffer):
        atm, abundances = atmosphere
        sol = self.korg.synthesize(atm, self.lines, abundances, windows, vmic=vmic,
                                   return_cntm=True, line_buffer=line_buffer,
                                   hydrogen_lines=True, use_MHD_for_hydrogen_lines=True)
        return np.array(sol.wavelengths), np.array(sol.flux), np.array(sol.cntm)


@lru_cache(maxsize=1)
def _get_runtime():
    # Share the initialized Julia module and parsed line list across API calls.
    return _Runtime()


def _resample(solution, target, intervals):
    wavelength, flux, continuum = map(np.asarray, solution)
    if (wavelength.ndim != 1 or flux.shape != wavelength.shape
            or continuum.shape != wavelength.shape or len(wavelength) < 2
            or not np.all(np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0)
            or not np.all(np.isfinite(flux)) or not np.all(np.isfinite(continuum))
            or np.any(continuum <= 0)):
        raise ValueError("Korg returned invalid wavelengths, flux or continuum")
    normalized = flux / continuum
    rows = []
    for row in target:
        lo, hi = next((lo, hi) for lo, hi in intervals if lo <= row[0] and hi >= row[-1])
        keep = (wavelength >= lo) & (wavelength <= hi)
        w, f = wavelength[keep], normalized[keep]
        if len(w) < 2 or w[0] > row[0] or w[-1] < row[-1]:
            raise ValueError("Korg does not bracket the output wavelengths; no extrapolation")
        rows.append(np.interp(row, w, f))
    result = np.asarray(rows)
    if not np.all(np.isfinite(result)):
        raise ValueError("Korg normalized flux is nonfinite")
    return result


def _node_digest(flux):
    return hashlib.sha256(np.asarray(flux, dtype="<f8").tobytes()).hexdigest()


def _cached_node(path, signature, coordinates, shape, vmic):
    try:
        with np.load(path, allow_pickle=False) as data:
            flux = data["flux"]
            if (str(data["signature"].item()) != signature
                    or not np.array_equal(data["coordinates"], coordinates)
                    or float(data["vmic"]) != vmic or flux.shape != shape
                    or flux.dtype != np.dtype("float64") or not np.all(np.isfinite(flux))
                    or str(data["sha256"].item()) != _node_digest(flux)):
                raise ValueError("node content or checksum mismatch")
            return flux
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"Invalid Korg cache node {path}: {exc}") from exc


def prepare_korg(*, axes, wavelength_ranges, pixels=None, regions=None,
                 wavelength_medium, normalized=True, sampling="log", velocity_step=None,
                 synthesis_step=.0025, line_buffer=10., cache_dir=None,
                 progress=None, return_report=False):
    """Synthesize an intrinsic normalized optical grid through Python/JuliaCall.

    Axes are teff (K), logg (cgs), feh ([Fe/H]) and alpha ([alpha/Fe]).
    Each node uses empirical_vmic(teff, logg, feh) in km/s, without a vmic axis.
    The existing relation is evaluated as-is, including outside its documented
    validity range; those nodes are recorded in the report. Nonpositive/nonfinite
    vmic and MARCS interpolation failures are errors, never clipped or replaced.

    Bounds and output wavelengths are Angstrom in the explicitly declared
    air/vacuum medium. Native synthesis uses uniform vacuum synthesis_step,
    then linear interpolation onto the output grid; no convolution is applied.
    sampling is 'log' or 'linear'; specify pixels or log velocity_step (km/s).
    JAX x64 must be enabled before calling, as for other float64 spectral work.

    Requires optional juliacall and Korg 1.3.1 with GES (molecules included),
    MARCS, Bergemann 2025 solar abundances and Korg's standard alpha elements.
    See scripts/specfit/README.md for setup. All atmospheres are checked before
    synthesis. First use may download Julia/Korg data through their installers.

    cache_dir enables atomic per-node checkpoints, retained after failure.
    Only identical axes, wavelengths, physics, source hashes and versions reuse
    a cache namespace. Corrupt nodes raise an error. progress, if supplied,
    receives dictionaries with stage/done/total and node timing information.
    Returns the same spectral carrier as prepare_bosz; return_report=True adds
    a JSON-serializable provenance/timing report as a second return value.
    """
    axes = validate_korg_axes(axes)
    validate_korg_options(normalized=normalized, synthesis_step=synthesis_step, line_buffer=line_buffer)
    bounds = _region_bounds(wavelength_ranges)
    if wavelength_medium not in {"air", "vacuum"}:
        raise ValueError("Korg wavelength_medium must be explicitly 'air' or 'vacuum'")
    if sampling not in {"log", "linear"}:
        raise ValueError("Korg sampling must be 'log' or 'linear'")
    _sampling_mode(sampling, pixels, velocity_step)
    if not jax.config.x64_enabled:
        raise ValueError("prepare_korg requires jax.config.update('jax_enable_x64', True)")
    wavelength = _target_wavelengths(bounds, sampling, pixels, velocity_step)
    regions = list(range(len(bounds))) if regions is None else list(regions)
    if (len(regions) != len(bounds)
            or any(type(r) not in (str, int) for r in regions)):
        raise ValueError("regions must supply one string/integer label per wavelength range")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    def notify(stage, **info):
        if progress is not None:
            progress({"stage": stage, **info})

    points = [dict(zip(KORG_AXES, map(float, values))) for values in product(*axes.values())]
    vmics = [float(empirical_vmic(p["teff"], p["logg"], p["feh"])) for p in points]
    for point, vmic in zip(points, vmics):
        if not np.isfinite(vmic) or vmic <= 0:
            raise ValueError(f"Invalid empirical vmic={vmic} km/s at {point}")
    notify("initializing", total=len(points))
    runtime = _get_runtime()
    target = runtime.to_vacuum(wavelength, wavelength_medium)
    if synthesis_step > np.min(np.diff(target, axis=1)):
        raise ValueError("synthesis_step must be no coarser than the output sampling in vacuum")
    intervals = _merged_intervals(target[:, [0, -1]] + [-synthesis_step, synthesis_step])
    for index, point in enumerate(points):
        try:
            runtime.atmosphere(point)
        except Exception as exc:
            raise ValueError(f"MARCS interpolation failed at {point}: {exc}") from exc
        notify("preflight", done=index + 1, total=len(points), point=point)
    notify("loading_linelist", total=len(points))
    runtime.load_lines()
    if any(lo - line_buffer < runtime.coverage[0] or hi + line_buffer > runtime.coverage[1]
           for lo, hi in intervals):
        raise ValueError("Requested wavelengths plus synthesis padding and line_buffer exceed "
                         f"GES vacuum coverage {runtime.coverage.tolist()} Angstrom")
    windows = runtime.windows(intervals, float(synthesis_step))
    contract = {
        "cache_format": 1, "library": "korg", "axes": {k: v.tolist() for k, v in axes.items()},
        "regions": regions, "wavelength_medium": wavelength_medium,
        "wavelength_sha256": _node_digest(wavelength), "wavelength_shape": list(wavelength.shape),
        "synthesis_step_A": float(synthesis_step), "line_buffer_A": float(line_buffer),
        "vacuum_intervals_A": intervals, "vmic_km_s": vmics,
        "empirical_vmic_source": inspect.getsource(empirical_vmic),
        "implementation_sha256": _digest_file(__file__),
        "runtime": runtime.provenance,
    }
    signature = hashlib.sha256(_json(contract).encode()).hexdigest()
    cache = None if cache_dir is None else Path(cache_dir).expanduser().resolve() / signature
    if cache is not None:
        _atomic_write(cache / "contract.json", lambda stream: stream.write(_json(contract).encode()))
    flux = np.empty(tuple(len(a) for a in axes.values()) + wavelength.shape, dtype=np.float64)
    report = {"signature": signature, "contract": contract,
              "cache_path": None if cache is None else str(cache), "nodes": [],
              "vmic_outside_documented_range": [p for p in points if not (
                  4000 < p["teff"] < 7000 and 0 < p["logg"] < 5 and -4.5 < p["feh"] < 1)]}
    flat = flux.reshape((len(points),) + wavelength.shape)
    for index, (point, vmic) in enumerate(zip(points, vmics)):
        start = perf_counter()
        path = None if cache is None else cache / f"node_{index:07d}.npz"
        cached = path is not None and path.exists()
        try:
            if cached:
                current = _cached_node(path, signature, list(point.values()), wavelength.shape, vmic)
            else:
                current = _resample(runtime.synthesize(runtime.atmosphere(point), windows,
                                                       vmic, float(line_buffer)), target, intervals)
                if path is not None:
                    _atomic_write(path, lambda stream: np.savez(
                        stream, flux=current, coordinates=list(point.values()), vmic=vmic,
                        signature=signature, sha256=_node_digest(current)))
            flat[index] = current
        except Exception as exc:
            raise ValueError(f"Korg node failed at {point}: {exc}") from exc
        record = dict(point=point, vmic_km_s=vmic, cached=cached, seconds=perf_counter() - start)
        report["nodes"].append(record)
        notify("node", done=index + 1, total=len(points), **record)
    spectra = _assemble("korg", axes, wavelength, flux, regions, wavelength_medium,
                        "normalized", (str(runtime.line_path),))
    return (spectra, report) if return_report else spectra
