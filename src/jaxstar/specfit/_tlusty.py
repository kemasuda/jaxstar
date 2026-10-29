"""TLUSTY OB2025 file planning and fixed-vt preparation (no downloads)."""

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

from ._data import _validate_wavelength


# Hubeny et al. (2025), Section 2: both families use these eight nominal Zs.
# The distributed OSTAR files spell the 0.032-solar grid 'z0033'. Record this
# explicit naming alias rather than interpreting filename digits as abundance.
_Z_CODES = {"001": .01, "002": .02, "0032": .032, "010": .1,
            "020": .2, "050": .5, "100": 1., "200": 2.}
_SWITCH_TEFF = 30000.


def _metallicity_code(logz):
    matches = [code for code, z in _Z_CODES.items()
               if np.isclose(logz, np.log10(z), rtol=0, atol=1e-10)]
    if len(matches) != 1:
        raise ValueError("TLUSTY OB2025 logZ must be log10 of a native Z/Z_sun: "
                         "0.01, 0.02, 0.032, 0.1, 0.2, 0.5, 1 or 2")
    return matches[0]


def _paths(root, node):
    from .libraries import _integer

    teff, logg, logz = node
    family = "BSTAR" if teff < _SWITCH_TEFF else "OSTAR"
    code = _metallicity_code(logz)
    if family == "OSTAR" and code == "0032":
        code = "0033"
    stem = f"z{code}t{_integer(teff)}g{_integer(logg * 100):03d}v2"
    directory = root / f"{family}_z{code}v2"
    return tuple(directory / f"{stem}.{kind}.gz" for kind in ("spec", "cont"))


def discover_axes(root):
    """Infer native axes from regular v2 filenames, never v2old or v5/v10."""
    nodes = []
    for family in ("BSTAR", "OSTAR"):
        for canonical, z in _Z_CODES.items():
            code = "0033" if family == "OSTAR" and canonical == "0032" else canonical
            directory = root / f"{family}_z{code}v2"
            pattern = re.compile(rf"z{code}t(\d+)g(\d{{3}})v2\.spec\.gz")
            for path in directory.glob("*.spec.gz"):
                match = pattern.fullmatch(path.name)
                if match is None:
                    continue
                teff, logg = float(match[1]), float(match[2]) / 100
                if (family == "BSTAR") != (teff < _SWITCH_TEFF) or logg <= 2.9:
                    continue
                nodes.append((teff, logg, np.log10(z)))
    if not nodes:
        raise ValueError("no selected TLUSTY OB2025 vt=2 spectra found")
    return {name: np.unique([node[i] for node in nodes])
            for i, name in enumerate(("teff", "logg", "logZ"))}


def _lowest_logg_pair(root, node):
    """Find the lower native boundary at fixed family, Teff, Z and vt.

    Include incomplete pairs in discovery so a missing continuum or empty
    file at the boundary cannot silently move the boundary to higher logg.
    """
    spectrum, _ = _paths(root, node)
    prefix = spectrum.name.split("g")[0]
    pattern = re.compile(rf"{re.escape(prefix)}g(\d{{3}})v2\.(?:spec|cont)\.gz")
    gravities = []
    for path in spectrum.parent.glob(f"{prefix}g*v2.*.gz"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            gravities.append(float(match[1]) / 100)
    if not gravities:
        return None
    source = (node[0], min(gravities), node[2])
    return source, _paths(root, source)


def _missing_files(pair):
    return [str(p) for p in pair if not p.is_file() or p.stat().st_size == 0]


def source_plan(data_dir, axes, *, missing_nodes="error"):
    """Check every requested pair using only file metadata, before flux I/O."""
    if missing_nodes not in {"error", "lowest_logg"}:
        raise ValueError("TLUSTY OB2025 missing_nodes must be 'error' or 'lowest_logg'")
    root = Path(data_dir).expanduser()
    paths, sources, unresolved, replacements, boundaries = {}, [], [], [], {}

    def coordinates(node):
        return {name: float(value) for name, value in zip(axes, node)}

    for node in product(*axes.values()):
        pair = _paths(root, node)
        paths[node] = pair
        absent = _missing_files(pair)
        if not absent:
            sources.extend(str(p) for p in pair)
            continue
        entry = dict(node=coordinates(node), missing_path=absent[0], missing_paths=absent)
        reason = "missing or empty spectrum/continuum files: " + ", ".join(absent)
        # Replace an entirely absent lower-gravity node only. Partial pairs,
        # empty files, interior holes and upper-edge gaps remain errors.
        if missing_nodes == "lowest_logg" and not any(p.exists() or p.is_symlink() for p in pair):
            key = (node[0], node[2])
            if key not in boundaries:
                boundaries[key] = _lowest_logg_pair(root, node)
            boundary = boundaries[key]
            if boundary is not None and node[1] < boundary[0][1]:
                source, replacement = boundary
                missing_source = _missing_files(replacement)
                if not missing_source:
                    paths[node] = replacement
                    sources.extend(str(p) for p in replacement)
                    replacements.append({**entry, "method": "copy_lowest_logg", "sources": [dict(
                        node=coordinates(source), path=str(replacement[0]),
                        continuum_path=str(replacement[1]), weight=1.)]})
                    continue
                reason = f"lowest logg={source[1]:g} source pair is missing or empty"
                entry["missing_neighbors"] = missing_source
            else:
                reason += "; lowest_logg only fills nodes below the available logg range at fixed Teff and Z"
        unresolved.append({**entry, "reason": reason})
    report = dict(ready=not unresolved, requested_nodes=len(paths),
                  missing_nodes=missing_nodes, sources=list(dict.fromkeys(sources)),
                  replacements=replacements, unresolved=unresolved)
    return paths, report


def _read_continuum(path):
    data = np.loadtxt(path, ndmin=2, usecols=(0, 1))
    if data.shape[0] == 0:
        raise ValueError(f"{path}: continuum contains no samples")
    data = data[np.argsort(data[:, 0], kind="stable")]
    repeated = np.diff(data[:, 0]) == 0
    if np.any(repeated & (np.diff(data[:, 1]) != 0)):
        raise ValueError(f"{path}: conflicting continuum flux at a repeated wavelength")
    data = data[np.r_[True, ~repeated]]
    _validate_wavelength(data[:, 0], ndim=1)
    if not np.all(np.isfinite(data[:, 1])) or np.any(data[:, 1] <= 0):
        raise ValueError(f"{path}: continuum flux must be finite and positive")
    return data


def _vacuum_to_air(wavelength):
    from .libraries import _air_to_vacuum

    # Invert the same Morton relation used by the other library preparers.
    # The caller restricts this conversion to vacuum wavelengths >=2000 A.
    air = np.array(wavelength, dtype=np.float64, copy=True)
    for _ in range(5):
        air *= wavelength / _air_to_vacuum(air)
    return air


def prepare_ob2025(root, *, axes, wavelength_ranges, pixels, regions, flux_kind,
                   wavelength_medium, sampling, velocity_step, missing_nodes):
    from .libraries import (_preparation_axes, _prepare_nodes, _region_bounds,
                            _require_source_plan)
    from .sampling import _sampling_mode

    _sampling_mode(sampling, pixels, velocity_step)
    bounds = _region_bounds(wavelength_ranges)
    medium = "vacuum" if wavelength_medium == "unknown" else wavelength_medium
    if medium not in {"vacuum", "air"}:
        raise ValueError("TLUSTY OB2025 wavelength_medium must be 'air' or 'vacuum'")
    if medium == "air" and np.any(bounds[:, 0] < 2000):
        raise ValueError("TLUSTY air regions must lie above 2000 Angstrom; use vacuum for UV")
    axes = _preparation_axes("tlusty", discover_axes(root) if axes is None else axes)
    paths, report = source_plan(root, axes, missing_nodes=missing_nodes)
    _require_source_plan(report)

    def reader(node, bounds_only=False):
        spectrum, continuum = paths[node]
        data = np.loadtxt(spectrum, ndmin=2, usecols=(0,) if bounds_only else (0, 1))
        wav, flux = data[:, 0], None if bounds_only else data[:, 1]
        _validate_wavelength(wav, ndim=1)
        if not bounds_only and flux_kind == "normalized":
            cont = _read_continuum(continuum)
            denominator = np.interp(wav, cont[:, 0], cont[:, 1], left=np.nan, right=np.nan)
            flux = flux / denominator
        if medium == "air":
            keep = wav >= 2000
            wav = _vacuum_to_air(wav[keep])
            if flux is not None:
                flux = flux[keep]
        return wav, flux

    spectra = _prepare_nodes("tlusty", axes, reader, bounds, pixels, regions,
                             medium, flux_kind, tuple(report["sources"]),
                             sampling=sampling, velocity_step=velocity_step)
    provenance = dict(
        release="TLUSTY OB2025", reference_doi="10.3847/1538-3881/adb1bf",
        synthesis_vmic_kms=2., atmosphere_vmic_kms={"BSTAR": 2., "OSTAR": 10.},
        family_selection={"BSTAR": "teff < 30000 K", "OSTAR": "teff >= 30000 K"},
        native_wavelength_medium="vacuum", native_flux="Eddington H_lambda",
        native_resolving_power=100000,
        metallicity_z_over_sun={**_Z_CODES, "0033": .032},
        metallicity_note="OSTAR z0033 denotes the nominal 0.032-solar grid listed in the paper",
        missing_nodes=missing_nodes, replacements=report["replacements"],
    )
    return replace(spectra, provenance_json=json.dumps(provenance))
