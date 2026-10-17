"""Build a photometric track NPZ from cached MIST v1.2 or v2.5 EEP archives.

Run ``python -m jaxstar.mistfit.mistgrid.create_tracks --help`` for the CLI.
By default retain all native masses and EEPs, with explicit missing cells.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tarfile
import tempfile

import numpy as np

from . import _tracks_v12 as v12
from . import _tracks_v25 as v25
from .cache import fetch_archive
from .sources import bolometric_correction_archive, track_archives, track_metallicities


__all__ = ["create_track_grid"]
_LOG = logging.getLogger(__name__)
_SCALARS = ("logage", "age_gyr", "logTeff", "logg", "logL", "logR",
            "star_mass", "bc_metallicity", "cmd_surface_feh")


def _bounds(value, name, *, positive=True):
    bounds = np.asarray(value, dtype=float)
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1] or (positive and bounds[0] <= 0)):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must contain two {qualifier}increasing finite bounds")
    return tuple(bounds.tolist())


def _file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size}


def _read_archive(path, expected_feh, rotation, afe, mass_range, eep_range, bc_axes, bc_fields,
                  allow_missing, release=v12, extra_columns=()):
    tracks, records = {}, []
    native_found = False
    with tarfile.open(path, "r|xz") as archive:
        for member in archive:
            # *_INTERP alternatives are not native tracks and are not used to
            # fill missing combinations. Never extract an archive member path.
            match = re.fullmatch(r"(\d+)M\.track\.eep", Path(member.name).name)
            if not member.isfile() or match is None:
                continue
            native_found = True
            named_mass = int(match[1])/100
            if mass_range is not None and not mass_range[0] <= named_mass <= mass_range[1]:
                continue
            try:
                with archive.extractfile(member) as stream:
                    data, eep, info = release.read_track(stream)
                missing_columns = set(extra_columns) - data.keys()
                if missing_columns:
                    raise ValueError(f"Missing requested native columns: {sorted(missing_columns)}")
                header_mass, composition = info["initial_mass"], info["composition"]
                # v2.5 high-mass headers can differ from their nominal file
                # labels by ~1e-9 Msun. Do not create duplicate nearly equal
                # mass axes across FeH planes; retain the original in metadata.
                matches = (np.isclose(header_mass, named_mass, rtol=0., atol=1e-8)
                           if release is v25 else header_mass == named_mass)
                if not matches:
                    raise ValueError("Initial mass disagrees with the member filename")
                mass = named_mass
                if (composition["[Fe/H]"], composition["[a/Fe]"], composition["v/vcrit"]) != (
                        expected_feh, afe, rotation):
                    raise ValueError("Track composition/rotation disagrees with the requested archive")
                mask = np.ones(len(eep), dtype=bool) if eep_range is None else (
                    (eep >= eep_range[0]) & (eep <= eep_range[1]))
                if not allow_missing and eep_range is not None and (
                        eep[0] > eep_range[0] or eep[-1] < eep_range[1]):
                    raise ValueError(f"Requested EEP {eep_range}; track provides {eep[0]}–{eep[-1]}")
                selected = {key: values[mask] for key, values in data.items()}
                age, hydrogen = selected["star_age"], selected["surface_h1"]
                if not allow_missing and (not np.all(np.isfinite(age) & (age > 0))
                        or not np.all(np.diff(age) > 0)
                        or not np.all(np.isfinite(hydrogen) & (hydrogen > 0))):
                    raise ValueError("Selected ages must increase; ages and surface hydrogen must be positive/finite")
                if release is v25:
                    bc_z, surface_feh = v25.surface_composition(selected, expected_feh)
                else:
                    bc_z = np.minimum(selected["log_surf_z"]-np.log10(v12.ZSUN), expected_feh+.1)
                    bc_z = np.where(np.isfinite(selected["log_surf_z"]), bc_z, np.nan)
                    valid_h = np.where(np.isfinite(hydrogen) & (hydrogen > 0), hydrogen, np.nan)
                    surface_feh = bc_z-np.log10(valid_h/v12.XSUN)
                queries = np.column_stack((bc_z, selected["log_Teff"], selected["log_g"]))
                supported = v12.bc_coverage(bc_axes, queries)[1]
                if not allow_missing and not np.all(supported):
                    first = int(np.flatnonzero(~supported)[0])
                    raise ValueError(f"Unsupported BC coverage at EEP {eep[mask][first]:g}: "
                                     f"(metallicity, logTeff, logg)={tuple(queries[first])}; "
                                     "no clipping or extrapolation is applied")
                magnitudes = np.full((len(age), len(v12.BANDS)), np.nan)
                magnitudes[supported] = (v12.MBOL_SUN - 2.5*selected["log_L"][supported, None]
                                        - v12.track_bc(bc_axes, bc_fields, queries[supported]))
                age = np.where(np.isfinite(age) & (age > 0), age, np.nan)
                scalar_values = (np.log10(age), age/1e9, selected["log_Teff"], selected["log_g"],
                                 selected["log_L"], selected["log_R"], selected["star_mass"],
                                 bc_z, surface_feh)
                values = np.column_stack((*scalar_values,
                                          *(selected[name] for name in extra_columns), magnitudes))
                if not allow_missing and not np.all(np.isfinite(values)):
                    raise ValueError("Nonfinite selected stellar fields or photometry")
                values[~np.isfinite(values)] = np.nan
                if mass in tracks:
                    raise ValueError("Duplicate native initial mass")
                tracks[mass] = (eep[mask], values)
                records.append({"member": member.name, "initial_mass": mass,
                                "initial_feh": expected_feh, "native_eep_range": [int(eep[0]), int(eep[-1])],
                                "primary_eeps": info["primary_eeps"],
                                "selected_nodes": len(age), "unsupported_bc_nodes": int((~supported).sum()),
                                "nonincreasing_age_steps": int(np.count_nonzero(np.diff(age) <= 0))})
                if header_mass != mass:
                    records[-1]["header_initial_mass"] = header_mass
            except ValueError as exc:
                raise ValueError(f"{path.name}, mass={named_mass:g}, FeH={expected_feh:g}, "
                                 f"member={member.name}: {exc}") from exc
    if not native_found or (not tracks and not allow_missing):
        raise ValueError(f"No native tracks within mass_range={mass_range} in {path.name}")
    _LOG.info("Read %d native tracks at FeH=%g", len(tracks), expected_feh)
    return tracks, records


def create_track_grid(path, *, mass_range=None, eep_range=None, feh=None, feh_range=None, version="1.2",
                      rotation=.4, afe=0., raw_cache_dir=None, local_dir=None,
                      offline=False, overwrite=False, allow_missing=True, include_structure=False):
    """Create a float64 NPZ compatible with MistModel.from_tracks; return its Path.

    mass_range=None and eep_range=None retain the union of all native masses
    and EEPs. Optional mass bounds are inclusive, in solar masses; optional
    eep_range is an inclusive integer pair (at least three nodes). Every EEP
    is retained. Supply feh with at least two distinct catalogue values, or
    feh_range=(min, max) to select every published metallicity in that inclusive
    range. The bounds need not be catalogue values; at least two values must
    be selected. These options are mutually exclusive. Omitting both selects
    all published metallicities. Both rotation rates are accepted.
    Versions 1.2 (default) and 2.5, afe=0, UBVRIplus, Av=0 and Rv=3.1 are
    supported. Nonzero v2.5 alpha is catalogued but its CMD chemistry is not
    yet validated for building photometric grids.

    include_structure=True (v2.5 only) also retains all 66 native non-isotope
    columns under their original names and units. star_mass is already stored;
    the remaining 65 are added alongside the existing fields and photometry.
    Nonfinite values remain missing per field. Finite source sentinels/zeros
    are preserved, and phase is a linearly interpolated diagnostic, not a
    categorical label. Additional columns do not change the inference outputs
    or likelihood unless explicitly requested/used.

    Archives are reused from the shared raw cache. local_dir may contain the
    official archive filenames for import; otherwise missing inputs download
    unless offline=True. A different NPZ output never changes the raw cache.

    Missing mass/FeH combinations and ragged track ends are preserved as NaN
    cells with a track_present mask. Unsupported BC coordinates invalidate
    only photometry. Invalid physical quantities are missing per field;
    nonincreasing ages invalidate their centered derivative stencils on load.
    allow_missing=False instead requires a complete finite grid with strictly
    increasing ages. Malformed files always raise. No age cut, mass/EEP
    interpolation, thinning, atmosphere clipping or extrapolation is applied.

    The NPZ carries numeric fields plus scalar metadata_json with configuration,
    input hashes and builder/parser hashes. No pickle or companion file is
    required. Publish atomically after successful validation; existing output
    requires overwrite=True. Grid construction itself does not require JAX x64.
    """
    mass_range = None if mass_range is None else _bounds(mass_range, "mass_range")
    eep_range = None if eep_range is None else _bounds(eep_range, "eep_range")
    if feh is not None and feh_range is not None:
        raise ValueError("feh and feh_range are mutually exclusive")
    feh_range = None if feh_range is None else _bounds(feh_range, "feh_range", positive=False)
    if not isinstance(allow_missing, bool):
        raise TypeError("allow_missing must be a bool")
    if not isinstance(include_structure, bool):
        raise TypeError("include_structure must be a bool")
    if include_structure and version != "2.5":
        raise ValueError("include_structure currently requires MIST v2.5")
    extra_columns = tuple(name for name in v25.STRUCTURE_COLUMNS if name not in _SCALARS) if include_structure else ()
    scalar_names = (*_SCALARS, *extra_columns)
    if eep_range is not None and (any(value != int(value) for value in eep_range)
                                  or eep_range[1]-eep_range[0] < 2):
        raise ValueError("eep_range must select at least three consecutive integer EEPs")
    # Resolve version/composition before any disk writes or downloads.
    metallicities = sorted(track_metallicities(version=version, rotation=rotation, afe=afe, feh=feh))
    if feh_range is not None:
        metallicities = [value for value in metallicities if feh_range[0] <= value <= feh_range[1]]
    if afe != 0.:
        raise ValueError("Photometric track grids currently require afe=0; nonzero v2.5 alpha chemistry is not validated")
    release = v25 if version == "2.5" else v12
    if len(metallicities) < 2:
        if feh_range is not None:
            raise ValueError(f"feh_range {feh_range} selects {len(metallicities)} published FeH values; "
                             "a rectangular track grid needs at least two")
        raise ValueError("A rectangular track grid needs at least two FeH values")
    bc_source = bolometric_correction_archive(version=version, afe=afe)
    sources = track_archives(version=version, rotation=rotation, afe=afe, feh=metallicities)
    path = Path(path).expanduser()
    if path.suffix != ".npz":
        raise ValueError("Track grid output must have a .npz extension")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; use overwrite=True")
    local_dir = None if local_dir is None else Path(local_dir).expanduser()
    if local_dir is not None and not local_dir.is_dir():
        raise FileNotFoundError(f"Local archive directory not found: {local_dir}")
    inputs = []

    def acquire(source):
        local = None if local_dir is None else local_dir / source.filename
        if local is not None and not local.is_file():
            local = None
        _LOG.info("Acquiring %s", source.filename)
        archive = fetch_archive(source, cache_dir=raw_cache_dir, local_path=local, offline=offline)
        inputs.append({"source": asdict(source), **_file_record(archive)})
        return archive

    bc_axes, bc_fields = release.read_bc_archive(acquire(bc_source))
    tracks, records = {}, []
    for source, metallicity in zip(sources, metallicities):
        members, metadata = _read_archive(acquire(source), metallicity, rotation, afe,
                                          mass_range, eep_range, bc_axes, bc_fields, allow_missing,
                                          release, extra_columns)
        tracks.update({(mass, metallicity): values for mass, values in members.items()})
        records.extend(metadata)
    masses = sorted({mass for mass, _ in tracks})
    if len(masses) < 2:
        raise ValueError("Selected domain needs at least two native initial masses")
    missing = [(m, z) for m in masses for z in metallicities if (m, z) not in tracks]
    if missing and not allow_missing:
        raise ValueError(f"Missing native mass/FeH combinations: {missing[:8]} "
                         f"({len(missing)} total); choose a complete domain")
    available = [eep for eep, _ in tracks.values() if len(eep)]
    if not available:
        raise ValueError("No native EEP nodes overlap the requested range")
    limits = eep_range or (min(e[0] for e in available), max(e[-1] for e in available))
    eeps = np.arange(int(limits[0]), int(limits[1])+1, dtype=np.float64)
    if len(eeps) < 3:
        raise ValueError("At least three native EEPs are required")
    shape = (len(masses), len(metallicities), len(eeps))
    values = np.full((*shape, len(scalar_names)+len(v12.BANDS)), np.nan)
    present = np.zeros(shape, dtype=bool)
    for i, mass in enumerate(masses):
        for j, z in enumerate(metallicities):
            if (mass, z) not in tracks:
                continue
            native_eep, fields = tracks[mass, z]
            indices = (native_eep-eeps[0]).astype(int)
            values[i, j, indices] = fields
            present[i, j, indices] = True
    if not allow_missing and not present.all():
        raise ValueError("Native EEP coverage is incomplete; choose a complete domain")
    arrays = dict(initial_mass=np.array(masses), initial_feh=np.array(metallicities), eep=eeps,
                  **{name: values[..., i] for i, name in enumerate(scalar_names)},
                  absolute_magnitudes=values[..., len(scalar_names):], bands=np.array(v12.BANDS))
    metadata = {
        "format_version": 2 if allow_missing else 1, "kind": "track", "mist_version": version,
        "allow_missing": allow_missing,
        "coverage": {"present_nodes": int(present.sum()), "total_nodes": int(present.size),
                     "missing_mass_feh": missing,
                     "finite_nodes": {name: int(np.isfinite(values[..., i]).sum())
                                      for i, name in enumerate((*scalar_names, *v12.BANDS))}},
        "rotation": rotation, "afe": afe, "photometry": "UBVRIplus", "Av": 0., "Rv": 3.1,
        "mass_range": mass_range, "eep_range": eep_range, "feh": metallicities, "feh_range": feh_range,
        "shape": list(values.shape[:3]), "native_tracks": len(records), "inputs": inputs,
        "tracks": records, "BC_recipe": f"v{version} Steffen logT (linear edges) / linear logg / Steffen metallicity (three-point edges)",
        "chemistry": {"surface_feh": "CMD log(Z/Zsun) - log(X/Xsun)",
                      "bc_cap": "initial_feh + 0.1", "log_z_column":
                      "log_surf_cell_z" if version == "2.5" else "log_surf_z"},
        "Zsun": release.ZSUN, "Xsun": release.XSUN, "Mbol_sun": release.MBOL_SUN,
        "implementation_sha256": {p.name: _file_record(p)["sha256"] for p in
                                  ((Path(__file__), Path(v12.__file__)) +
                                   ((Path(v25.__file__),) if version == "2.5" else ()))},
    }
    if include_structure:
        metadata["include_structure"] = True
        metadata["native_columns"] = list(v25.STRUCTURE_COLUMNS)
    if allow_missing:
        arrays["track_present"] = present
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication even if another builder finishes
            # after the initial existence check. Both paths share a filesystem.
            os.link(temporary, path)
        _LOG.info("Saved %s: %d tracks, grid shape %s", path, len(records), values.shape[:3])
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="output NPZ")
    parser.add_argument("--mass-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="default: all native masses")
    parser.add_argument("--eep-range", nargs=2, type=int, metavar=("MIN", "MAX"), help="default: all native EEPs")
    feh_options = parser.add_mutually_exclusive_group()
    feh_options.add_argument("--feh", nargs="+", type=float,
                             help="explicit published FeH values; default: all published values")
    feh_options.add_argument("--feh-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                             help="select all published FeH values in this inclusive range (at least two)")
    parser.add_argument("--version", default="1.2")
    parser.add_argument("--rotation", type=float, default=.4)
    parser.add_argument("--afe", type=float, default=0.)
    parser.add_argument("--raw-cache-dir", type=Path)
    parser.add_argument("--local-dir", type=Path, help="existing official archive filenames")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-structure", action="store_true",
                        help="v2.5: retain all 66 native non-isotope columns (original names/units)")
    parser.add_argument("--strict", dest="allow_missing", action="store_false",
                        help="require a complete finite grid with increasing ages")
    args = vars(parser.parse_args(argv))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_track_grid(**args)


if __name__ == "__main__":
    main()
