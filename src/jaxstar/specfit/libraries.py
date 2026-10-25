"""Library-specific I/O and preparation; numerical interpolation lives in grid.

Legacy jaxspec NPZ loaders retain stored wavelength samples and flux dtype
(subject to JAX dtype canonicalization). They do not convert wavelengths or
normalize flux. New common artifacts use storage.load_spectral_grid instead.
Raw preparers operate offline and never resample onto observed wavelengths;
persist their results once with storage.save_spectral_grid for future sessions.
"""

from collections.abc import Mapping
from itertools import product
from pathlib import Path
import re

import numpy as np

from jaxstar.grid import Field, RectilinearGrid
from ._data import _SpectralLibrary, _validate_wavelength
from ._selection import _covering_indices, _wavelength_request
from .sampling import _sampling_mode, _target_wavelengths


_SCHEMAS = {
    "coelho": (("teff", "tgrid"), ("logg", "ggrid"),
               ("feh", "fgrid"), ("alpha", "agrid")),
    "bosz": (("teff", "tgrid"), ("logg", "ggrid"), ("mh", "mgrid"),
             ("alpha", "agrid"), ("carbon", "cgrid"), ("vmic", "vgrid")),
    "tlusty": (("teff", "tgrid"), ("logg", "ggrid"), ("logZ", "zgrid")),
}


def select_grid_paths(gridpath, wavelength, *, wav_margin=3., pattern="*.npz",
                      regions=None):
    """Select one legacy NPZ per requested wavelength region, in input row order.

    ``wavelength`` is a positive increasing array in Angstrom, shaped (pixel,)
    or (region, pixel). All pixels, including any observation's masked pixels,
    must be covered with at least ``wav_margin`` Angstrom on EACH side.
    Only stored ``wavgrid`` arrays are inspected, never filenames or flux data.
    ``pattern`` filters filenames within ``gridpath`` (e.g. '*_normed.npz').
    Optional ``regions`` labels identify rows in selection errors.

    Return a list of pathlib.Path objects. A file can serve several regions;
    zero or multiple covering candidates for any row raise ValueError. No
    matching files raise FileNotFoundError. Malformed candidates fail with
    their path. Common-format NPZs and raw atmosphere files are not supported.
    Wavelength medium/units are not converted or inferred from the files.
    """
    wave, labels, margin = _wavelength_request(wavelength, regions, wav_margin)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty glob pattern")
    root = Path(gridpath).expanduser()
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No grids matching {pattern!r} in {root}")
    candidates = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                stored = data["wavgrid"]
                if stored.dtype.kind not in "fiu":
                    raise ValueError("wavgrid must contain real numeric values")
                _validate_wavelength(stored, ndim=1)
                candidates.append((str(path), float(stored[0]), float(stored[-1])))
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"{path}: cannot read a valid legacy wavgrid: {exc}") from exc
    return [paths[i] for i in _covering_indices(wave, labels, margin, candidates)]


def load_coelho(paths, *, pixel_slices=None, regions=None,
                wavelength_medium="unknown", flux_kind="as_stored"):
    """Compatibility adapter for one or more legacy jaxspec Coelho NPZ regions.

    Each file contains tgrid/ggrid/fgrid/agrid, wavgrid (Angstrom) and flux.
    ``pixel_slices`` may be one slice for all files or a sequence of slices,
    enabling small windows from full grids without wavelength resampling.
    Files must share atmosphere axes, flux dtype and selected pixel count.
    ``regions`` optionally labels rows (strings/integers; repetition allowed).
    Medium/normalization are caller declarations: old NPZs do not record them.

    Returns a private immutable carrier with .grid, .wavelength and metadata;
    evaluate .grid.interpolate(coordinates)['flux'] for intrinsic spectra.
    New common artifacts use load_spectral_grid, independent of their library.
    """
    return _load_legacy("coelho", paths, pixel_slices, regions,
                          wavelength_medium, flux_kind)


def load_bosz(paths, *, pixel_slices=None, regions=None,
              wavelength_medium="unknown", flux_kind="as_stored"):
    """Adapt legacy jaxspec BOSZ NPZs; see load_coelho for loading arguments.

    Native keys tgrid/ggrid/mgrid/agrid/cgrid/vgrid map to teff/logg/mh/alpha/
    carbon/vmic, where abundances mean [M/H], [alpha/M] and [C/M].
    """
    return _load_legacy("bosz", paths, pixel_slices, regions,
                          wavelength_medium, flux_kind)


def load_tlusty(paths, *, pixel_slices=None, regions=None,
                wavelength_medium="unknown", flux_kind="as_stored"):
    """Adapt legacy jaxspec TLUSTY NPZs; see load_coelho for loading arguments.

    tgrid/ggrid/zgrid map to teff/logg/logZ; logZ is log10(Z/Z_sun).
    Nonuniform axes and wavelength samples are retained unchanged.
    """
    return _load_legacy("tlusty", paths, pixel_slices, regions,
                          wavelength_medium, flux_kind)


def _load_legacy(library, paths, pixel_slices, regions, medium, flux_kind):
    if isinstance(paths, (str, Path)):
        paths = (Path(paths),)
    else:
        paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError("at least one prepared grid path is required")
    if pixel_slices is None or isinstance(pixel_slices, slice):
        selections = (pixel_slices or slice(None),) * len(paths)
    else:
        selections = tuple(pixel_slices)
    if len(selections) != len(paths) or any(not isinstance(s, slice) for s in selections):
        raise ValueError("pixel_slices must supply one slice per path")
    if any(s.step is not None and s.step <= 0 for s in selections):
        raise ValueError("pixel slices must have a positive step")

    axes, packed, wavelength = None, None, None
    for region, (path, selection) in enumerate(zip(paths, selections)):
        with np.load(path, allow_pickle=False) as data:
            required = {key for _, key in _SCHEMAS[library]} | {"wavgrid", "flux"}
            if library == "coelho":
                required.remove("agrid")
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing entries {sorted(missing)}")
            current_axes = {name: data[key] for name, key in _SCHEMAS[library]
                            if key in data.files}
            flux = data["flux"]
            if library == "coelho" and "alpha" not in current_axes:
                current_axes["alpha"] = np.array([0.0])
                if flux.ndim == 4:
                    flux = np.expand_dims(flux, 3)
                elif flux.ndim != 5 or flux.shape[3] != 1:
                    raise ValueError(f"{path}: missing agrid cannot describe multiple alpha nodes")
            if any(value.ndim != 1 or value.size == 0 for value in current_axes.values()):
                raise ValueError(f"{path}: atmosphere axes must be non-empty one-dimensional arrays")
            wav = data["wavgrid"]
            _validate_wavelength(wav, ndim=1)
            shape = tuple(len(current_axes[name]) for name, _ in _SCHEMAS[library])
            if flux.shape != shape + (wav.size,):
                raise ValueError(f"{path}: flux shape {flux.shape} does not match axes and wavelength")
            if not np.issubdtype(flux.dtype, np.floating):
                raise TypeError(f"{path}: flux must have a floating-point dtype")
            wav, selected = wav[selection], flux[..., selection]
            _validate_wavelength(wav, ndim=1)
            if axes is None:
                axes = current_axes
                packed = np.empty(shape + (len(paths), wav.size), dtype=flux.dtype)
                wavelength = np.empty((len(paths), wav.size), dtype=wav.dtype)
            else:
                if any(not np.array_equal(axes[name], current_axes[name]) for name in axes):
                    raise ValueError("regions must share identical atmosphere axes; load separately")
                if selected.shape != packed.shape[:-2] + (packed.shape[-1],):
                    raise ValueError("regions must share a pixel count; load different lengths separately")
                if selected.dtype != packed.dtype:
                    raise ValueError("regions must share a flux dtype; load different dtypes separately")
                if wav.dtype != wavelength.dtype:
                    raise ValueError("regions must share a wavelength dtype")
            packed[..., region, :] = selected
            wavelength[region] = wav
            # Do not retain a full source flux array while reading the next one.
            del flux, selected
    return _assemble(library, axes, wavelength, packed, regions, medium,
                     flux_kind, tuple(str(path) for path in paths))


def _assemble(library, axes, wavelength, flux, regions, medium, flux_kind, sources):
    grid = RectilinearGrid(axes=axes, fields={
        "flux": Field(flux, dims=tuple(axes), payload_dims=("region", "pixel")),
    })
    if regions is None:
        regions = tuple(range(wavelength.shape[0]))
    return _SpectralLibrary(grid, wavelength, tuple(regions), library,
                            medium, flux_kind, tuple(sources))


def prepare_korg(*, axes, wavelength_ranges, wavelength_medium, pixels=None,
                 regions=None, normalized=True, sampling="log", velocity_step=None,
                 synthesis_step=.0025, line_buffer=10., cache_dir=None,
                 progress=None, return_report=False):
    """Prepare a normalized Korg grid with optional JuliaCall dependencies.

    Axes are teff/logg/feh/alpha, with alpha=[alpha/Fe]. Microturbulence uses
    empirical_vmic at each node; it is not an independent axis. wavelength_medium
    must explicitly be 'air' or 'vacuum'. See :func:`jaxstar.specfit.korg.prepare_korg`
    for physics, installation, sampling, caching and provenance options.
    Importing this entry point does not initialize Julia.
    """
    from .korg import prepare_korg as synthesize_grid
    return synthesize_grid(
        axes=axes, wavelength_ranges=wavelength_ranges, wavelength_medium=wavelength_medium,
        pixels=pixels, regions=regions, normalized=normalized, sampling=sampling,
        velocity_step=velocity_step, synthesis_step=synthesis_step, line_buffer=line_buffer,
        cache_dir=cache_dir, progress=progress, return_report=return_report)


def prepare_coelho(data_dir, *, axes, wavelength_ranges, pixels=None,
                   regions=None, wavelength_medium="vacuum", normalized=True,
                   source_overrides=None, sampling=None, velocity_step=None):
    """Prepare Coelho FITS nodes directly into a common spectral library.

    ``axes`` maps teff/logg/feh/alpha to increasing coordinate arrays.
    ``wavelength_ranges`` is a sequence of open (lower, upper) Angstrom bounds
    in the output medium. When sampling is omitted, ``pixels=None`` keeps native
    samples; each region must then have equal length and identical samples across
    all nodes. An integer ``pixels`` instead prepares common linear sampling over
    the intersection of selected node coverages, as in the old preparer.

    Fitting grids use ``sampling='log', velocity_step=dv`` (positive finite km/s,
    scalar or per-region), with dlnlambda = velocity_step / 299792.458. This is
    numerical model sampling, not instrumental resolving power or detector pixel
    spacing; dv must be supplied explicitly. ``pixels`` remains a mutually
    exclusive alternative for compatibility/tests/special-purpose count control.
    Log mode directly interpolates native spectra over the requested
    bounds, including bracketing source samples; every node must cover those
    bounds. Count mode includes both endpoints; fixed-step mode retains complete
    intervals inside the upper bound. Regions must have a common pixel count.
    Omitted sampling retains M2 behavior; explicit 'native'/'linear' are allowed.

    Row 0 is normalized flux; row 1 is unnormalized flux. Raw output is float32,
    matching frozen preparation. ``source_overrides`` maps atmosphere tuples
    to explicit FITS paths for known missing-model replacements/naming aliases;
    no replacement is applied silently. All input paths are checked before
    opening any spectra. Paths are recorded in .sources.
    Returns an in-memory private spectral carrier; persist it once with
    save_spectral_grid and use load_spectral_grid in subsequent processes.
    """
    axes = _preparation_axes("coelho", axes)
    _sampling_mode(sampling, pixels, velocity_step)
    paths, _, report = _spectral_source_plan("coelho", data_dir, axes, "error", source_overrides)
    _require_source_plan(report)
    kind = "normalized" if normalized else "unnormalized"
    _output_medium(wavelength_medium)

    def reader(node, bounds_only=False):
        from astropy.io import fits
        with fits.open(paths[node], memmap=True) as hdus:
            header = hdus[0].header
            wav = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CD1_1"]
            if wavelength_medium == "vacuum":
                wav = _air_to_vacuum(wav)
            flux = None
            if not bounds_only:
                data = hdus[0].data
                row = 0 if normalized else 1
                if data.ndim != 2 or data.shape[0] <= row:
                    raise ValueError(f"{paths[node]}: missing Coelho flux row {row}")
                flux = np.asarray(data[row], dtype=np.float64)
        return wav, flux

    return _prepare_nodes("coelho", axes, reader, wavelength_ranges, pixels,
                          regions, wavelength_medium, kind, tuple(paths.values()),
                          sampling=sampling, velocity_step=velocity_step)


def prepare_bosz(data_dir, *, axes, wavelength_ranges, pixels=None,
                 regions=None, wavelength_medium="vacuum", normalized=True,
                 source_overrides=None, sampling=None, velocity_step=None,
                 missing_nodes="error"):
    """Prepare BOSZ-2024 gzip ASCII nodes (wavelength, H, continuum).

    Axes are teff/logg/mh/alpha/carbon/vmic. Normalized flux is H/continuum;
    ``normalized=False`` retains H. For fitting use ``sampling='log',
    velocity_step=dv`` in km/s: dlnlambda = dv / 299792.458. Supply dv explicitly
    as numerical model sampling, independent of instrumental resolving power
    and detector pixels. ``pixels`` is a mutually exclusive count-controlled
    alternative for compatibility/tests/special purposes. See prepare_coelho
    for region coverage and native/linear options.
    Filenames use mp for logg > 3, ms otherwise, and m±X.XX subdirectories.
    All input paths are checked before opening any spectra. Missing files fail
    by default. With ``missing_nodes='logg'``, missing nodes use the two native
    neighbours at logg +/- 0.5, holding all other coordinates fixed. Each source
    is resampled onto the output wavelengths before averaging its flux. At the
    upper end of the requested logg axis only, an absent upper neighbour permits
    copying the lower neighbour. There is no recursive filling or linear extrapolation.
    Neighbours may lie outside the requested axes. Corrupt files and insufficient
    wavelength coverage still fail. Use check_spectral_inputs to inspect and
    retain the replacement report; .sources lists the actual files used.
    ``source_overrides`` supports deliberate replacements in strict mode only.
    Native BOSZ wavelengths below 2000 Angstrom are already vacuum; above they
    are air. Vacuum conversion respects this convention. Air output requires
    selected regions above 2000 Angstrom. Some native ASCII files repeat a
    wavelength at their printed precision. Equal wavelengths are collapsed by
    averaging H/continuum (or H when unnormalized), before resampling. Invalid
    flux is not discarded in this average.
    Returns an in-memory private spectral carrier for save_spectral_grid.
    """
    axes = _preparation_axes("bosz", axes)
    _sampling_mode(sampling, pixels, velocity_step)
    paths, node_sources, report = _spectral_source_plan(
        "bosz", data_dir, axes, missing_nodes, source_overrides)
    _require_source_plan(report)
    _output_medium(wavelength_medium)
    bounds = _region_bounds(wavelength_ranges)
    if wavelength_medium == "air" and np.any(bounds[:, 0] < 2000):
        raise ValueError("BOSZ air regions must lie above 2000 Angstrom; use vacuum for UV")

    def reader(node, bounds_only=False):
        data = np.loadtxt(paths[node], ndmin=2, usecols=(0,) if bounds_only else (0, 1, 2))
        wav = data[:, 0]
        flux = None
        if not bounds_only:
            flux = data[:, 1]
            if normalized:
                flux = np.divide(flux, data[:, 2], out=np.full_like(flux, np.nan),
                                 where=data[:, 2] != 0)
        if np.any(np.diff(wav) <= 0):
            # Original-resolution ASCII nodes can round onto the same printed
            # wavelength. Merge exact duplicates; preserve distinct close nodes.
            order = np.argsort(wav, kind="stable")
            wav, first, counts = np.unique(wav[order], return_index=True, return_counts=True)
            if flux is not None:
                flux = np.add.reduceat(flux[order], first) / counts
        if wavelength_medium == "vacuum":
            wav = np.where(wav > 2000, _air_to_vacuum(wav), wav)
        return wav, flux

    return _prepare_nodes("bosz", axes, reader, bounds, pixels, regions,
                          wavelength_medium, "normalized" if normalized else "unnormalized",
                          tuple(report["sources"]), sampling=sampling, velocity_step=velocity_step,
                          node_sources=node_sources)


def check_spectral_inputs(library, data_dir, *, axes, missing_nodes="error", source_overrides=None):
    """Check local BOSZ/Coelho paths without opening spectra or allocating a grid.

    Return a JSON-serializable report with ready, requested_nodes, sources,
    replacements and unresolved entries. Replacement entries identify target
    coordinates, method, source coordinates/paths and weights. Unresolved entries
    include every missing target and the unavailable neighbours. This checks
    local file existence, not file contents or the upstream catalogue. Download
    intended native inputs first; a missing download is also treated as missing.
    Preparation repeats this check before reading spectra.

    ``missing_nodes='logg'`` is BOSZ-only; see prepare_bosz for its interpolation
    and upper-edge copy rules. The default is strict for both libraries.
    """
    if library not in {"bosz", "coelho"}:
        raise ValueError("input checking supports only 'bosz' and 'coelho'")
    axes = _preparation_axes(library, axes)
    return _spectral_source_plan(library, data_dir, axes, missing_nodes, source_overrides)[2]


def _native_spectral_path(library, root, node):
    if library == "coelho":
        t, g, f, a = node
        filename = f"{_integer(t)}_{_integer(g * 10):02d}_{'p' if f >= 0 else 'm'}{_integer(abs(f) * 10):02d}p{_integer(a * 10):02d}.ms.fits"
        return root / filename
    t, g, m, a, c, v = node
    for encoded in (g * 10, m * 100, a * 100, c * 100):
        _integer(encoded)
    geometry = "mp" if g > 3 else "ms"
    filename = f"bosz2024_{geometry}_t{_integer(t)}_g{g:+.1f}_m{m:+.2f}_a{a:+.2f}_c{c:+.2f}_v{_integer(v)}_rorig_noresam.txt.gz"
    return root / f"m{m:+.2f}" / filename


def _spectral_source_plan(library, data_dir, axes, missing_nodes, source_overrides):
    if missing_nodes not in ("error", "logg"):
        raise ValueError("missing_nodes must be 'error' or 'logg'")
    if missing_nodes == "logg":
        if library != "bosz":
            raise ValueError("missing_nodes='logg' is supported only for BOSZ")
        if source_overrides:
            raise ValueError("missing_nodes='logg' cannot be combined with source_overrides")
        for g in axes["logg"]:
            _integer(g * 2)
    root = Path(data_dir).expanduser()
    nodes = tuple(product(*axes.values()))
    paths = {node: _native_spectral_path(library, root, node) for node in nodes}
    paths = _source_overrides(paths, source_overrides, root)
    node_sources, replacements, unresolved = {}, [], []

    def coordinates(node):
        return {key: float(value) for key, value in zip(axes, node)}

    for node in nodes:
        path = paths[node]
        if path.is_file():
            node_sources[node] = ((node, 1.),)
            continue
        entry = dict(node=coordinates(node), missing_path=str(path))
        if missing_nodes == "error":
            unresolved.append({**entry, "reason": "input file is missing"})
            continue
        t, g, *rest = node
        lower, upper = (t, g - .5, *rest), (t, g + .5, *rest)
        for neighbor in (lower, upper):
            paths[neighbor] = _native_spectral_path(library, root, neighbor)
        has_lower, has_upper = paths[lower].is_file(), paths[upper].is_file()
        if has_lower and has_upper:
            method, contributors = "linear_logg", ((lower, .5), (upper, .5))
        elif has_lower and not has_upper and g == axes["logg"][-1]:
            method, contributors = "copy_lower_logg", ((lower, 1.),)
        else:
            absent = [str(paths[n]) for n in (lower, upper) if not paths[n].is_file()]
            unresolved.append({**entry, "missing_neighbors": absent,
                               "reason": "adjacent native logg files are required; lower-only copying is allowed only at the requested upper edge"})
            continue
        node_sources[node] = contributors
        replacements.append({**entry, "method": method, "sources": [
            dict(node=coordinates(source), path=str(paths[source]), weight=weight)
            for source, weight in contributors]})
    sources = list(dict.fromkeys(str(paths[source]) for contributors in node_sources.values()
                                for source, _ in contributors))
    report = dict(ready=not unresolved, requested_nodes=len(nodes), missing_nodes=missing_nodes,
                  sources=sources, replacements=replacements, unresolved=unresolved)
    return paths, node_sources, report


def _require_source_plan(report):
    if report["unresolved"]:
        details = []
        for entry in report["unresolved"]:
            details.append(f"{entry['missing_path']}: {entry['reason']}")
            details.extend(f"  Missing neighbor: {path}" for path in entry.get("missing_neighbors", ()))
        raise FileNotFoundError(
            f"Input check failed for {len(report['unresolved'])} node(s); no spectra were read.\n"
            + "\n".join(details))


def prepare_tlusty(data_dir, *, wavelength_ranges, axes=None, pixels=None,
                   regions=None, flux_kind="normalized", wavelength_medium="unknown",
                   source_overrides=None, sampling=None, velocity_step=None):
    """Prepare matched TLUSTY .7.gz spectra and .17.gz continua.

    Discover Z*_vt2/Z*_ostar directories, logZ=log10(Z/Z_sun), and teff/logg
    from filenames. Default axes retain logg > 2.9 and skip OSTAR teff <= 30000,
    as frozen preparation does. Explicit axes select a complete subgrid.
    Missing nodes fail rather than silently copying the nearest logg. Optional
    source_overrides map a target node to an explicit (spectrum, continuum) pair.

    flux_kind is normalized (continuum division), median_scaled (the legacy
    'absolute' option, scaled per selected region), or unnormalized (native H).
    No wavelength-medium conversion is performed; callers may declare a known
    medium. For fitting use ``sampling='log', velocity_step=dv`` in km/s:
    dlnlambda = dv / 299792.458. Supply dv explicitly as numerical model sampling,
    independent of instrumental resolving power and detector pixels. ``pixels``
    is a mutually exclusive alternative for compatibility/tests/special-purpose
    count control. See prepare_coelho for region coverage and native/linear options.
    Returns an in-memory private spectral carrier for save_spectral_grid.
    """
    if flux_kind not in {"normalized", "median_scaled", "unnormalized"}:
        raise ValueError("unsupported TLUSTY flux_kind")
    root = Path(data_dir).expanduser()
    discovered = {}
    for directory in sorted((*root.glob("Z*_vt2"), *root.glob("Z*_ostar"))):
        z = np.log10(float(directory.name.split("_")[0][1:]))
        for spectrum in sorted(directory.glob("*.7.gz")):
            match = re.match(r"^[A-Za-z]{1,2}(\d+(?:\.\d+)?)g(\d{3})", spectrum.name)
            if match is None:
                raise ValueError(f"cannot parse TLUSTY atmosphere from {spectrum}")
            t, g = float(match[1]), float(match[2]) / 100
            if directory.name.endswith("_ostar") and t <= 30000:
                continue
            node = (t, g, z)
            if node in discovered:
                raise ValueError(f"ambiguous TLUSTY node {node}")
            continuum = spectrum.with_name(spectrum.name[:-5] + ".17.gz")
            discovered[node] = (spectrum, continuum)
    if not discovered:
        raise ValueError("no TLUSTY spectra found in Z*_vt2/Z*_ostar directories")
    if axes is None:
        axes = {name: np.unique([node[index] for node in discovered])
                for index, (name, _) in enumerate(_SCHEMAS["tlusty"])}
        axes["logg"] = axes["logg"][axes["logg"] > 2.9]
    axes = _preparation_axes("tlusty", axes)
    nodes = tuple(product(*axes.values()))
    paths = {node: discovered[node] for node in nodes if node in discovered}
    for node, pair in (source_overrides or {}).items():
        if node not in nodes or len(pair) != 2:
            raise ValueError("TLUSTY source_overrides must map a selected node to two paths")
        paths[node] = tuple(root / Path(path) for path in pair)
    missing = set(nodes) - set(paths)
    if missing:
        raise ValueError(f"missing TLUSTY atmosphere nodes: {sorted(missing)}")

    def reader(node, bounds_only=False):
        spectrum, continuum = paths[node]
        data = np.loadtxt(spectrum, ndmin=2, usecols=(0,) if bounds_only else (0, 1))
        if bounds_only:
            return data[:, 0], None
        wav, flux = data[:, 0], data[:, 1]
        if flux_kind == "normalized":
            cont = np.loadtxt(continuum, ndmin=2, usecols=(0, 1))
            cont = cont[np.argsort(cont[:, 0])]
            _validate_wavelength(cont[:, 0], ndim=1)
            denominator = np.interp(wav, cont[:, 0], cont[:, 1], left=np.nan, right=np.nan)
            flux = np.divide(flux, denominator, out=np.full_like(flux, np.nan),
                             where=denominator != 0)
        return wav, flux

    sources = tuple(str(path) for node in nodes for path in paths[node])
    return _prepare_nodes("tlusty", axes, reader, wavelength_ranges, pixels,
                          regions, wavelength_medium, flux_kind, sources,
                          sampling=sampling, velocity_step=velocity_step)


def _preparation_axes(library, axes):
    names = tuple(name for name, _ in _SCHEMAS[library])
    if not isinstance(axes, Mapping) or set(axes) != set(names):
        raise ValueError(f"{library} axes must contain exactly {names}")
    result = {}
    for name in names:
        values = np.asarray(axes[name], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a non-empty finite one-dimensional axis")
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} must be strictly increasing")
        result[name] = values
    return result


def _source_overrides(paths, overrides, root):
    for node, path in (overrides or {}).items():
        if node not in paths:
            raise ValueError(f"source override {node} is not a selected atmosphere node")
        paths[node] = root / Path(path)
    return paths


def _integer(value):
    integer = int(round(value))
    if not np.isclose(value, integer, rtol=0, atol=1e-8):
        raise ValueError(f"{value} cannot be encoded in the native filename")
    return integer


def _output_medium(medium):
    if medium not in {"air", "vacuum"}:
        raise ValueError("raw Coelho/BOSZ output wavelength_medium must be 'air' or 'vacuum'")


def _air_to_vacuum(wavelength):
    # The Morton conversion used by frozen modelgrid.air_to_vac.
    s = 1e4 / wavelength
    n = (1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s * s)
         + 0.0001599740894897 / (38.92568793293 - s * s))
    return wavelength * n


def _region_bounds(bounds):
    bounds = np.asarray(bounds, dtype=np.float64)
    if (bounds.ndim != 2 or bounds.shape[0] == 0 or bounds.shape[1] != 2
            or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] <= 0)
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("wavelength_ranges must contain positive finite (lower, upper) pairs")
    return bounds


def _selected_regions(reader, node, bounds, *, bounds_only=False, median_scaled=False, bracket=False):
    wav, flux = reader(node, bounds_only=bounds_only)
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim != 1:
        raise ValueError(f"node {node}: native wavelength must be one-dimensional")
    if not bounds_only:
        flux = np.asarray(flux)
        if flux.shape != wav.shape:
            raise ValueError(f"node {node}: flux shape does not match native wavelength")
    if np.any(np.diff(wav) <= 0):
        order = np.argsort(wav)
        wav = wav[order]
        if not bounds_only:
            flux = flux[order]
    _validate_wavelength(wav, ndim=1)
    result = []
    for lower, upper in bounds:
        keep = (wav > lower) & (wav < upper)
        if bracket:
            if lower < wav[0] or upper > wav[-1]:
                raise ValueError(f"node {node}: native wavelength does not cover requested log range {(lower, upper)}")
            selection = slice(max(0, np.searchsorted(wav, lower, side="right") - 1),
                              np.searchsorted(wav, upper, side="left") + 1)
        else:
            selection = keep
        selected = wav[selection]
        if selected.size == 0:
            raise ValueError(f"node {node}: empty wavelength region {(lower, upper)}")
        values = None if bounds_only else flux[selection]
        if not bounds_only:
            if not np.all(np.isfinite(values)):
                raise ValueError(f"node {node}: nonfinite flux or missing continuum coverage")
            if median_scaled:
                # Keep the existing per-region intrinsic scaling convention;
                # extra bracketing samples must not change the median.
                scaling_values = flux[keep]
                if scaling_values.size == 0:
                    raise ValueError(f"node {node}: empty region for median scaling")
                median = np.median(scaling_values)
                if not np.isfinite(median) or median == 0:
                    raise ValueError(f"node {node}: invalid median for flux scaling")
                values = values / median
        result.append((selected, values))
    return result


def _prepare_nodes(library, axes, reader, bounds, pixels, regions, medium, kind, sources,
                   *, sampling=None, velocity_step=None, node_sources=None):
    mode = _sampling_mode(sampling, pixels, velocity_step)
    bounds = _region_bounds(bounds)
    nodes = tuple(product(*axes.values()))
    if node_sources is None:
        node_sources = {node: ((node, 1.),) for node in nodes}
    wavelength, flux = None, None
    if mode == "log":
        # Requested bounds determine the target once; each source is interpolated
        # straight to this grid, without a linear prepared intermediate or an
        # extra coverage-discovery pass.
        wavelength = _target_wavelengths(bounds, mode, pixels, velocity_step)
    elif mode == "linear":
        # A bounds-only pass avoids retaining every native spectrum or building
        # large DataFrames. The second pass streams one node into the output.
        lower = np.full(len(bounds), -np.inf)
        upper = np.full(len(bounds), np.inf)
        input_nodes = dict.fromkeys(source for contributors in node_sources.values()
                                   for source, _ in contributors)
        for node in input_nodes:
            samples = _selected_regions(reader, node, bounds, bounds_only=True)
            lower = np.maximum(lower, [wav[0] for wav, _ in samples])
            upper = np.minimum(upper, [wav[-1] for wav, _ in samples])
        if np.any(upper <= lower):
            raise ValueError("nodes have no common wavelength interval for a region")
        wavelength = np.stack([np.linspace(lo, hi, pixels) for lo, hi in zip(lower, upper)])

    for flat_index, node in enumerate(nodes):
        combined = None
        for source, weight in node_sources[node]:
            samples = _selected_regions(reader, source, bounds, median_scaled=kind == "median_scaled",
                                        bracket=mode == "log")
            if wavelength is None:
                sizes = {wav.size for wav, _ in samples}
                if len(sizes) != 1:
                    raise ValueError("native regions have different lengths; specify pixels or prepare separately")
                wavelength = np.stack([wav for wav, _ in samples])
            if flux is None:
                flux = np.empty(tuple(len(axis) for axis in axes.values()) + wavelength.shape,
                                dtype=np.float32)
            if combined is None:
                combined = np.zeros(wavelength.shape, dtype=np.float64)
            for region, (wav, values) in enumerate(samples):
                if mode == "native":
                    if not np.array_equal(wav, wavelength[region]):
                        raise ValueError("native wavelength samples differ across nodes; specify pixels")
                    combined[region] += weight * values
                else:
                    combined[region] += weight * np.interp(wavelength[region], wav, values)
        index = np.unravel_index(flat_index, flux.shape[:len(axes)])
        flux[index] = combined
    return _assemble(library, axes, wavelength, flux, regions, medium, kind, sources)
