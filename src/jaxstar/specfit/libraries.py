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


_SCHEMAS = {
    "coelho": (("teff", "tgrid"), ("logg", "ggrid"),
               ("feh", "fgrid"), ("alpha", "agrid")),
    "bosz": (("teff", "tgrid"), ("logg", "ggrid"), ("mh", "mgrid"),
             ("alpha", "agrid"), ("carbon", "cgrid"), ("vmic", "vgrid")),
    "tlusty": (("teff", "tgrid"), ("logg", "ggrid"), ("logZ", "zgrid")),
}


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


def prepare_coelho(data_dir, *, axes, wavelength_ranges, pixels=None,
                   regions=None, wavelength_medium="vacuum", normalized=True,
                   source_overrides=None):
    """Prepare Coelho FITS nodes directly into a common spectral library.

    ``axes`` maps teff/logg/feh/alpha to increasing coordinate arrays.
    ``wavelength_ranges`` is a sequence of open (lower, upper) Angstrom bounds
    in the output medium. Native samples are kept when ``pixels=None``; each
    region must then have equal length and identical samples across all nodes.
    An integer ``pixels`` explicitly prepares common linear sampling over the
    intersection of the selected node coverages, as in the old preparer.

    Row 0 is normalized flux; row 1 is unnormalized flux. Raw output is float32,
    matching frozen preparation. ``source_overrides`` maps atmosphere tuples
    to explicit FITS paths for known missing-model replacements/naming aliases;
    no replacement is applied silently. Paths are recorded in .sources.
    Returns an in-memory private spectral carrier; persist it once with
    save_spectral_grid and use load_spectral_grid in subsequent processes.
    """
    axes = _preparation_axes("coelho", axes)
    root = Path(data_dir).expanduser()
    paths = {}
    for node in product(*axes.values()):
        t, g, f, a = node
        filename = f"{_integer(t)}_{_integer(g * 10):02d}_{'p' if f >= 0 else 'm'}{_integer(abs(f) * 10):02d}p{_integer(a * 10):02d}.ms.fits"
        paths[node] = root / filename
    paths = _source_overrides(paths, source_overrides, root)
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
                          regions, wavelength_medium, kind, tuple(paths.values()))


def prepare_bosz(data_dir, *, axes, wavelength_ranges, pixels=None,
                 regions=None, wavelength_medium="vacuum", normalized=True,
                 source_overrides=None):
    """Prepare BOSZ-2024 gzip ASCII nodes (wavelength, H, continuum).

    Axes are teff/logg/mh/alpha/carbon/vmic. Normalized flux is H/continuum;
    ``normalized=False`` retains H. See prepare_coelho for region/pixel options.
    Filenames use mp for logg > 3, ms otherwise, and m±X.XX subdirectories.
    Missing files fail; source_overrides explicitly supports deliberate
    replacements instead of the old automatic logg=4 fallback.
    Native BOSZ wavelengths below 2000 Angstrom are already vacuum; above they
    are air. Vacuum conversion respects this convention. Air output requires
    selected regions above 2000 Angstrom.
    Returns an in-memory private spectral carrier for save_spectral_grid.
    """
    axes = _preparation_axes("bosz", axes)
    root = Path(data_dir).expanduser()
    paths = {}
    for node in product(*axes.values()):
        t, g, m, a, c, v = node
        for encoded in (g * 10, m * 100, a * 100, c * 100):
            _integer(encoded)
        geometry = "mp" if g > 3 else "ms"
        filename = f"bosz2024_{geometry}_t{_integer(t)}_g{g:+.1f}_m{m:+.2f}_a{a:+.2f}_c{c:+.2f}_v{_integer(v)}_rorig_noresam.txt.gz"
        paths[node] = root / f"m{m:+.2f}" / filename
    paths = _source_overrides(paths, source_overrides, root)
    _output_medium(wavelength_medium)
    bounds = _region_bounds(wavelength_ranges)
    if wavelength_medium == "air" and np.any(bounds[:, 0] < 2000):
        raise ValueError("BOSZ air regions must lie above 2000 Angstrom; use vacuum for UV")

    def reader(node, bounds_only=False):
        data = np.loadtxt(paths[node], ndmin=2, usecols=(0,) if bounds_only else (0, 1, 2))
        wav = data[:, 0]
        if wavelength_medium == "vacuum":
            wav = np.where(wav > 2000, _air_to_vacuum(wav), wav)
        if bounds_only:
            return wav, None
        flux = data[:, 1]
        if normalized:
            flux = np.divide(flux, data[:, 2], out=np.full_like(flux, np.nan),
                             where=data[:, 2] != 0)
        return wav, flux

    return _prepare_nodes("bosz", axes, reader, bounds, pixels, regions,
                          wavelength_medium, "normalized" if normalized else "unnormalized",
                          tuple(paths.values()))


def prepare_tlusty(data_dir, *, wavelength_ranges, axes=None, pixels=None,
                   regions=None, flux_kind="normalized", wavelength_medium="unknown",
                   source_overrides=None):
    """Prepare matched TLUSTY .7.gz spectra and .17.gz continua.

    Discover Z*_vt2/Z*_ostar directories, logZ=log10(Z/Z_sun), and teff/logg
    from filenames. Default axes retain logg > 2.9 and skip OSTAR teff <= 30000,
    as frozen preparation does. Explicit axes select a complete subgrid.
    Missing nodes fail rather than silently copying the nearest logg. Optional
    source_overrides map a target node to an explicit (spectrum, continuum) pair.

    flux_kind is normalized (continuum division), median_scaled (the legacy
    'absolute' option, scaled per selected region), or unnormalized (native H).
    No wavelength-medium conversion is performed; callers may declare a known
    medium. See prepare_coelho for native sampling versus explicit pixels.
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
                          regions, wavelength_medium, flux_kind, sources)


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


def _selected_regions(reader, node, bounds, *, bounds_only=False, median_scaled=False):
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
        selected = wav[keep]
        if selected.size == 0:
            raise ValueError(f"node {node}: empty wavelength region {(lower, upper)}")
        values = None if bounds_only else flux[keep]
        if not bounds_only:
            if not np.all(np.isfinite(values)):
                raise ValueError(f"node {node}: nonfinite flux or missing continuum coverage")
            if median_scaled:
                median = np.median(values)
                if not np.isfinite(median) or median == 0:
                    raise ValueError(f"node {node}: invalid median for flux scaling")
                values = values / median
        result.append((selected, values))
    return result


def _prepare_nodes(library, axes, reader, bounds, pixels, regions, medium, kind, sources):
    bounds = _region_bounds(bounds)
    nodes = tuple(product(*axes.values()))
    wavelength, flux = None, None
    if pixels is not None:
        if not isinstance(pixels, (int, np.integer)) or isinstance(pixels, bool) or pixels < 2:
            raise ValueError("pixels must be an integer >= 2, or None for native samples")
        # A bounds-only pass avoids retaining every native spectrum or building
        # large DataFrames. The second pass streams one node into the output.
        lower = np.full(len(bounds), -np.inf)
        upper = np.full(len(bounds), np.inf)
        for node in nodes:
            samples = _selected_regions(reader, node, bounds, bounds_only=True)
            lower = np.maximum(lower, [wav[0] for wav, _ in samples])
            upper = np.minimum(upper, [wav[-1] for wav, _ in samples])
        if np.any(upper <= lower):
            raise ValueError("nodes have no common wavelength interval for a region")
        wavelength = np.stack([np.linspace(lo, hi, pixels) for lo, hi in zip(lower, upper)])

    for flat_index, node in enumerate(nodes):
        samples = _selected_regions(reader, node, bounds, median_scaled=kind == "median_scaled")
        if wavelength is None:
            sizes = {wav.size for wav, _ in samples}
            if len(sizes) != 1:
                raise ValueError("native regions have different lengths; specify pixels or prepare separately")
            wavelength = np.stack([wav for wav, _ in samples])
        if flux is None:
            flux = np.empty(tuple(len(axis) for axis in axes.values()) + wavelength.shape,
                            dtype=np.float32)
        index = np.unravel_index(flat_index, flux.shape[:len(axes)])
        for region, (wav, values) in enumerate(samples):
            if pixels is None:
                if not np.array_equal(wav, wavelength[region]):
                    raise ValueError("native wavelength samples differ across nodes; specify pixels")
                flux[index + (region,)] = values
            else:
                flux[index + (region,)] = np.interp(wavelength[region], wav, values)
    return _assemble(library, axes, wavelength, flux, regions, medium, kind, sources)
