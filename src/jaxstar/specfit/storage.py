"""Common prepared spectral-grid NPZ storage, independent of raw libraries."""

import json
from pathlib import Path

import numpy as np

from jaxstar.grid import Axis, Field, RectilinearGrid
from ._data import _SpectralLibrary


_FORMAT = "jaxstar.spectral_grid"
_VERSION = 1
_METADATA_KEYS = {
    "format", "version", "axis_names", "axis_kinds", "flux_dims",
    "payload_dims", "boundary", "regions", "library", "wavelength_medium",
    "wavelength_unit", "flux_kind", "sources",
}


def save_spectral_grid(path, spectra, *, overwrite=False):
    """Persist a loader/preparer result in the common, library-neutral format.

    This offline step saves the numerical arrays currently held by ``spectra``
    without conversion, normalization or resampling. All axes and their order,
    flux dimensions, wavelengths, region identifiers and interpretation metadata
    are retained. Source paths are provenance only, never runtime dependencies.

    Uses uncompressed NPZ without pickle or a raw-library-specific schema. The
    exact path is used (no extension is appended); returns that path as a Path.
    Existing files require ``overwrite=True``. Save after preparation once, then
    use ``load_spectral_grid`` in subsequent processes without the raw files.
    """
    if not isinstance(spectra, _SpectralLibrary):
        raise TypeError("spectra must be a spectral-library loader/preparer result")
    path = Path(path).expanduser()
    grid = spectra.grid
    field = grid.field("flux")
    metadata = {
        "format": _FORMAT,
        "version": _VERSION,
        "axis_names": grid.axis_names,
        "axis_kinds": grid.axis_kinds,
        "flux_dims": field.dims,
        "payload_dims": field.payload_dims,
        "boundary": grid.boundary,
        "regions": spectra.regions,
        "library": spectra.library,
        "wavelength_medium": spectra.wavelength_medium,
        "wavelength_unit": spectra.wavelength_unit,
        "flux_kind": spectra.flux_kind,
        "sources": spectra.sources,
    }
    arrays = {f"axis_{i}": np.asarray(grid.axis(name))
              for i, name in enumerate(grid.axis_names)}
    arrays.update(
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)),
        flux=np.asarray(field.values), wavelength=np.asarray(spectra.wavelength),
        fill_value=np.asarray(grid.fill_value),
    )
    with path.open("wb" if overwrite else "xb") as stream:
        np.savez(stream, **arrays)
    return path


def load_spectral_grid(path):
    """Load a common prepared artifact, without consulting any source library.

    Only the versioned format written by ``save_spectral_grid`` is accepted.
    For legacy jaxspec NPZs use ``load_coelho``, ``load_bosz`` or ``load_tlusty``
    once, then save their result in the common format if desired.

    Returns the same private immutable carrier as raw preparers. Numerical
    interpolation belongs to ``result.grid``. No wavelength transformation or
    continuum division occurs. Normal JAX dtype canonicalization applies; use
    the same x64 setting to retain saved float64 precision across processes.
    """
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as data:
        metadata = _read_metadata(data)
        names = metadata["axis_names"]
        required = {f"axis_{i}" for i in range(len(names))} | {
            "flux", "wavelength", "fill_value",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing prepared spectral-grid arrays {sorted(missing)}")
        grid = RectilinearGrid(
            axes={name: Axis(data[f"axis_{i}"], kind)
                  for i, (name, kind) in enumerate(zip(names, metadata["axis_kinds"]))},
            fields={"flux": Field(data["flux"], metadata["flux_dims"],
                                  payload_dims=metadata["payload_dims"])},
            boundary=metadata["boundary"], fill_value=data["fill_value"],
        )
        return _SpectralLibrary(
            grid=grid, wavelength=data["wavelength"], regions=metadata["regions"],
            library=metadata["library"], wavelength_medium=metadata["wavelength_medium"],
            flux_kind=metadata["flux_kind"], sources=metadata["sources"],
            wavelength_unit=metadata["wavelength_unit"],
        )


def _read_metadata(data):
    if "metadata" not in data.files:
        raise ValueError("not a common prepared spectral grid; use library-specific loaders "
                         "for legacy jaxspec NPZs")
    encoded = data["metadata"]
    if encoded.shape != () or encoded.dtype.kind != "U":
        raise ValueError("prepared spectral-grid metadata must be a scalar JSON string")
    try:
        metadata = json.loads(encoded.item())
    except json.JSONDecodeError as error:
        raise ValueError("invalid prepared spectral-grid JSON metadata") from error
    if not isinstance(metadata, dict) or metadata.get("format") != _FORMAT:
        raise ValueError("not a common prepared spectral-grid format")
    if type(metadata.get("version")) is not int or metadata["version"] != _VERSION:
        raise ValueError(f"unsupported prepared spectral-grid version {metadata.get('version')!r}")
    missing = _METADATA_KEYS - metadata.keys()
    if missing:
        raise ValueError(f"missing prepared spectral-grid metadata {sorted(missing)}")
    for key in ("axis_names", "axis_kinds", "flux_dims", "payload_dims", "sources"):
        if not isinstance(metadata[key], list) or any(not isinstance(v, str) for v in metadata[key]):
            raise ValueError(f"prepared spectral-grid {key} must be a list of strings")
    names = metadata["axis_names"]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("prepared spectral-grid axis names must be non-empty and unique")
    if len(metadata["axis_kinds"]) != len(names):
        raise ValueError("one prepared spectral-grid axis kind is required per axis")
    if not isinstance(metadata["regions"], list):
        raise ValueError("prepared spectral-grid regions must be a list")
    for key in ("boundary", "library", "wavelength_medium", "wavelength_unit", "flux_kind"):
        if not isinstance(metadata[key], str):
            raise ValueError(f"prepared spectral-grid {key} must be a string")
    return metadata
