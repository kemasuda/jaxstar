"""Portable inference results, run provenance and persistence (no grid on reload)."""

__all__ = ["to_inference_data", "get_run_metadata", "summarize", "save_result", "load_result"]

from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
import inspect
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

_FORMAT_VERSION = 1
_METADATA_ATTR = "jaxstar_run_metadata"
# MultiNest's documented output suffixes, including INS and resume state. Never
# glob the whole directory: a caller can legitimately use an empty basename.
_NATIVE_SUFFIXES = (
    ".txt", "post_equal_weights.dat", "post_separate.dat", "stats.dat",
    "summary.txt", "resume.dat", "ev.dat", "live.points", "phys_live.points",
    "IS.iterinfo", "IS.points", "IS.ptprob", "dead-birth.txt",
    "phys_live-birth.txt", "params.json",
)


def _json_value(value):
    """Describe configuration as JSON without pickling executable objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, partial):
        return {"callable": _json_value(value.func), "args": _json_value(value.args),
                "kwargs": _json_value(value.keywords)}
    if callable(value):
        return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}",
                "note": "Callable implementation/closure is not serialized."}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        return {"values": _json_value(array.tolist()), "dtype": str(array.dtype), "shape": list(array.shape)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "note": "Object not serialized; supply an external description if needed."}


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid_record(model):
    """Fingerprint the actual in-memory grid, including custom grids/precision."""
    import jax
    digest = sha256()
    schema = {"axes": list(model.grid.axis_names), "axis_kinds": list(model.grid.axis_kinds),
              "fields": {k: list(model.grid.field(k).dims) for k in model.grid.field_names}}
    digest.update(json.dumps(schema, sort_keys=True).encode())
    for leaf in jax.tree_util.tree_leaves(model.data):
        array = np.ascontiguousarray(leaf)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return {"kind": model.kind, "legacy_interpolation": model.legacy_interpolation,
            "grid_sha256": digest.hexdigest(), "grid_schema": schema,
            "sources": [{"path": str(path), "sha256": _file_hash(path) if Path(path).is_file() else None}
                        for path in model.data.sources]}


def _model_record(model, args, kwargs):
    from .inference import MistObservations, _default_model
    from .model import MistModel

    owner = getattr(model, "__self__", None)
    if isinstance(getattr(owner, "physical_model", None), MistModel):
        args = (owner.observations, owner.physical_model, *args)
        model = _default_model(owner.physical_model)
    try:
        bound = inspect.signature(model).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
    except (ValueError, TypeError):
        arguments = {"args": args, "kwargs": kwargs}
    record = {"callable": _json_value(model), "arguments": {}}
    for key, value in arguments.items():
        if isinstance(value, MistModel):
            record["physical_model"] = _grid_record(value)
        elif isinstance(value, MistObservations):
            record["observations"] = {"keys": list(value.keys), "values": list(value.values),
                                      "errors": list(value.errors)}
        else:
            record["arguments"][key] = _json_value(value)
    return _json_value(record)


def _new_run_record(sampler, model, args, kwargs):
    import jax
    packages = {}
    for name in ("jaxstar", "jax", "jaxlib", "numpyro", "numpy", "arviz", "numpyro-inferutils", "pymultinest"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(source_root)): _file_hash(path) for path in (
        source_root / "mistfit" / "model.py", source_root / "mistfit" / "inference.py",
        source_root / "mistfit" / "_data.py", source_root / "mistfit" / "mistgrid" / "tracks.py",
        source_root / "mistfit" / "gyrochrone_likelihood.py", source_root / "grid" / "core.py",
    ) if path.is_file()}
    return {"format_version": _FORMAT_VERSION, "sampler": sampler, "source_sha256": source_hashes,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "software": packages, "jax_enable_x64": jax.config.x64_enabled,
            "devices": [str(device) for device in jax.local_devices()],
            "model": _model_record(model, args, kwargs)}


def _finish_run_record(result, record):
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    # Store a detached snapshot on the native result, never on MistFit.
    setattr(result, "_jaxstar_run_metadata", json.loads(json.dumps(_json_value(record), allow_nan=False)))


def get_run_metadata(result):
    """Return a detached JSON-compatible run record from a result or InferenceData."""
    if hasattr(result, "attrs") and _METADATA_ATTR in result.attrs:
        return json.loads(result.attrs[_METADATA_ATTR])
    record = getattr(result, "_jaxstar_run_metadata", None)
    if record is not None:
        return json.loads(json.dumps(record))
    # External samplers can still be plotted/saved, but never invent lost RNG/configuration.
    return {"format_version": _FORMAT_VERSION, "metadata_complete": False,
            "note": "Run was not recorded by jaxstar; original RNG/configuration may be unavailable."}


def _quantity_info(key):
    known = {
        "age": ("Age", "Gyr"), "logage": ("log10(age / yr)", ""),
        "mass": ("Initial mass", "M_sun"), "mass_init": ("Initial mass", "M_sun"),
        "star_mass": ("Current mass", "M_sun"), "mass_current": ("Current mass", "M_sun"),
        "radius": ("Radius", "R_sun"), "teff": ("Teff", "K"),
        "logg": ("log10(g / cm s^-2)", ""), "distance": ("Distance", "kpc"),
        "parallax": ("Parallax", "mas"), "eep": ("EEP", ""),
        "feh_init": ("Initial [Fe/H]", "dex"), "feh": ("Surface [Fe/H]", "dex"),
        "feh_surface": ("Surface [Fe/H]", "dex"), "feh_photosphere": ("Surface [Fe/H]", "dex"),
        "rho": ("Density (current mass)", "rho_sun"),
        "rho_initial": ("Density (initial mass)", "rho_sun"),
        "dmdeep": ("d(initial mass)/dEEP", "M_sun / EEP"),
        "dtdeep": ("d(age)/dEEP", "Gyr / EEP"),
    }
    if key in known:
        name, units = known[key]
    elif "mag" in key:
        name, units = (("Absolute " if key.endswith("_abs") else "Apparent ") + key.removesuffix("_abs"), "mag")
    else:
        name, units = key, ""
    return {"long_name": name, "units": units, "label": name + (f" [{units}]" if units else "")}


def _multinest_samples(runner):
    if runner.result is None or "samples" not in runner.result:
        raise ValueError("MultiNestRunner must have a completed run with equal-weight samples")
    array = np.asarray(runner.result["samples"])
    if array.ndim != 2 or array.shape[1] != runner.ndim:
        raise ValueError("unexpected MultiNest equal-weight sample shape")
    samples, start = {}, 0
    for site in runner.site_info:
        size = site["size"]
        samples[site["name"]] = array[:, start:start + size].reshape((len(array),) + tuple(site["shape"]))
        start += size
    samples.update(runner.get_deterministic_samples(samples, site_names=runner.deterministic_site_names))
    return {key: np.asarray(value)[None, ...] for key, value in samples.items()}


def to_inference_data(result):
    """Convert native MCMC/MultiNest results, or return existing InferenceData.

    MCMC conversion only copies saved arrays. MultiNest additionally evaluates
    its registered deterministic sites once per conversion. Nested samples have
    one storage chain, explicitly marked as equal-weight, non-MCMC draws.
    No pointwise likelihood is recomputed (the model contains prior factors).
    """
    import arviz as az
    from numpyro.infer import MCMC

    if isinstance(result, az.InferenceData):
        if not hasattr(result, "posterior"):
            raise ValueError("InferenceData must contain a posterior group")
        return result
    record = get_run_metadata(result)
    stats = None
    if isinstance(result, MCMC):
        samples = result.get_samples(group_by_chain=True)
        rename = {"num_steps": "n_steps", "accept_prob": "acceptance_rate", "adapt_state.step_size": "step_size"}
        stats = {rename.get(key, key): np.asarray(value)
                 for key, value in result.get_extra_fields(group_by_chain=True).items()
                 if not isinstance(value, (tuple, dict))}
        record.setdefault("sampler", "mcmc")
        if isinstance(result.last_state.z, Mapping):
            record.setdefault("latent_names", list(result.last_state.z))
    elif hasattr(result, "site_info") and hasattr(result, "get_deterministic_samples"):
        samples = _multinest_samples(result)
        record["sampler"] = "multinest"
        record["posterior_weighting"] = "equal"
        record["latent_names"] = list(result.site_names)
        record["evidence"] = {key: _json_value(result.result.get(key)) for key in ("logZ", "logZerr")}
        if result.outputfiles_basename is not None:
            record.setdefault("native_output_prefix", os.path.abspath(result.outputfiles_basename))
            # abspath removes the trailing separator, which is meaningful for an empty basename.
            if str(result.outputfiles_basename).endswith(os.sep):
                record["native_output_prefix"] = record["native_output_prefix"].rstrip(os.sep) + os.sep
    else:
        raise TypeError("expected NumPyro MCMC, MultiNestRunner or ArviZ InferenceData")
    observed = record.get("model", {}).get("observations")
    extra = {}
    if observed:
        extra = dict(observed_data={"value": np.asarray(observed["values"])},
                     constant_data={"error": np.asarray(observed["errors"])},
                     coords={"observable": observed["keys"]},
                     dims={"value": ["observable"], "error": ["observable"]})
    idata = az.from_dict(posterior={k: np.asarray(v) for k, v in samples.items()}, sample_stats=stats, **extra)
    idata.attrs[_METADATA_ATTR] = json.dumps(record, allow_nan=False, sort_keys=True)
    for key in idata.posterior:
        idata.posterior[key].attrs.update(_quantity_info(key))
    if stats and "potential_energy" in stats:
        idata.sample_stats["potential_energy"].attrs["long_name"] = "Negative log density in unconstrained coordinates"
    return idata


def summarize(result, *, keys=None, hdi_prob=0.68):
    """Posterior summary; chain diagnostics are omitted for MultiNest results."""
    import arviz as az
    idata = to_inference_data(result)
    # xarray treats a tuple as one compound variable name, not a selection.
    names = None if keys is None else ([keys] if isinstance(keys, str) else list(keys))
    nested = get_run_metadata(idata).get("sampler") == "multinest"
    return az.summary(idata, var_names=names, hdi_prob=hdi_prob,
                      kind="stats" if nested else "all", round_to="none")


def _native_files(prefix):
    return [Path(prefix + suffix) for suffix in _NATIVE_SUFFIXES if Path(prefix + suffix).is_file()]


def save_result(result, directory, *, overwrite=False, hdi_prob=0.68, copy_native=True):
    """Save posterior.nc, run.json and summary.csv; return the output directory.

    Existing managed files require overwrite=True. MultiNest output files stay
    alongside the saved data and figures. Files already in directory are kept
    in place; saving elsewhere copies them into that directory. InferenceData
    loaded from a bundle can also be saved elsewhere.
    Figures are saved separately by plot_diagnostics. No grid or pickle is saved.
    """
    idata = to_inference_data(result)
    record = get_run_metadata(idata)
    directory = Path(directory).expanduser().resolve()
    targets = ("posterior.nc", "run.json", "summary.csv")
    if not overwrite and any((directory / name).exists() for name in targets):
        raise FileExistsError(f"result already exists in {directory}; use overwrite=True or another directory")
    native = []
    if copy_native and record.get("sampler") == "multinest":
        archive = idata.attrs.get("jaxstar_native_archive")
        if archive:
            native = [Path(archive) / name for name in json.loads(idata.attrs.get("jaxstar_native_files", "[]"))]
            missing = [str(p) for p in native if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"missing archived MultiNest files: {missing}")
        elif record.get("native_output_prefix"):
            prefix = record["native_output_prefix"]
            if "native_files" in record:
                native = [Path(os.path.dirname(prefix)) / name for name in record["native_files"]]
            else:
                native = _native_files(prefix)
        for source in native:
            if not source.is_file():
                raise FileNotFoundError(f"missing MultiNest file: {source}; use copy_native=False to save draws only")
            expected = record.get("native_files", {}).get(source.name)
            if expected is not None and _file_hash(source) != expected:
                raise ValueError(f"MultiNest file changed since this run: {source}; use separate prefixes or copy_native=False")
            destination = directory / source.name
            if destination.exists() and destination.resolve() != source.resolve() and not overwrite:
                raise FileExistsError(f"MultiNest output exists: {destination}; use overwrite=True")
    directory.parent.mkdir(parents=True, exist_ok=True)
    table = summarize(idata, hdi_prob=hdi_prob)
    with TemporaryDirectory(prefix=".mistfit-", dir=directory.parent) as temporary:
        stage = Path(temporary)
        stored = idata.copy()
        # Paths used to copy an already-loaded archive are runtime-only, not persisted.
        stored.attrs.pop("jaxstar_native_archive", None)
        stored.attrs.pop("jaxstar_native_files", None)
        stored.to_netcdf(stage / "posterior.nc", engine="h5netcdf")
        table.to_csv(stage / "summary.csv", index_label="parameter")
        paths = [Path("posterior.nc"), Path("summary.csv")]
        checksums = {str(p): {"sha256": _file_hash(stage / p)} for p in paths}
        for source in native:
            relative = Path(source.name)
            if source.resolve() == (directory / relative).resolve():
                checksums[str(relative)] = {"sha256": _file_hash(source)}
            else:
                shutil.copy2(source, stage / relative)
                paths.append(relative)
                checksums[str(relative)] = {"sha256": _file_hash(stage / relative)}
        manifest = {"format_version": _FORMAT_VERSION, "run": record,
                    "summary": {"hdi_prob": hdi_prob}, "copy_native": copy_native,
                    "native_files": [source.name for source in native],
                    "files": checksums}
        (stage / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        directory.mkdir(parents=True, exist_ok=True)
        for relative in paths:
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative, directory / relative)
        os.replace(stage / "run.json", directory / "run.json")
    return directory


def load_result(path):
    """Load a saved directory (with integrity check) or a standalone posterior.nc.

    Returns fully loaded InferenceData, independent of the grid, fitter and
    native samplers. Read provenance with get_run_metadata(result).
    """
    import arviz as az
    path = Path(path).expanduser().resolve()
    manifest = None
    if path.is_dir():
        manifest = json.loads((path / "run.json").read_text())
        if manifest.get("format_version") != _FORMAT_VERSION:
            raise ValueError("unsupported MistFit result format")
        filename = path / "posterior.nc"
        if _file_hash(filename) != manifest["files"]["posterior.nc"]["sha256"]:
            raise ValueError("posterior.nc does not match run.json")
    else:
        filename = path
    idata = az.from_netcdf(filename, engine="h5netcdf")
    try:
        idata.load()
    finally:
        idata.close()
    if not hasattr(idata, "posterior"):
        raise ValueError("saved result has no posterior")
    if manifest:
        if get_run_metadata(idata) != manifest["run"]:
            raise ValueError("posterior metadata does not match run.json")
        native = manifest.get("native_files")
        archive = path
        if native is None:
            # Read bundles written before MultiNest files were saved flat.
            native = [Path(p).name for p in manifest["files"] if p.startswith("multinest/")]
            archive = path / "multinest"
        if native:
            idata.attrs["jaxstar_native_archive"] = str(archive)
            idata.attrs["jaxstar_native_files"] = json.dumps(native)
    return idata
