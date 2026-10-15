"""Portable spectral-fit results and descriptive model/grid provenance."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from jaxstar._arviz import (
    uses_datatree as _arviz_uses_datatree,
    is_inference_data as _is_inference_data,
    from_dict as _arviz_from_dict,
)
from .observation import Observation

__all__ = ["FitResult", "save_fit_result", "load_fit_result"]

_FORMAT = "jaxstar.specfit_result"
_VERSION = 1
_MANAGED = ("arrays.npz", "posterior.nc", "result.json")
_LATENT_NAMES = "jaxstar_specfit_latent_names"


def _describe(value):
    """Record settings, including priors, without serializing executable code."""
    import numpyro.distributions as dist

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration mapping keys must be strings")
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value]
    if isinstance(value, dist.Distribution):
        record = {
            "distribution": f"{type(value).__module__}.{type(value).__qualname__}",
            "parameters": {key: _describe(getattr(value, key))
                           for key in value.arg_constraints},
            "batch_shape": list(value.batch_shape), "event_shape": list(value.event_shape),
        }
        if hasattr(value, "base_dist"):
            record["base_dist"] = _describe(value.base_dist)
        if hasattr(value, "transforms"):
            record["transforms"] = [_describe(item) for item in value.transforms]
        return record
    if isinstance(value, partial):
        return {"callable": _describe(value.func), "args": _describe(value.args),
                "kwargs": _describe(value.keywords)}
    if callable(value):
        record = {
            "callable": f"{getattr(value, '__module__', '')}."
                        f"{getattr(value, '__qualname__', type(value).__qualname__)}",
            "note": "Implementation and closure are not serialized.",
        }
        try:
            record["source_sha256"] = sha256(inspect.getsource(value).encode()).hexdigest()
        except (OSError, TypeError):
            pass
        return record
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        return {"values": _describe(array.tolist()), "dtype": str(array.dtype),
                "shape": list(array.shape)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value), "note": "Object not serialized."}


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid_record(specmodel):
    spectra = specmodel.spectra
    grid = spectra.grid
    flux = grid.field("flux")
    schema = {"axis_names": list(grid.axis_names), "axis_kinds": list(grid.axis_kinds),
              "flux_dims": list(flux.dims), "payload_dims": list(flux.payload_dims),
              "boundary": grid.boundary}
    digest = sha256(json.dumps(schema, sort_keys=True).encode())
    for value in (*[grid.axis(name) for name in grid.axis_names],
                  flux.values, grid.fill_value, spectra.wavelength):
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return {
        **schema, "grid_sha256": digest.hexdigest(),
        "sources": [str(Path(path).expanduser().resolve()) for path in spectra.sources],
        "library": spectra.library, "regions": list(spectra.regions),
        "wavelength_medium": spectra.wavelength_medium,
        "wavelength_unit": spectra.wavelength_unit, "flux_kind": spectra.flux_kind,
        "axes": {name: _describe(grid.axis(name)) for name in grid.axis_names},
        "wavelength": _describe(spectra.wavelength),
        "velocity_grid": _describe(specmodel.velocity_grid),
        "flux_shape": list(flux.values.shape), "flux_dtype": str(flux.values.dtype),
        "vmax": specmodel.vmax, "broadening_operator": _describe(specmodel.broadening_operator),
        "fixed_ip": _describe(specmodel.fixed_ip),
    }


def _model_record(model, sf):
    from .configured_model import ConfiguredModel

    if not callable(model):
        raise TypeError("model must be callable")
    if isinstance(model, ConfiguredModel):
        model._check_specmodel(sf.specmodel)
        return {"function": _describe(model.function), "settings": _describe(model.settings),
                "reconstruct_fn": _describe(model.reconstruct_fn),
                "site_annotations": _describe(model.site_annotations)}
    return {"function": _describe(model),
            "note": "Recreate this callable explicitly to predict or refit."}


def _validate_posterior(idata):
    if not _is_inference_data(idata) or not hasattr(idata, "posterior"):
        raise ValueError("idata must be an ArviZ InferenceData or DataTree with a posterior group")
    posterior = idata.posterior
    if not posterior.data_vars or any(posterior.sizes.get(dim, 0) == 0
                                      for dim in ("chain", "draw")):
        raise ValueError("posterior must contain nonempty chain and draw dimensions")
    for name, values in posterior.data_vars.items():
        if not {"chain", "draw"}.issubset(values.dims) or values.dtype.kind not in "biuf":
            raise ValueError(f"posterior site {name!r} must have numeric chain/draw samples")


def _mcmc_to_inference_data(mcmc, *, regions=None):
    """Convert stored samples only; never recompute a likelihood or prediction."""
    from numpyro.infer import MCMC

    if not isinstance(mcmc, MCMC) or mcmc.last_state is None:
        raise ValueError("mcmc must be a completed NumPyro MCMC run")
    samples = {name: np.asarray(value)
               for name, value in mcmc.get_samples(group_by_chain=True).items()}
    renames = {"num_steps": "n_steps", "accept_prob": "acceptance_rate",
               "adapt_state.step_size": "step_size"}
    stats = {renames.get(name, name): np.asarray(value)
             for name, value in mcmc.get_extra_fields(group_by_chain=True).items()
             if not isinstance(value, (Mapping, tuple))}
    coords, dims = {}, {}
    if (regions is not None and "rv" in samples
            and samples["rv"].shape[2:] == (len(regions),)):
        coords["region"] = list(regions)
        if len({type(label) for label in regions}) > 1:
            coords["region"] = [str(label) for label in regions]
        dims["rv"] = ["region"]
    groups = {"posterior": samples}
    if stats:
        groups["sample_stats"] = stats
    idata = _arviz_from_dict(groups, coords=coords, dims=dims)
    if isinstance(mcmc.last_state.z, Mapping):
        # NetCDF-safe metadata keeps sampled sites distinguishable from fixed
        # and derived deterministic outputs, even when loading posterior.nc alone.
        idata.posterior.attrs[_LATENT_NAMES] = json.dumps(list(mcmc.last_state.z))
    return idata


def _mcmc_data(mcmc, sf, model):
    from numpyro.infer import MCMC

    if not isinstance(mcmc, MCMC) or mcmc.last_state is None:
        raise ValueError("mcmc must be a completed NumPyro MCMC run")
    bound = mcmc.sampler.model
    verified = False
    # SpecFit.run_nuts binds a snapshot of the effective observation and model.
    if (isinstance(bound, partial) and len(bound.args) == 2
            and isinstance(bound.args[0], Observation)):
        observation, specmodel = bound.args
        if bound.func is not model or specmodel is not sf.specmodel:
            raise ValueError("sf/model do not match the supplied MCMC run")
        current = sf.fit_observation()
        for name in ("wavelength", "flux", "uncertainty", "mask"):
            if not np.array_equal(getattr(observation, name), getattr(current, name),
                                  equal_nan=True):
                raise ValueError("observation or fit mask changed since the MCMC run")
        if observation.region != current.region or observation.exposure != current.exposure:
            raise ValueError("observation labels changed since the MCMC run")
        verified = True
    idata = _mcmc_to_inference_data(mcmc, regions=sf.regions)
    record = {"kind": "mcmc", "num_warmup": mcmc.num_warmup,
              "num_samples": mcmc.num_samples, "num_chains": mcmc.num_chains,
              "thinning": mcmc.thinning, "fit_state_verified": verified}
    if isinstance(mcmc.last_state.z, Mapping):
        record["latent_names"] = list(mcmc.last_state.z)
    return idata, record


@dataclass(frozen=True, eq=False)
class FitResult:
    """Loaded data and provenance, without importing or rebuilding a model.

    point is a native-site dictionary or None. idata retains chain/draw
    dimensions, sample statistics and any additional supplied ArviZ groups.
    samples flattens chain/draw for SpecFit.predict/plot_models, preserving
    each site's remaining dimensions. metadata describes model settings,
    grid provenance and the save-time environment; it is not executable config.
    """

    observation: Observation
    fit_mask: np.ndarray
    point: dict | None
    idata: object
    metadata: dict

    @property
    def samples(self):
        if self.idata is None:
            raise ValueError("no posterior samples were saved")
        _validate_posterior(self.idata)
        result = {}
        for name, data in self.idata.posterior.data_vars.items():
            trailing = [dim for dim in data.dims if dim not in ("chain", "draw")]
            values = np.asarray(data.transpose("chain", "draw", *trailing))
            result[name] = values.reshape((-1, *values.shape[2:]))
        return result


def save_fit_result(directory, *, sf, model, point=None, mcmc=None, idata=None,
                    metadata=None, overwrite=False):
    """Save arrays.npz, optional posterior.nc and result.json; return a Path.

    Supply a point and/or one of mcmc/idata. MCMC conversion preserves chains,
    draws and collected diagnostics without evaluating the model. A supplied
    InferenceData (ArviZ 0.x) or DataTree (1.x) is saved as-is.
    Observations and masks are captured from sf;
    for SpecFit MCMC runs they must still match the sampling-time state.

    Resolved model settings (including CCF-derived bounds), callable references,
    actual grid sampling, source paths and a grid fingerprint are recorded.
    Distributions/callables/custom objects are descriptive records, not objects
    to execute on reload. The grid payload and sampler continuation state are
    not stored. User metadata can record run labels, RNG seeds or other context.

    Existing managed files require overwrite=True; unrelated files are kept.
    A staged write and per-file hashes prevent loading a partial/mixed bundle.
    """
    import jax
    from . import SpecFit

    if not isinstance(sf, SpecFit):
        raise TypeError("sf must be a SpecFit")
    if mcmc is not None and idata is not None:
        raise ValueError("supply only one of mcmc or idata")
    if point is None and mcmc is None and idata is None:
        raise ValueError("supply point, mcmc or idata")
    directory = Path(directory).expanduser().resolve()
    if not overwrite and any((directory / name).exists() for name in _MANAGED):
        raise FileExistsError(f"result already exists in {directory}; use overwrite=True")
    model_record = _model_record(model, sf)
    arrays = {name: np.asarray(getattr(sf.observation, name))
              for name in ("wavelength", "flux", "uncertainty", "mask")}
    arrays["fit_mask"] = np.asarray(sf.fit_mask)
    point_names = None
    if point is not None:
        if not isinstance(point, Mapping) or not all(isinstance(name, str) for name in point):
            raise TypeError("point must be a mapping of native site names to numeric values")
        point_names = list(point)
        for i, (name, value) in enumerate(point.items()):
            value = np.asarray(value)
            if value.dtype.kind not in "biuf":
                raise TypeError(f"point site {name!r} must be numeric")
            arrays[f"point_{i}"] = value
    sampler = None
    if mcmc is not None:
        idata, sampler = _mcmc_data(mcmc, sf, model)
    elif idata is not None:
        sampler = {"kind": "inference_data", "fit_state_verified": False}
    if idata is not None:
        _validate_posterior(idata)
    packages = {}
    for name in ("jaxstar", "jax", "jaxlib", "numpy", "numpyro", "arviz"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    record = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "software": packages, "jax_enable_x64": bool(jax.config.x64_enabled),
        "model": model_record, "grid": _grid_record(sf.specmodel), "sampler": sampler,
        "ccf": None if sf.ccf is None else _describe({
            key: sf.ccf[key] for key in ("rv", "width", "region_rv")}),
        "user": _describe(metadata),
    }
    manifest = {
        "format": _FORMAT, "version": _VERSION, "point_names": point_names,
        "observation": {"region": sf.observation.region, "exposure": sf.observation.exposure},
        "metadata": record,
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".specfit-", dir=directory.parent) as temporary:
        stage = Path(temporary)
        np.savez(stage / "arrays.npz", **arrays)
        files = ["arrays.npz"]
        if idata is not None:
            idata.to_netcdf(stage / "posterior.nc", engine="h5netcdf")
            files.append("posterior.nc")
        manifest["files"] = {name: _file_hash(stage / name) for name in files}
        (stage / "result.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            os.replace(stage / name, directory / name)
        os.replace(stage / "result.json", directory / "result.json")
        if idata is None and overwrite:
            (directory / "posterior.nc").unlink(missing_ok=True)
    return directory


def load_fit_result(directory):
    """Load a complete saved bundle into FitResult, independent of grid files.

    Numerical arrays use pickle-free NPZ; posterior.nc is eagerly loaded and
    closed. Loading never reconstructs/imports saved model callables or priors.
    Recreate the model explicitly and restore fit_mask before prediction.
    """
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != _FORMAT or type(manifest.get("version")) is not int
            or manifest["version"] != _VERSION):
        raise ValueError("unsupported spectral-fit result format")
    files = manifest.get("files", {})
    if (not isinstance(files, dict) or "arrays.npz" not in files
            or set(files) - {"arrays.npz", "posterior.nc"}):
        raise ValueError("invalid spectral-fit result file manifest")
    for name, checksum in files.items():
        if _file_hash(directory / name) != checksum:
            raise ValueError(f"{name} does not match result.json")
    names = manifest.get("point_names")
    if names is not None and (not isinstance(names, list)
                              or not all(isinstance(name, str) for name in names)
                              or len(set(names)) != len(names)):
        raise ValueError("invalid saved point site names")
    with np.load(directory / "arrays.npz", allow_pickle=False) as data:
        observation = Observation(
            data["wavelength"], data["flux"], data["uncertainty"], data["mask"],
            **manifest["observation"])
        fit_mask = np.array(data["fit_mask"], copy=True)
        if fit_mask.dtype.kind != "b" or fit_mask.shape != observation.shape:
            raise ValueError("saved fit mask must be Boolean with the observation shape")
        fit_mask.setflags(write=False)
        point = None
        if names is not None:
            point = {}
            for i, name in enumerate(names):
                value = np.array(data[f"point_{i}"], copy=True)
                if value.dtype.kind not in "biuf":
                    raise ValueError(f"saved point site {name!r} must be numeric")
                value.setflags(write=False)
                point[name] = value
    idata = None
    if "posterior.nc" in files:
        import arviz as az
        idata = az.from_netcdf(directory / "posterior.nc", engine="h5netcdf")
        try:
            idata.load()
        finally:
            idata.close()
        _validate_posterior(idata)
    return FitResult(observation, fit_mask, point, idata, manifest["metadata"])
