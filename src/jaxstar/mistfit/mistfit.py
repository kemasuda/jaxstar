"""Convenience interface combining a MistModel with fixed observations."""

# Preserve the previous import paths; implementations live in inference.py.
__all__ = ["MistObservations", "MistFit", "model_isochrone", "make_init_strategy",
           "run_nuts", "run_multinest"]

from collections.abc import Mapping
from dataclasses import dataclass

from .model import MistModel
from .inference import (MistObservations, model_isochrone, make_init_strategy,
                        run_nuts, run_multinest, smbound, _normalize_output_keys)


def _model_options(model_kwargs, output_keys):
    """Copy model options and merge the per-run additional deterministic keys."""
    options = dict(model_kwargs or {})
    options["output_keys"] = tuple(dict.fromkeys((
        *_normalize_output_keys(options.get("output_keys")),
        *_normalize_output_keys(output_keys),
    )))
    return options


@dataclass(frozen=True, init=False)
class MistFit:
    """A physical model and fixed observations, reusable across independent runs.

    Construct with ``MistFit(observations={"teff": (5700., 70.), ...}, model=model)``
    or supply a path to load an isochrone model. Changing data means constructing
    another fit, reusing the same MistModel. Each sampler call returns its own
    native result; this object has no mutable sampler/sample state.
    """

    physical_model: MistModel
    observations: MistObservations

    def __init__(self, path=None, *, observations, model=None, legacy_interpolation=None):
        if isinstance(observations, Mapping):
            observations = MistObservations.from_mapping(observations)
        if not isinstance(observations, MistObservations):
            raise TypeError("observations must be MistObservations or a mapping of (value, error) pairs")
        if model is not None:
            if not isinstance(model, MistModel):
                raise TypeError("model must be a MistModel")
            if path is not None or legacy_interpolation is not None:
                raise ValueError("model cannot be combined with path or legacy_interpolation")
        else:
            model = MistModel.from_isochrones(
                path, legacy_interpolation=False if legacy_interpolation is None else legacy_interpolation,
            )
        unknown = set(observations.keys) - set(model.output_keys)
        if unknown:
            raise ValueError(f"unknown MistModel observation keys: {sorted(unknown)}")
        object.__setattr__(self, "physical_model", model)
        object.__setattr__(self, "observations", observations)

    def model(self, *args, **kwargs):
        """NumPyro callable delegating to model_isochrone with this fit's data."""
        return model_isochrone(self.observations, self.physical_model, *args, **kwargs)

    def run_nuts(self, rng_key, *, output_keys=None, model_kwargs=None,
                 init_strategy=None, **sampler_kwargs):
        """Return a new MCMC result (four chains by default).

        output_keys adds deterministic outputs for this run. model_kwargs holds
        prior/likelihood options; sampler_kwargs holds run_nuts sampler options.
        Initialization is unchanged unless an explicit init_strategy is supplied.
        """
        if init_strategy is None:
            init_strategy = make_init_strategy(self.observations)
        return run_nuts(self.model, rng_key, init_strategy=init_strategy,
                        model_kwargs=_model_options(model_kwargs, output_keys), **sampler_kwargs)

    # Retain the familiar name with the new single-call/result-returning API.
    run_hmc = run_nuts

    def run_multinest(self, *, outputfiles_basename, output_keys=None, model_kwargs=None,
                      **sampler_kwargs):
        """Return a MultiNestRunner; scientific options match run_nuts exactly."""
        return run_multinest(self.model, outputfiles_basename=outputfiles_basename,
                             model_kwargs=_model_options(model_kwargs, output_keys), **sampler_kwargs)

    def to_inference_data(self, result):
        """Convert saved draws/provenance to portable ArviZ InferenceData."""
        from .results import to_inference_data
        return to_inference_data(result)

    def save_result(self, result, directory, **kwargs):
        """Save posterior, run record and summary without mutating this fitter."""
        from .results import save_result
        return save_result(result, directory, **kwargs)

    def plot_trace(self, result, **kwargs):
        """Return a chain trace/marginal Figure using the result's observations."""
        from .plotting import plot_trace
        return plot_trace(result, **kwargs)

    def plot_corner(self, result, **kwargs):
        """Return a correlation Figure using the result's observations."""
        from .plotting import plot_corner
        return plot_corner(result, **kwargs)

    def plot_diagnostics(self, result, **kwargs):
        """Return diagnostic Figures; optionally save them in output_dir."""
        from .plotting import plot_diagnostics
        return plot_diagnostics(result, **kwargs)
