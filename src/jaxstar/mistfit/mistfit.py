"""Convenience interface combining a MistModel with fixed observations."""

# Preserve the previous import paths; implementations live in inference.py.
__all__ = ["MistObservations", "MistFit", "model_isochrone", "model_track", "find_initial", "make_init_strategy",
           "run_nuts", "run_multinest"]

from collections.abc import Mapping
from dataclasses import dataclass
from numpyro.infer import init_to_value

from .model import MistModel
from .inference import (MistObservations, model_isochrone, model_track, find_initial, make_init_strategy,
                        run_nuts, run_multinest, smbound, _normalize_output_keys, _default_model)


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
        """Default NumPyro model selected by the physical model's kind."""
        return _default_model(self.physical_model)(self.observations, self.physical_model, *args, **kwargs)

    def find_initial(self, *, feh_init=None, distance=None, model_kwargs=None, return_grid=False,
                     feh_step=None):
        """Return latent starting values, optionally scanning initial FeH.

        Pass the same model_kwargs as the intended run. Track searches use
        native mass x EEP; isochrone searches use native log-age x EEP.
        FeH defaults to the observed surface value, or zero if absent.
        feh_step=0.1 scans the allowed initial FeH range at 0.1 dex intervals,
        also including this starting value when supported. Distance stays fixed.
        return_grid=True also returns the search for optional coverage checks.
        See inference.find_initial for scoring, bounds and distance defaults.
        """
        return find_initial(self.observations, self.physical_model,
                            feh_init=feh_init, distance=distance, model_kwargs=model_kwargs,
                            return_grid=return_grid, feh_step=feh_step)

    def run_nuts(self, rng_key, *, output_keys=None, model_kwargs=None,
                 grid_init=True, init_kwargs=None, init_strategy=None, **sampler_kwargs):
        """Return a new MCMC result (four chains by default).

        output_keys adds deterministic outputs for this run. model_kwargs holds
        prior/likelihood options; sampler_kwargs holds run_nuts sampler options.
        By default, search once with find_initial and use that start for all
        chains. init_kwargs can set feh_init, distance and feh_step for this search only.
        grid_init=False retains the historical parallax/NumPyro fallback rule.
        Explicit init_strategy or init_params bypass automatic initialization
        (including init_kwargs), and do not require a parallax observation.
        """
        options = _model_options(model_kwargs, output_keys)
        if init_strategy is None and sampler_kwargs.get("init_params") is None:
            if grid_init:
                initial = self.find_initial(model_kwargs=options, **dict(init_kwargs or {}))
                init_strategy = init_to_value(values=initial)
            else:
                init_strategy = make_init_strategy(self.observations)
        return run_nuts(self.model, rng_key, init_strategy=init_strategy,
                        model_kwargs=options, **sampler_kwargs)

    # Retain the familiar name with the new single-call/result-returning API.
    run_hmc = run_nuts

    def check_missed_modes(self, result, *, feh_step=0.1, feh_init=None, distance=None,
                            threshold=5., min_samples=5, neighbor_cells=1,
                            max_unvisited_fraction=0.1, batch_size=512):
        """Scan independently of NUTS initialization for potentially missed modes.

        result is a jaxstar-recorded MCMC result or saved InferenceData. Recover
        the exact scientific options from its run metadata and require matching
        observations and physical grid. No model_kwargs need to be repeated.
        By default scan FeH at 0.1 dex, including the observed FeH guess, with
        distance fixed to inverse observed parallax. feh_init and distance
        override the scan's guess/slice only; feh_step=None fixes FeH.

        The returned dict includes grid_search (the full InitialGridSearch),
        sample scores/counts and region flags. Use recheck_missed_modes to
        change diagnostic thresholds without scanning or scoring again.
        Sampling and this immutable fitter are unchanged. See the standalone
        check_missed_modes for flag definitions and projection limitations.
        """
        from .diagnostics import _check_fit_missed_modes
        return _check_fit_missed_modes(
            self, result, feh_step=feh_step, feh_init=feh_init, distance=distance,
            threshold=threshold, min_samples=min_samples, neighbor_cells=neighbor_cells,
            max_unvisited_fraction=max_unvisited_fraction, batch_size=batch_size,
        )

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

    def plot_track(self, mass_init, feh_init, **kwargs):
        """Plot a track at fixed initial mass/FeH using this fitter's model."""
        from .plotting import plot_track
        return plot_track(self.physical_model, mass_init, feh_init, **kwargs)

    def plot_posterior_tracks(self, result, **kwargs):
        """Plot joint posterior tracks/locations with the result's observations."""
        from .plotting import plot_posterior_tracks
        return plot_posterior_tracks(result, self.physical_model, **kwargs)

    def plot_diagnostics(self, result, **kwargs):
        """Return diagnostic Figures, including tracks for observed track photometry."""
        from .plotting import plot_diagnostics
        return plot_diagnostics(result, model=self.physical_model, **kwargs)
