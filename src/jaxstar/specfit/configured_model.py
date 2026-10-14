"""A configured NumPyro callable and its explicitly paired reconstruction.

Custom models remain ordinary functions of (observation, specmodel, **settings).
This small wrapper binds settings, never a fit mask or an inference result.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import Parameter, signature
from types import MappingProxyType

import jax
import numpy as np
from numpyro import handlers
import numpyro.distributions as dist


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
    elif isinstance(value, (list, tuple)):
        value = tuple(_freeze(item) for item in value)
    return value


def _format_value(value):
    return np.array2string(np.asarray(value), separator=", ", threshold=12,
                          max_line_width=1000, formatter={"float_kind": lambda x: f"{x:.6g}"})


def _format_prior(prior):
    while isinstance(prior, (dist.Independent, dist.ExpandedDistribution)):
        prior = prior.base_dist
    name = type(prior).__name__
    parameters = {key: getattr(prior, key) for key in prior.arg_constraints}
    if isinstance(prior, (dist.LeftTruncatedDistribution, dist.RightTruncatedDistribution,
                          dist.TwoSidedTruncatedDistribution)):
        base = prior.base_dist
        name = f"Truncated{type(base).__name__}"
        parameters = {**{key: getattr(base, key) for key in base.arg_constraints}, **parameters}
    positional = isinstance(prior, (dist.Uniform, dist.LogUniform, dist.Normal))
    arguments = ", ".join(("" if positional else f"{key}=") + _format_value(value)
                          for key, value in parameters.items())
    return f"{name}({arguments})"


@dataclass(frozen=True, eq=False, init=False)
class ConfiguredModel:
    """Bind a model function, settings and optional reconstruction callback.

    ``function(observation, specmodel, **settings)`` is an ordinary NumPyro
    model. Inference accepts this wrapper or a plain configured callable.
    ``reconstruct_fn(observation, specmodel, returned, trace, settings)`` uses
    the model's return value and replayed sites to reconstruct a supplied point.
    No continuum or likelihood convention is imposed on custom callbacks.

    For check_prior, supply an observation and specmodel and implement the
    function's ``prior_only=True`` path, recording all parameter sites without
    evaluating spectra or likelihoods. ``site_annotations`` supplies only
    display metadata: depends_on (site names) and expression (text). Conditional
    sample sites can instead attach these through ``infer['prior']``.
    Keyword defaults from the function's signature are resolved at construction,
    with explicit settings taking precedence. Settings and annotations are
    copied and made read-only. Rebuild a model to change them; inference and
    reconstruction always use the same settings.
    """

    function: object
    settings: Mapping
    reconstruct_fn: object
    observation: object
    specmodel: object
    site_annotations: Mapping

    def __init__(self, function, *, settings=None, reconstruct_fn=None,
                 observation=None, specmodel=None, site_annotations=None):
        if not callable(function):
            raise TypeError("function must be a NumPyro model callable")
        if reconstruct_fn is not None and not callable(reconstruct_fn):
            raise TypeError("reconstruct_fn must be callable")
        model_signature = signature(function)
        inputs = model_signature.bind_partial(None, None).arguments
        defaults = {
            name: parameter.default
            for name, parameter in model_signature.parameters.items()
            if name not in inputs
            and parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
            and parameter.default is not Parameter.empty
        }
        settings = {**defaults, **(settings or {})}
        for name, value in dict(function=function, settings=_freeze(settings),
                                reconstruct_fn=reconstruct_fn, observation=observation,
                                specmodel=specmodel,
                                site_annotations=_freeze(site_annotations or {})).items():
            object.__setattr__(self, name, value)

    def _check_specmodel(self, specmodel):
        if self.specmodel is not None and specmodel is not self.specmodel:
            raise ValueError("configured model belongs to a different SpecModel; rebuild it for this fit")

    def __call__(self, observation, specmodel):
        self._check_specmodel(specmodel)
        return self.function(observation, specmodel, **self.settings)

    def prior_trace(self, *, rng_key=None, condition=None):
        """Evaluate parameter sites only; conditional values use the given seed.

        Condition values substitute sample sites in physical units. They must
        lie in the site's support; fixed/derived sites cannot be overridden.
        """
        if self.observation is None or self.specmodel is None:
            raise ValueError("prior inspection requires an observation and specmodel")
        condition = dict(condition or {})

        def substitute(site):
            if site["type"] == "sample" and not site["is_observed"]:
                return condition.get(site["name"])
            return None

        options = {**self.settings, "prior_only": True}
        seeded = handlers.seed(self.function, jax.random.PRNGKey(0) if rng_key is None else rng_key)
        trace = handlers.trace(handlers.substitute(seeded, substitute_fn=substitute)).get_trace(
            self.observation, self.specmodel, **options)
        for name, value in condition.items():
            site = trace.get(name)
            if site is None or site["type"] != "sample" or site["is_observed"]:
                raise ValueError(f"condition must name a sampled parameter, got {name!r}")
            if np.shape(value) != site["fn"].batch_shape + site["fn"].event_shape:
                raise ValueError(f"condition for {name!r} has the wrong shape")
            if not np.all(site["fn"].support.check(value)):
                raise ValueError(f"condition for {name!r} is outside its prior support")
        return trace

    def check_prior(self, *, rng_key=None, condition=None):
        """Return a compact parameter/prior table, without evaluating likelihood.

        Conditional parameters are labeled ``name | dependencies`` and show
        their expression. A conditional column is added only when dependencies
        are supplied or fixed, showing the condition and numerical distribution
        or derived value. Use prior_trace() for the underlying site objects.
        """
        import pandas as pd

        trace = self.prior_trace(rng_key=rng_key, condition=condition)
        known = dict(condition or {})
        rows = []
        for name, site in trace.items():
            if site["type"] not in ("sample", "deterministic") or site.get("is_observed", False):
                continue
            annotation = dict(self.site_annotations.get(name, {}))
            annotation.update(site.get("infer", {}).get("prior", {}))
            dependencies = tuple(annotation.get("depends_on", ()))
            sampled = site["type"] == "sample"
            concrete = all(parent in known for parent in dependencies)
            numeric = (_format_prior(site["fn"]) if sampled else _format_value(site["value"])) if concrete else None
            label = f"{name} | {', '.join(dependencies)}" if dependencies else name
            if dependencies:
                description = annotation.get("expression", "conditional prior" if sampled else "derived")
            else:
                description = numeric if sampled else f"{numeric} (fixed)"
            if sampled and np.shape(site["value"]):
                description += f" [shape={np.shape(site['value'])}]"
            row = dict(parameter=label, prior=description)
            if dependencies and concrete:
                at = ", ".join(f"{parent}={_format_value(known[parent])}" for parent in dependencies)
                row["conditional"] = f"{at}: {numeric}"
            if not sampled and concrete:
                known[name] = site["value"]
            rows.append(row)
        return pd.DataFrame(rows, columns=["parameter", "prior"] + (
            ["conditional"] if any("conditional" in row for row in rows) else []
        )).fillna("").set_index("parameter")

    def predict(self, point, observation, specmodel):
        """Replay every latent from point and call this model's reconstruction."""
        self._check_specmodel(specmodel)
        if self.reconstruct_fn is None:
            raise ValueError("this model has no reconstruct_fn; provide its reconstruction callback")
        if not isinstance(point, Mapping):
            raise TypeError("point must be a dict of native sample-site values")

        def substitute(site):
            if site["type"] == "sample" and not site["is_observed"]:
                if site["name"] not in point:
                    raise ValueError(f"point must supply latent site {site['name']!r}")
                return point[site["name"]]
            return None

        with handlers.trace() as trace, handlers.substitute(substitute_fn=substitute):
            returned = self(observation, specmodel)
        return self.reconstruct_fn(observation, specmodel, returned, trace, self.settings)
