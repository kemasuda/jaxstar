"""Reconstruction paired with the marginalized-continuum single-star model."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from .continuum import chebyshev_basis, continuum_posterior, evaluate_continuum
from .likelihood import gp_continuum_posterior, gp_conditional_mean


def _mean_prediction(predict, samples, *, max_draws):
    """Average curves and summarize continuum/GP means across draws on the host."""
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("samples must be a nonempty dict with a leading draw axis")
    samples = {name: np.asarray(value) for name, value in samples.items()}
    if any(value.ndim == 0 for value in samples.values()):
        raise ValueError("each samples value must have a leading draw axis")
    counts = {len(value) for value in samples.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("samples must have the same positive number of draws at every site")
    count = counts.pop()
    if max_draws is not None:
        if isinstance(max_draws, bool) or not isinstance(max_draws, Integral) or max_draws < 1:
            raise ValueError("max_draws must be a positive integer or None")
        count = min(count, max_draws)
    indices = np.linspace(0, len(next(iter(samples.values()))) - 1, count, dtype=int)
    curve_names = ("physical_flux", "continuum", "continuum_flux", "mean_flux", "gp_mean")
    means = {}
    continuum_draws = []
    gp_draws = []
    for index in indices:
        prediction = predict({name: value[index] for name, value in samples.items()})
        if not isinstance(prediction, Mapping) or "mean_flux" not in prediction:
            raise ValueError("model prediction must return a mapping containing mean_flux")
        curves = {name: np.asarray(prediction[name]) for name in curve_names if name in prediction}
        if not means:
            means = {name: np.zeros(value.shape, dtype=np.result_type(value.dtype, float))
                     for name, value in curves.items()}
        if curves.keys() != means.keys() or any(curves[name].shape != means[name].shape for name in means):
            raise ValueError("model prediction curve names and shapes must be the same across draws")
        for name, value in curves.items():
            means[name] += value / len(indices)
        if "continuum_flux" in curves:
            continuum_draws.append(curves["continuum_flux"].copy())
            # Match the plotted GP curve, including callbacks without gp_mean.
            gp_draws.append(curves["mean_flux"] - curves["continuum_flux"])
    result = {**means, "draw_indices": indices}
    if continuum_draws:
        result["continuum_flux_quantiles"] = np.quantile(continuum_draws, [.16, .84], axis=0)
        result["gp_mean_quantiles"] = np.quantile(gp_draws, [.16, .84], axis=0)
    return result


def reconstruct_single(obs, specmodel, params, trace, settings):
    """Reconstruct continuum/GP conditional means from a replayed model trace.

    This callback is specific to Gaussian, analytically marginalized continuum
    coefficients. A custom likelihood or sampled-continuum model supplies its
    own callback to ConfiguredModel. Settings must already include the model's
    resolved degree and basis, plus gp_solver when a GP is used.
    """
    names = ("sigma_constant", "sigma_continuum", "jitter")
    if not isinstance(params, Mapping) or "components" not in params or not all(name in trace for name in names):
        raise ValueError("reconstruction model must return SpecModel params and record "
                         "sigma_constant, sigma_continuum and jitter sites")
    physical = specmodel(params, obs.wavelength)
    degree = settings["degree"]
    basis = settings["basis"]
    if basis is None:
        basis = chebyshev_basis(obs.wavelength, degree=degree)
    noise = {name: trace[name]["value"] for name in names}
    gp_names = ("gp_amplitude", "gp_scale")
    gp = {name: trace[name]["value"] for name in gp_names if name in trace}
    if gp and len(gp) != 2:
        raise ValueError("reconstruction requires paired gp_amplitude/gp_scale sites")
    gp_options = dict(gp, gp_solver=settings["gp_solver"]) if gp else {}
    post = (gp_continuum_posterior(obs, physical, degree=degree, basis=basis, **noise, **gp_options)
            if gp else continuum_posterior(obs, physical, degree=degree, basis=basis, **noise))
    continuum = evaluate_continuum(basis, post.mean)
    baseline = physical * continuum
    result = {"params": params, "physical_flux": physical, "continuum": continuum,
              "continuum_posterior": post, "continuum_flux": baseline,
              "mean_flux": baseline, "jitter": noise["jitter"]}
    if gp:
        result["gp_mean"] = gp_conditional_mean(obs, baseline, jitter=noise["jitter"], **gp_options)
        result["mean_flux"] = baseline + result["gp_mean"]
    return result
