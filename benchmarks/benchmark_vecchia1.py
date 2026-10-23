"""GPU/CPU timing and approximation accuracy for a masked M32 spectral problem.

JAX_PLATFORMS=cuda PYTHONPATH=src python benchmarks/benchmark_vecchia1.py --fit
Requires tinygp with QuasisepSolver(parallel=True). No installed code is edited:
a scoped benchmark-only constructor adapter enables parallel execution in the
unchanged production continuum-marginalization path. Timed inputs are dynamic,
device-resident float64 arrays. Compilation, simulation and fitting are separate.
"""

import argparse
from contextlib import nullcontext
from functools import partial
import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import tinygp
from unittest.mock import patch

from _benchmark_utils import emit_result, environment, measure
from jaxstar.specfit import (
    Observation, chebyshev_basis, gp_marginalized_continuum_log_likelihood,
    marginalized_continuum_log_likelihood, prepare_gp_observation,
)


PARAMETERS = ["depth", "rv_km_s", "log_width", "log_gp_amplitude", "log_gp_scale",
              "log_jitter", "log_sigma_constant", "log_sigma_continuum"]


def line_model(theta, wave, centers):
    depth, rv, log_width = theta[:3]
    rest = wave / (1 + rv / 299792.458)
    profile = jnp.exp(-.5 * ((rest[..., None]-centers[:, None, :]) / jnp.exp(log_width))**2)
    strengths = jnp.linspace(.4, 1., centers.shape[-1])
    return 1 - depth * jnp.sum(profile * strengths, axis=-1)


def problem(pixels, regions, degree, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(.015, .025, size=(regions, pixels)), axis=1)
    wave = 5000. + 1000 * np.arange(regions)[:, None] + x
    centers = wave[:, :1] + (wave[:, -1:] - wave[:, :1]) * np.linspace(.1, .9, 12)
    theta = jnp.array([.35, 2., *np.log([.12, .025, .55, .009, .03, .01])])
    basis = chebyshev_basis(wave, degree)
    prior_scale = np.r_[.03, np.full(degree, .01)]
    coefficients = rng.normal(size=(regions, degree+1)) * prior_scale
    coefficients[:, 0] += 1
    error = rng.uniform(.012, .03, size=wave.shape)
    flux = np.asarray(line_model(theta, jnp.asarray(wave), jnp.asarray(centers)))
    y = flux * np.sum(np.asarray(basis) * coefficients[:, None, :], axis=-1)
    # Exact M32 + independent observational/jitter noise, not a Vecchia draw.
    for i in range(regions):
        gp = tinygp.GaussianProcess(
            tinygp.kernels.quasisep.Matern32(sigma=.025, scale=.55),
            jnp.asarray(wave[i]-wave[i, 0]), diag=jnp.asarray(error[i]**2+.009**2),
            solver=tinygp.solvers.QuasisepSolver, assume_sorted=True, mean=0.)
        y[i] += np.asarray(gp.sample(jax.random.PRNGKey(seed+i)))
    mask = rng.random(wave.shape) < .03
    mask[:, pixels//3:pixels//3+pixels//40] = True
    mask[:, 0] = mask[:, -1] = False
    y[mask], error[mask] = np.nan, np.inf
    observation = Observation(wave, y, error, mask)
    return theta, prepare_gp_observation(observation), jnp.asarray(centers), basis, coefficients


def objective(method, degree):
    def logp(theta, data, centers, basis):
        obs = data.observation
        flux = line_model(theta, obs.wavelength, centers)
        a, length, jitter, constant, continuum = jnp.exp(theta[3:])
        options = dict(basis=basis, degree=degree, jitter=jitter,
                       sigma_constant=constant, sigma_continuum=continuum)
        if method == "white":
            return marginalized_continuum_log_likelihood(obs, flux, **options)
        return gp_marginalized_continuum_log_likelihood(
            data, flux, gp_amplitude=a, gp_scale=length,
            gp_solver="vecchia1" if method == "vecchia1" else "quasisep", **options)
    return logp


def parallel_context(method):
    if method != "tinygp_parallel":
        return nullcontext()
    return patch.object(tinygp.solvers, "QuasisepSolver",
                        partial(tinygp.solvers.QuasisepSolver, parallel=True))


def decoded(theta):
    result = dict(zip(PARAMETERS[:2], map(float, theta[:2])))
    result.update({name.removeprefix("log_"): float(np.exp(value))
                   for name, value in zip(PARAMETERS[2:], theta[2:])})
    return result


def fit_model(fn, inputs, maxiter):
    compiled = jax.jit(jax.value_and_grad(fn)).lower(*inputs).compile()
    initial, *data = inputs
    count = sum(map(len, inputs[1].indices))
    def loss(theta):
        value, gradient = jax.block_until_ready(compiled(jnp.asarray(theta), *data))
        return -float(value)/count, -np.asarray(gradient)/count
    bounds = [(.05, .8), (-10., 10.)] + [tuple(np.log(pair)) for pair in
              ((.04, .4), (.001, .2), (.02, 10.), (.0001, .1), (.001, .2), (.001, .1))]
    result = minimize(loss, np.asarray(initial), jac=True, bounds=bounds, method="L-BFGS-B",
                      options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-7, maxls=40))
    theta = np.asarray(result.x)
    distance = [min(v-lo, hi-v) for v, (lo, hi) in zip(theta, bounds)]
    return dict(success=bool(result.success), message=str(result.message),
                iterations=int(result.nit), evaluations=int(result.nfev),
                log_likelihood=float(-result.fun*count), parameters=decoded(theta),
                theta=theta.tolist(), bounds=[list(pair) for pair in bounds],
                near_bound=[name for name, gap in zip(PARAMETERS, distance) if gap < 1e-4],
                gradient_per_pixel=np.asarray(result.jac).tolist())


def compare_fits(report, inputs):
    """Score all fitted points with each density, separating shifts from offsets."""
    points = {name: result["fit"]["theta"] for name, result in report["results"].items()
              if "fit" in result}
    comparison = {}
    for method in points:
        with parallel_context(method):
            compiled = jax.jit(objective(method, report["degree"])).lower(*inputs).compile()
        values = {name: float(compiled(jnp.asarray(theta), *inputs[1:]))
                  for name, theta in points.items()}
        comparison[method] = dict(log_likelihood_at_fit=values,
                                 difference_from_own_fit={name: value-values[method]
                                                         for name, value in values.items()})
    report["fit_comparison"] = comparison


def run(args):
    jax.config.update("jax_enable_x64", True)
    if "parallel" not in inspect.signature(tinygp.solvers.QuasisepSolver).parameters:
        raise RuntimeError("This comparison requires tinygp with QuasisepSolver(parallel=True); "
                           "install a supporting version in a separate environment.")
    theta, data, centers, basis, coefficients = problem(
        args.pixels, args.regions, args.degree, args.seed)
    inputs = jax.device_put((theta, data, centers, basis))
    jax.block_until_ready(inputs)
    methods = ["tinygp_parallel", "vecchia1", "white"]
    report = dict(benchmark="vecchia1-m32-continuum", environment=environment(jax.devices()[0]),
                  xla_flags=os.environ.get("XLA_FLAGS", ""),
                  tinygp_version=tinygp.__version__, tinygp_revision=args.tinygp_revision,
                  pixels=args.pixels, regions=args.regions, degree=args.degree, seed=args.seed,
                  usable_pixels=[len(i) for i in data.indices], rounds=args.rounds,
                  gradient_parameters=PARAMETERS, truth=decoded(theta),
                  continuum_draws=coefficients.tolist(), methods=methods, results={},
                  notes=["Synthetic 12-line spectrum/order, not an atmosphere-grid recovery test.",
                         "Data drawn from exact M32 plus heteroscedastic errors and jitter.",
                         "All methods marginalize the same Gaussian Chebyshev continuum.",
                         "White noise drops the GP; its amplitude/scale derivatives are zero.",
                         "Independent order loop matches production and supports unequal mask sizes."])
    for method in methods:
        print(f"{method}: compiling and timing", flush=True)
        fn = objective(method, args.degree)
        result = {}
        with parallel_context(method):
            for label, operation in (("likelihood", fn), ("value_and_grad", jax.value_and_grad(fn))):
                timing, memory, value = measure(operation, inputs, rounds=args.rounds, warmup=5)
                result[label] = dict(timing=timing, compiler_memory=memory)
                if label == "value_and_grad":
                    lp, grad = value
                    if not np.isfinite(lp) or not np.isfinite(grad).all():
                        raise ValueError(f"{method}: nonfinite density/gradient")
                    result.update(log_likelihood=float(lp), gradient=np.asarray(grad).tolist())
            if args.fit and method != "white":
                print(f"{method}: fitting physical and nuisance parameters", flush=True)
                result["fit"] = fit_model(fn, inputs, args.maxiter)
        report["results"][method] = result

    # Verify that the benchmark adapter changes only the exact solver algorithm.
    sequential = jax.jit(jax.value_and_grad(objective("sequential", args.degree)))(*inputs)
    baseline = report["results"]["tinygp_parallel"]
    np.testing.assert_allclose(sequential[0], baseline["log_likelihood"], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(sequential[1], baseline["gradient"], rtol=1e-8, atol=1e-7)
    report["parallel_vs_sequential"] = dict(
        log_likelihood_abs=float(abs(sequential[0]-baseline["log_likelihood"])),
        gradient_max_abs=float(np.max(np.abs(np.asarray(sequential[1])-baseline["gradient"]))))
    reference_gradient = np.asarray(baseline["gradient"])
    for result in report["results"].values():
        for label in ("likelihood", "value_and_grad"):
            result[label]["speedup_vs_tinygp_parallel"] = (
                baseline[label]["timing"]["warm_median_seconds"] /
                result[label]["timing"]["warm_median_seconds"])
        difference = np.asarray(result["gradient"]) - reference_gradient
        result["difference_vs_tinygp_parallel"] = dict(
            log_likelihood=result["log_likelihood"]-baseline["log_likelihood"],
            gradient=difference.tolist(), gradient_max_abs=float(np.max(np.abs(difference))),
            gradient_l2_relative=float(np.linalg.norm(difference)/np.linalg.norm(reference_gradient)))
    if args.fit:
        compare_fits(report, inputs)
    emit_result(report, args.json_out)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=2048)
    parser.add_argument("--regions", type=int, default=8)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--tinygp-revision")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.pixels < 32 or args.regions < 1 or args.degree < 0 or args.rounds < 1 or args.maxiter < 1:
        parser.error("pixels >= 32, regions/rounds/maxiter >= 1 and degree >= 0 are required")
    run(args)
