"""Known-truth spectral recovery using ONE NumPyro model for SVI and NUTS.

Run from the checkout with PYTHONPATH=src and the spectral-inference extra.
CPU/GPU selection uses JAX_PLATFORMS before Python starts, not a library toggle.
This longer validation is intentionally outside CI; use --help for run options.
"""

import argparse
from functools import partial
import importlib.metadata
from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.util import log_density

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev_notebooks/specfit"))
from _spectral_inference_data import synthetic_case, point_params
from _benchmark_utils import environment, measure, emit_result
from jaxstar.specfit import (model_single, continuum_posterior,
                            evaluate_continuum, apply_continuum)


def timed_svi(case, *, initial, steps, step_size, seed):
    """Instrument the same guide/Adam/ELBO used by find_map_svi; no new objective.

    Observation/library are dynamic arguments here. Record scan compilation
    separately from device execution, and preserve losses for convergence plots.
    The external helper is also actually run and compared below.
    """
    model = partial(model_single, **case.priors)
    guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=initial))
    svi = SVI(model, guide, numpyro.optim.Adam(step_size), Trace_ELBO())
    start = perf_counter()
    state = svi.init(jax.random.PRNGKey(seed), case.observation, case.specmodel)
    jax.block_until_ready(state)
    initialize_seconds = perf_counter() - start

    def run(state, obs, specmodel):
        def step(state, _):
            return svi.update(state, obs, specmodel)
        return jax.lax.scan(step, state, None, length=steps)

    start = perf_counter()
    compiled = jax.jit(run).lower(state, case.observation, case.specmodel).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    final, losses = jax.block_until_ready(compiled(state, case.observation, case.specmodel))
    optimize_seconds = perf_counter() - start
    estimate = {name: value for name, value in guide.median(svi.get_params(final)).items()
                if name in case.latent_names}
    losses = np.asarray(losses)
    assert np.all(np.isfinite(losses)), "nonfinite SVI objective"
    return estimate, losses, {"initialize_seconds": initialize_seconds,
        "compile_seconds": compile_seconds, "optimization_seconds": optimize_seconds,
        "steps": steps, "initial_loss": float(losses[0]), "final_loss": float(losses[-1]),
        "last_100_loss_range": float(np.ptp(losses[-100:]))}


def recovery_table(case, estimate, samples):
    rows = []
    for name in estimate:
        truth = np.atleast_1d(case.truth.get(name, case.reference_sigma_continuum))
        values = np.asarray(samples[name]).reshape(len(samples[name]), -1)
        point = np.atleast_1d(estimate[name])
        for index in range(values.shape[1]):
            draws = values[:, index]
            lower, median, upper = np.quantile(draws, [.05, .5, .95])
            interval95 = np.quantile(draws, [.025, .975]).tolist()
            std = float(draws.std(ddof=1))
            rows.append({"parameter": name if len(point) == 1 else f"{name}[{index}]",
                "truth": None if name == "sigma_continuum" else float(truth[index]),
                "reference": float(truth[index]) if name == "sigma_continuum" else None,
                "map_svi": float(point[index]),
                "posterior_mean": float(draws.mean()), "posterior_median": float(median),
                "posterior_std": std, "interval_90": [float(lower), float(upper)],
                "interval_95": interval95,
                "truth_offset_sigma": None if name == "sigma_continuum" else float((draws.mean() - truth[index]) / std)})
    return rows


def diagnostics(case, estimate, samples, losses, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    point = dict(case.truth, **{name: np.mean(value, axis=0) for name, value in samples.items()
                               if name in estimate})
    physical = np.asarray(case.specmodel(point_params(point, case.specmodel.spectra.grid.axis_names),
                                        case.observation.wavelength))
    posterior = continuum_posterior(case.observation, physical, basis=case.basis,
        sigma_constant=point["sigma_constant"], sigma_continuum=point["sigma_continuum"],
        jitter=point["jitter"])
    continuum = np.asarray(evaluate_continuum(case.basis, posterior.mean))
    fitted = np.asarray(apply_continuum(physical, case.basis, posterior.mean))
    obs = case.observation
    fig, axes = plt.subplots(obs.n_regions, 2, figsize=(13, 3.1 * obs.n_regions), squeeze=False)
    for r, (left, right) in enumerate(axes):
        good = ~np.asarray(obs.mask[r])
        x = np.asarray(obs.wavelength[r])
        left.plot(x[good], np.asarray(obs.flux[r])[good], ".", color=".65", ms=2, label="Synthetic data")
        left.plot(x, case.physical[r] * case.continuum[r], "--", color="black", lw=1, label="Truth + continuum")
        left.plot(x, fitted[r], color="tab:blue", lw=1, label="Recovered + conditional continuum")
        left.set(title=f"Region {r}", ylabel="Flux", xlabel="Wavelength [Å]")
        left.legend(fontsize=7)
        residual = np.asarray(obs.flux[r]) - fitted[r]
        right.plot(x[good], residual[good], ".", ms=2, color="tab:blue")
        right.axhline(0, color="black", lw=.6)
        right.set(title="Usable-pixel residual", ylabel="Flux residual", xlabel="Wavelength [Å]")
        for ax in (left, right):
            ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    fig.tight_layout()
    fig.savefig(output / "spectra-residuals.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in range(obs.n_regions):
        axes[0].plot(np.linspace(-1, 1, obs.n_pixels), case.continuum[r], "--", color=f"C{r}")
        axes[0].plot(np.linspace(-1, 1, obs.n_pixels), continuum[r], color=f"C{r}", label=f"Region {r}")
    axes[0].set(title="Continuum: truth dashed, recovered solid", xlabel="Region coordinate", ylabel="Continuum")
    axes[0].legend()
    for index, loss in enumerate(losses):
        axes[1].plot(loss - loss.min(), label=f"Initialization {index+1}")
    axes[1].set(yscale="symlog", title="SVI convergence", xlabel="Adam step", ylabel="Loss minus run minimum")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "continuum-convergence.png", dpi=150)
    plt.close(fig)
    selected = ("teff", "logg", "mh", "alpha", "vsini", "vmacro", "rv", "jitter", "sigma_continuum")
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for name, ax in zip(selected, axes.flat):
        ax.hist(np.asarray(samples[name]), bins=25, alpha=.65, density=True)
        reference = case.truth.get(name, case.reference_sigma_continuum)
        ax.axvline(reference, color="black", linestyle="--",
                   label="Reference prior scale" if name == "sigma_continuum" else "Truth")
        ax.axvline(float(estimate[name]), color="tab:red", label="MAP/SVI")
        ax.set(xlabel=name, ylabel="Density")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "posteriors.png", dpi=150)
    plt.close(fig)
    return {"continuum_rmse": float(np.sqrt(np.mean((continuum-case.continuum)**2))),
        "physical_flux_rmse": float(np.sqrt(np.mean((physical-case.physical)**2))),
        "usable_residual_rms": float(np.sqrt(np.mean((np.asarray(obs.flux)[~obs.mask]-fitted[~obs.mask])**2))),
        "coefficient_truth": case.coefficients.tolist(),
        "coefficient_posterior_mean": np.asarray(posterior.mean).tolist(),
        "coefficient_posterior_std": np.sqrt(np.diagonal(np.asarray(posterior.covariance), axis1=-2, axis2=-1)).tolist()}


def run(args):
    from numpyro_inferutils import find_map_svi
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case = synthetic_case(regions=args.regions, pixels=args.pixels,
                          model_pixels=args.model_pixels, dtype=np.dtype(args.dtype))
    # Transfer once before synchronized timing. Library arrays remain dynamic.
    case.specmodel, case.observation = jax.device_put((case.specmodel, case.observation))
    model = partial(model_single, **case.priors)
    _, trace = log_density(model, (case.observation, case.specmodel), {}, case.initial)
    case.latent_names = tuple(name for name, site in trace.items()
                             if site["type"] == "sample" and not site["is_observed"])
    theta = {name: jnp.asarray(value) for name, value in case.initial.items()}
    density = lambda point, obs, sm: log_density(model, (obs, sm), {}, point)[0]
    density_timing, density_memory, _ = measure(density, (theta, case.observation, case.specmodel), rounds=args.rounds)
    gradient_timing, gradient_memory, (_, gradient) = measure(jax.value_and_grad(density),
        (theta, case.observation, case.specmodel), rounds=args.rounds)
    assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(gradient))
    truth_values = {name: case.truth.get(name, case.reference_sigma_continuum) for name in theta}
    assert density(truth_values, case.observation, case.specmodel) > density(theta, case.observation, case.specmodel)
    estimates, histories, timings = [], [], []
    for index in range(args.initializations):
        initial = dict(case.initial)
        if index:
            initial.update(teff=5950., logg=4.4, mh=-.25, alpha=.08, vsini=8.5, rv=13.1,
                           jitter=.003, sigma_continuum=.06)
        estimate, loss, timing = timed_svi(case, initial=initial, steps=args.steps,
                                         step_size=args.step_size, seed=index)
        estimates.append(estimate)
        histories.append(loss)
        timings.append(timing)
        print(f"SVI {index+1}: {timing}", flush=True)
    best = int(np.argmin([loss[-1] for loss in histories]))
    estimate = estimates[best]
    start = perf_counter()
    external = find_map_svi(model_single, args.step_size, args.steps,
        rng_key=jax.random.PRNGKey(0), p_initial=case.initial, progress_bar=False,
        observation=case.observation, specmodel=case.specmodel, **case.priors)
    jax.block_until_ready(external)
    helper_seconds = perf_counter() - start
    external = {name: external[name] for name in case.latent_names}
    helper_difference = {name: float(np.max(np.abs(np.asarray(external[name]) - np.asarray(estimates[0][name]))))
                         for name in external}
    # Compare relative to physically meaningful scales, not equality across XLA layouts.
    scales = {"teff": 1500., "logg": 1.2, "mh": 1., "alpha": .4, "vsini": 11.,
              "vmacro": 5.5, "rv": 9., "resolving_power": 30000., "jitter": .01, "sigma_continuum": .04}
    helper_scaled_difference = max(helper_difference[k]/scales[k] for k in external)
    assert helper_scaled_difference < 1e-4, "instrumented and inferutils SVI disagree"
    print(f"find_map_svi wall: {helper_seconds:.3f}s; max scaled difference: {helper_scaled_difference:.3g}", flush=True)
    mcmc = MCMC(NUTS(model, init_strategy=init_to_value(values=estimate), dense_mass=True,
                     target_accept_prob=.9, max_tree_depth=8),
                num_warmup=args.warmup, num_samples=args.samples, progress_bar=False)
    start = perf_counter()
    mcmc.warmup(jax.random.PRNGKey(10), case.observation, case.specmodel)
    jax.block_until_ready(mcmc.last_state)
    warmup_seconds = perf_counter() - start
    # First posterior run compiles the sampling scan. Repeat with adapted state
    # to time a post-compilation segment, not infer throughput from total warmup.
    extra = ("num_steps", "accept_prob", "diverging", "potential_energy")
    start = perf_counter()
    mcmc.run(jax.random.PRNGKey(11), case.observation, case.specmodel, extra_fields=extra)
    jax.block_until_ready(mcmc.get_samples())
    first_sample_seconds = perf_counter() - start
    mcmc.post_warmup_state = mcmc.last_state
    start = perf_counter()
    mcmc.run(jax.random.PRNGKey(12), case.observation, case.specmodel, extra_fields=extra)
    jax.block_until_ready(mcmc.get_samples())
    warm_sample_seconds = perf_counter() - start
    samples, fields = mcmc.get_samples(), mcmc.get_extra_fields()
    assert all(np.all(np.isfinite(value)) for value in samples.values())
    assert not {"norm", "slope", "coefficients"}.intersection(samples)
    table = recovery_table(case, estimate, samples)
    assert all(abs(row["truth_offset_sigma"]) < 4 for row in table
               if row["truth_offset_sigma"] is not None), "synthetic physical/noise recovery failed"
    reconstruction = diagnostics(case, estimate, samples, histories, output)
    np.savez(output / "posterior-samples.npz", **{k: np.asarray(v) for k,v in samples.items()},
             svi_loss=np.stack(histories))
    report = {"benchmark": "single-star-numpyro-synthetic", "environment": environment(jax.devices()[0]),
        "numpyro": numpyro.__version__, "numpyro_inferutils": importlib.metadata.version("numpyro-inferutils"),
        "problem": {"regions": args.regions, "observed_pixels": args.pixels,
                    "model_pixels": args.model_pixels, "dtype": args.dtype, "seed": case.seed,
                    "truth_continuum_degree": 2, "analysis_continuum_degree": 4,
                    "sigma_continuum_reference_not_injected_truth": case.reference_sigma_continuum,
                    "minimum_observed_model_grid_separation_angstrom": case.minimum_grid_separation,
                    "sampled_sites": list(estimate), "latent_dimensions": sum(np.size(x) for x in estimate.values())},
        "log_density": density_timing, "value_and_grad": gradient_timing,
        "memory": {"log_density": density_memory, "value_and_grad": gradient_memory},
        "svi": {"runs": timings, "estimates": [{k: np.asarray(v).tolist() for k,v in p.items()} for p in estimates],
                "best_run": best, "inferutils_total_seconds": helper_seconds,
                "inferutils_point_difference": helper_difference, "inferutils_max_scaled_difference": helper_scaled_difference},
        "nuts": {"warmup_steps": args.warmup, "samples": args.samples,
                 "compile_initialize_warmup_seconds": warmup_seconds,
                 "first_sampling_compile_and_run_seconds": first_sample_seconds,
                 "warm_sampling_seconds": warm_sample_seconds,
                 "warm_draws_per_second": args.samples/warm_sample_seconds,
                 "warm_leapfrog_steps_per_second": float(np.sum(fields["num_steps"]))/warm_sample_seconds,
                 "median_leapfrog_steps": float(np.median(fields["num_steps"])),
                 "mean_acceptance": float(np.mean(fields["accept_prob"])),
                 "divergences": int(np.sum(fields["diverging"])),
                 "finite_potential": bool(np.all(np.isfinite(fields["potential_energy"])))},
        "recovery": table, "reconstruction": reconstruction,
        "limitations": ["One short chain, not a convergence-certified analysis",
                        "Synthetic independent depth responses, not real stellar physics",
                        "Fixed degree-2 coefficients: sigma_continuum is an analysis hyperparameter, not an injected truth",
                        "q1/q2 fixed; calibrated resolving-power and weak vmacro priors specified explicitly"]}
    emit_result(report, output / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, default=3)
    parser.add_argument("--pixels", type=int, default=384)
    parser.add_argument("--model-pixels", type=int, default=1201)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--step-size", type=float, default=.01)
    parser.add_argument("--initializations", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output-dir", default="benchmark-results/spectral-inference")
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.samples, args.rounds) < 1:
        parser.error("steps/warmup/samples/rounds must be positive")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    backend = jax.default_backend()
    if (args.require_cpu and backend != "cpu") or (args.require_gpu and backend != "gpu"):
        parser.error(f"required backend unavailable: {backend}")
    run(args)


if __name__ == "__main__":
    main()
