"""Two-solver Matérn-3/2 comparison for 10 independent spectral orders.

GPU: JAX_PLATFORMS=cuda python benchmarks/benchmark_tinygp_minimal.py 2048
Optional first argument is pixels per order. Requires parallel tinygp.
Uses a Python loop inside JIT, matching jaxstar's per-region GP evaluation.
Times the summed likelihood and its shared-hyperparameter gradient.
"""

import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import tinygp

jax.config.update("jax_enable_x64", True)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
n_orders = 10
order = jnp.arange(n_orders)[:, None]
x = jnp.linspace(0., 20., n)[None, :] * (1 + .01 * order)
y = .02 * jnp.sin(3 * x + .2 * order) + .01 * jnp.cos(7 * x)
diag = jnp.broadcast_to((.02 * (1 + .02 * order))**2, x.shape)
theta = jnp.log(jnp.array([.025, .55]))  # amplitude, correlation length
args = (theta, x, y, diag)
jax.block_until_ready(args)
print(f"JAX {jax.__version__}, tinygp {tinygp.__version__}, {jax.devices()}, "
      f"orders={n_orders}, pixels/order={n}", flush=True)


def objective(parallel):
    def logp(theta, x, y, diag):
        amplitude, scale = jnp.exp(theta)
        if parallel:
            kernel = tinygp.kernels.quasisep.Matern32(sigma=amplitude, scale=scale)
            options = dict(solver=tinygp.solvers.QuasisepSolver,
                           assume_sorted=True, parallel=True)
        else:
            kernel = amplitude**2 * tinygp.kernels.Matern32(scale=scale)
            options = dict(solver=tinygp.solvers.DirectSolver)
        total = jnp.zeros((), dtype=x.dtype)
        for i in range(x.shape[0]):
            gp = tinygp.GaussianProcess(kernel, x[i], diag=diag[i], mean=0., **options)
            total += gp.log_probability(y[i])
        return total
    return jax.jit(jax.value_and_grad(logp))


results, times = [], []
for name, parallel in (("direct", False), ("quasisep parallel", True)):
    print(f"{name}: compiling/warming up", flush=True)
    fn = objective(parallel)
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    elapsed = []
    for _ in range(20):
        start = perf_counter()
        result = jax.block_until_ready(fn(*args))
        elapsed.append(perf_counter() - start)
    value, gradient = map(np.asarray, result)
    assert np.isfinite(value) and np.all(np.isfinite(gradient)), name
    results.append(np.r_[value, gradient])
    times.append(float(np.median(elapsed)))
    print(f"{name}: {times[-1] * 1000:.3f} ms, logp={value:.10f}, grad={gradient}", flush=True)

np.testing.assert_allclose(results[1], results[0], rtol=1e-8, atol=1e-8)
print(f"Values/gradients agree; direct / parallel time = {times[0] / times[1]:.2f}x")
