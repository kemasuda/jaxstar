"""Teff-only PCHIP accuracy, shape, AD, factories and inference contracts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from jaxstar.grid import Field, RectilinearGrid
from jaxstar.specfit import Observation, SpecFit, SpecModel, save_spectral_grid
from jaxstar.specfit._data import _SpectralLibrary
from jaxstar.specfit._interpolation import interpolate_flux
from jaxstar.specfit.results import _grid_record


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def make_grid(dtype=np.float64, schema="coelho", count=6, permuted=False):
    t = np.array([4000., 4200., 4550., 5000., 5700., 6500.])[:count]
    metal = {"coelho": "feh", "bosz": "mh", "tlusty": "logZ"}[schema]
    axes = dict(teff=t, logg=[3., 4.5], **{metal: [-1., 0.3]})
    if schema != "tlusty":
        axes["alpha"] = [0.]
    if schema == "bosz":
        axes.update(carbon=[0.], vmic=[1., 2.])
    nodes = np.meshgrid(*axes.values(), indexing="ij")
    x = (nodes[0] - 4000) / 2500
    amplitude = .1 + .03 * nodes[1] + .02 * nodes[2]
    # Curved, flat, extremal and nearly flat pixels, in two payload regions.
    flux = np.stack([1 - amplitude * np.exp(x), np.ones_like(x),
                     .8 + .05 * np.sin(8*x + nodes[1]),
                     1 + 1.e-7 * x**2, .9 - .04 * x**3], axis=-1)
    flux = np.stack([flux, .95 * flux], axis=-2).astype(dtype)
    dims = tuple(axes)
    if permuted:
        order = tuple(reversed(range(len(axes))))
        flux = flux.transpose(order + (len(axes), len(axes) + 1))
        dims = tuple(reversed(dims))
    return RectilinearGrid(axes=axes, fields={
        "flux": Field(flux, dims, payload_dims=("region", "pixel"))})


def point(grid, teff=4800.):
    return {n: teff if n == "teff" else jnp.mean(grid.axis(n)) for n in grid.axis_names}


@pytest.mark.parametrize("schema", ["coelho", "bosz", "tlusty"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_nodes_linear_parity_and_scipy_reference(schema, dtype):
    grid = make_grid(dtype, schema, permuted=True)
    native = np.asarray(grid.axis("teff"))
    locations = point(grid, native)
    expected = grid.interpolate(locations)["flux"]
    np.testing.assert_array_equal(interpolate_flux(grid, locations, teff_interp="cubic"), expected)
    query = np.linspace(native[0], native[-1], 81)
    locations = point(grid, query)
    np.testing.assert_array_equal(interpolate_flux(grid, locations), grid.interpolate(locations)["flux"])
    fn = jax.jit(lambda g, p: interpolate_flux(g, p, teff_interp="cubic"))
    actual = fn(grid, locations)
    assert actual.shape == (81, 2, 5) and actual.dtype == dtype
    tol = 4e-7 if dtype == np.float32 else 2e-14
    np.testing.assert_allclose(actual, PchipInterpolator(native, expected)(query), atol=tol, rtol=tol)
    # The limiter must be applied AFTER bilinear interpolation of other axes.
    # The independent SciPy reference above uses precisely those native slices.


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_continuity_and_finite_gradients(dtype, mode):
    grid = make_grid(dtype)
    fn = lambda p: interpolate_flux(grid, p, teff_interp=mode)
    projection = lambda p: jnp.sum(fn(p) * jnp.arange(1., 6.))
    gradient = jax.jit(jax.grad(projection))
    native = np.asarray(grid.axis("teff"))
    reference = PchipInterpolator(native, np.asarray(grid.interpolate(point(grid, native))["flux"]))
    for t in np.linspace(4000., 6500., 31):
        assert all(np.all(np.isfinite(g)) for g in gradient(point(grid, t)).values())
    eps = .01 if dtype == np.float32 else 1.e-4
    for node in np.asarray(grid.axis("teff"))[1:-1]:
        below, at, above = (point(grid, node + dt) for dt in (-eps, 0, eps))
        np.testing.assert_allclose(fn(below), fn(at), atol=2e-6 if dtype == np.float32 else 2e-8)
        np.testing.assert_allclose(fn(above), fn(at), atol=2e-6 if dtype == np.float32 else 2e-8)
        if mode == "cubic":
            # Finite offsets differ by O(eps * curvature), even for a C1
            # interpolant. Bound that term with the independent reference.
            curvature = max(np.abs(reference(t, nu=2) * np.arange(1., 6.)).sum()
                            for t in (node-eps, node, node+eps))
            roundoff = 2e-9 if dtype == np.float32 else 1e-13
            np.testing.assert_allclose(gradient(below)["teff"], gradient(above)["teff"],
                                       rtol=0, atol=2.01*eps*curvature + roundoff)
            expected_slope = (reference(node, nu=1) * np.arange(1., 6.)).sum()
            np.testing.assert_allclose(gradient(at)["teff"], expected_slope, rtol=0, atol=roundoff)
    if mode == "linear":
        assert abs(float(gradient(point(grid, 5000-eps))["teff"]
                         - gradient(point(grid, 5000+eps))["teff"])) > 1.e-5


def test_broadcast_vmap_and_no_overshoot():
    grid = make_grid()
    fn = jax.jit(lambda p: interpolate_flux(grid, p, teff_interp="cubic"))
    t = np.asarray(grid.axis("teff"))
    for lo, hi in zip(t[:-1], t[1:]):
        p = point(grid, np.linspace(lo, hi, 13)[:, None])
        p["logg"] = jnp.array([3., 3.7, 4.5])[None, :]
        values = fn(p)
        assert values.shape == (13, 3, 2, 5)
        lower, upper = np.minimum(values[0], values[-1]), np.maximum(values[0], values[-1])
        assert np.all(values >= lower - 1e-14) and np.all(values <= upper + 1e-14)
    single = lambda t: fn(point(grid, t))
    np.testing.assert_allclose(jax.jit(jax.vmap(single))(t), fn(point(grid, t)))


def test_regular_grid_with_x64_disabled(x64_context):
    with x64_context(False):
        original = make_grid(np.float32)
        axes = {n: np.asarray(original.axis(n)) for n in original.axis_names}
        axes["teff"] = np.arange(4000., 5500., 250., dtype=np.float32)
        grid = RectilinearGrid(axes=axes, fields={"flux": original.field("flux")})
        fn = jax.jit(lambda g, p: interpolate_flux(g, p, teff_interp="cubic"))
        nodes = grid.axis("teff")
        expected = grid.interpolate(point(grid, nodes))["flux"]
        # Compiled float32 multilinear accumulation can differ from eager
        # evaluation by a few ulps even at native Teff coordinates.
        np.testing.assert_allclose(fn(grid, point(grid, nodes)), expected, rtol=3e-7, atol=1e-7)
        query = jnp.linspace(nodes[0], nodes[-1], 31)
        np.testing.assert_allclose(fn(grid, point(grid, query)),
                                   PchipInterpolator(nodes, expected)(query), atol=3e-7)
        grad = jax.jit(jax.vmap(jax.grad(lambda t: fn(grid, point(grid, t)).sum())))(query)
        assert np.all(np.isfinite(grad)) and grad.dtype == jnp.float32


@pytest.mark.parametrize("count", [1, 2, 3])
def test_small_grids_and_invalid_coordinates(count):
    grid = make_grid(count=count)
    ts = np.linspace(float(grid.axis("teff")[0]), float(grid.axis("teff")[-1]), 9)
    actual = interpolate_flux(grid, point(grid, ts), teff_interp="cubic")
    if count < 3:
        np.testing.assert_array_equal(actual, grid.interpolate(point(grid, ts))["flux"])
    else:
        native = grid.interpolate(point(grid, grid.axis("teff")))["flux"]
        np.testing.assert_allclose(actual, PchipInterpolator(grid.axis("teff"), native)(ts), atol=1e-14)
    for name in grid.axis_names:
        for invalid in (np.nan, np.inf, float(grid.axis(name)[-1]) + 1):
            p = {**point(grid, ts[0]), name: invalid}
            out = interpolate_flux(grid, p, teff_interp="cubic")
            assert np.all(np.isneginf(out))
            grad = jax.grad(lambda p: interpolate_flux(grid, p, teff_interp="cubic").sum())(p)
            assert all(np.all(np.isfinite(g)) for g in grad.values())


def model_library():
    velocity = np.linspace(-250., 250., 301)
    wave = 5000. * np.exp(velocity[None, :] / 299792.458)
    t = np.array([4500., 4800., 5300., 6000., 6500.])
    depth = .2 * np.exp((t-4500) / 2500)
    flux = 1 - depth[:, None, None] * np.exp(-.5*(velocity[None, None, :] / 5.)**2)
    grid = RectilinearGrid(axes={"teff": t}, fields={
        "flux": Field(flux, ("teff",), payload_dims=("region", "pixel"))})
    return _SpectralLibrary(grid, wave, (0,), "test", "vacuum", "normalized")


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_model_factories_metadata_and_full_gradient(tmp_path, mode):
    lib = model_library()
    model = SpecModel(lib, vmax=30., teff_interp=mode)
    wave = lib.wavelength[:, 100:201:5]
    obs = Observation(wave, jnp.ones_like(wave), jnp.full_like(wave, .01))
    path = save_spectral_grid(tmp_path / "grid.npz", lib)
    fit = SpecFit.from_grid(path, obs, vmax=30., wav_margin=1., teff_interp=mode)
    assert fit.specmodel.teff_interp == mode
    assert _grid_record(model)["teff_interp"] == mode
    leaves, tree = jax.tree_util.tree_flatten(model)
    assert jax.tree_util.tree_unflatten(tree, leaves).teff_interp == mode

    def evaluate(m, t):
        p = m.make_params(dict(teff=t, vsini=6., vmacro=3., u1=.4, u2=.2,
                               rv=.4, resolving_power=70000.))
        return m(p, wave)

    np.testing.assert_allclose(jax.jit(evaluate)(fit.specmodel, 5200.), evaluate(model, 5200.), atol=1e-13)
    value, grad = jax.jit(jax.value_and_grad(lambda t, m: evaluate(m, t).sum()))(5200., model)
    assert np.isfinite(value) and np.isfinite(grad)
    if mode == "linear":
        np.testing.assert_array_equal(evaluate(model, 5200.), evaluate(SpecModel(lib, vmax=30.), 5200.))


@pytest.mark.parametrize("invalid", [None, "pchip", 1])
def test_invalid_mode(invalid):
    with pytest.raises(ValueError, match="teff_interp"):
        SpecModel(model_library(), teff_interp=invalid)


def test_cubic_requires_named_teff():
    lib = model_library()
    grid = RectilinearGrid(axes={"custom_temperature": lib.grid.axis("teff")}, fields={
        "flux": Field(lib.grid.field("flux").values, ("custom_temperature",),
                      payload_dims=("region", "pixel"))})
    with pytest.raises(ValueError, match="'teff' grid axis"):
        SpecModel(replace(lib, grid=grid), teff_interp="cubic")


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_numpyro_svi_nuts(mode):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.infer.util import log_density
    from jaxstar.specfit import model_single

    model = SpecModel(model_library(), vmax=30., teff_interp=mode)
    wave = model.spectra.wavelength[:, 100:201:5]
    broadening = dict(vsini=6., vmacro=3., q1=.36, q2=.3)
    params = model.make_params(dict(teff=5200., **broadening, rv=.4, resolving_power=70000.))
    obs = Observation(wave, model(params, wave), jnp.full_like(wave, .01))
    kwargs = dict(atmosphere={"teff": dist.Uniform(4500., 6500.)}, broadening=broadening,
                  rv=.4, resolving_power=70000., sigma_constant=.1, sigma_continuum=.03, jitter=.004)
    density = lambda t: log_density(model_single, (obs, model), kwargs, {"teff": t})[0]
    value, grad = jax.jit(jax.value_and_grad(density))(5300.)
    assert np.isfinite(value) and np.isfinite(grad)
    initial = init_to_value(values={"teff": 5200.})
    guide = AutoNormal(model_single, init_loc_fn=initial)
    svi = SVI(model_single, guide, numpyro.optim.Adam(.02), Trace_ELBO(), **kwargs)
    result = svi.run(jax.random.PRNGKey(42), 15, obs, model, progress_bar=False)
    assert np.all(np.isfinite(result.losses))
    sampler = MCMC(NUTS(model_single, init_strategy=initial),
                   num_warmup=10, num_samples=10, progress_bar=False)
    sampler.run(jax.random.PRNGKey(43), obs, model, **kwargs)
    assert np.all(np.isfinite(sampler.get_samples()["teff"]))
