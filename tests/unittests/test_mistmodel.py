"""Offline numerical/API regression for the isochrone model/fit separation."""

from dataclasses import FrozenInstanceError
from itertools import product
import os

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import initialize_model, log_density
from numpyro.infer import init_to_value
import pytest
from scipy.interpolate import RegularGridInterpolator

from jaxstar.grid import Field, GridResult, RectilinearGrid
from jaxstar.mistfit import MistFit, MistModel, MistObservations
from _mistgrid_reference import NamedMistGridIso
from jaxstar.mistfit.mistgrid.paths import MISTGRID_ENV_VAR
from _mistfit_reference import ReferenceMistFit


BANDS = ("jmag", "hmag", "kmag", "gmag2", "bpmag2", "rpmag2",
         "gmag3", "bpmag3", "rpmag3", "umag", "bmag", "vmag", "rmag", "imag")
PARAMS = {"logage": 9.23, "feh_init": -.13, "eep": 378.2, "distance": .12}
OPTIONS = dict(logamin=8.1, logamax=9.9, fmin=-.45, fmax=.45,
               eepmin=310., eepmax=490., massmin=.88, massmax=1.45,
               dist_scale=1.7)


@pytest.fixture
def grid_path(tmp_path):
    """Nonlinear mass nodes and independent centered derivatives, all 14 bands."""
    ages, fehs, eeps = np.array([8., 9., 10.]), np.array([-.5, 0., .5]), np.arange(300., 501., 50.)
    a, f, e = np.meshgrid(ages, fehs, eeps, indexing="ij")
    x = (e - 300.) / 200.
    mass = .65 + .12 * (a - 8) + .05 * f + .4 * x + .3 * x**2 + .1 * x**3
    centered = (mass[..., 2:] - mass[..., :-2]) / 100.
    dmdeep = np.concatenate((centered[..., :1], centered, centered[..., -1:]), axis=-1)
    fields = dict(
        mass=mass, star_mass=.92 * mass, dmdeep=dmdeep,
        teff=5100. + 200. * mass + 30. * (a - 9) - 50. * f,
        logt=np.log10(5100. + 200. * mass), logl=.2 * mass,
        logg=4.6 - .5 * x + .1 * f, radius=.8 + .5 * x + .02 * (a - 9),
        feh_photosphere=f - .07 * x, logage=a,
        mmin=np.broadcast_to(mass[..., :1], mass.shape),
        mmax=np.broadcast_to(mass[..., -1:], mass.shape),
        eepmin=np.full_like(mass, 300.), eepmax=np.full_like(mass, 500.),
    )
    fields.update({name: 4. + i * .07 - .8 * x + .1 * f + .03 * (a - 9)
                   for i, name in enumerate(BANDS)})
    # A useful positive DR2 color for the gyro option, with nonzero gradients.
    fields["bpmag2"] = fields["rpmag2"] + .7 + .1 * x + .03 * f
    path = tmp_path / "isochrone.npz"
    np.savez(path, logagrid=ages, fgrid=fehs, eepgrid=eeps, **fields)
    return path


def make_model(path, backend="named"):
    return MistModel.from_isochrones(path, legacy_interpolation=(backend == "legacy"))


def reference_fields(path, params=PARAMS):
    with np.load(path) as data:
        axes = tuple(data[name] for name in ("logagrid", "fgrid", "eepgrid"))
        point = np.array([params[name] for name in ("logage", "feh_init", "eep")])
        return {name: RegularGridInterpolator(axes, data[name])(point)[0]
                for name in data.files if name not in ("logagrid", "fgrid", "eepgrid")}


@pytest.mark.parametrize("legacy", [False, True])
def test_factory_custom_grid_and_pytree_do_not_reload(grid_path, monkeypatch, legacy):
    model = MistModel.from_isochrones(grid_path, legacy_interpolation=legacy)
    assert model.kind == "isochrone"
    assert model.data.sources == (str(grid_path.resolve()),)
    assert set(model.grid.field_names) == set(reference_fields(grid_path))
    custom = MistModel(model.grid, legacy_interpolation=legacy)
    assert custom.grid is model.grid
    assert custom.data.sources == ()
    with pytest.raises(FrozenInstanceError):
        model.data.kind = "track"
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("evaluation must not load data"))
    leaves, treedef = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.data.sources == model.data.sources
    evaluate = jax.jit(lambda m, p: m(p, keys=("mass", "kmag", "parallax")))
    expected = evaluate(model, PARAMS)
    for candidate in (restored, custom):
        result = evaluate(candidate, PARAMS)
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])
    # A local inverse also uses only the in-memory mass field.
    np.testing.assert_allclose(restored.eep_given_mass(
        PARAMS["logage"], PARAMS["feh_init"], expected["mass"]), PARAMS["eep"], rtol=1e-6)


@pytest.mark.parametrize("legacy", [False, True])
def test_fits_share_model_but_keep_observations_separate(grid_path, monkeypatch, legacy):
    model = MistModel.from_isochrones(grid_path, legacy_interpolation=legacy)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("supplied model must be reused"))
    first = MistFit(model=model, observations={"teff": (5400., 70.)})
    second = MistFit(model=model, observations={"parallax": (8.4, .1)})
    assert first.physical_model is second.physical_model is model
    assert first.physical_model.legacy_interpolation is legacy
    assert not hasattr(first, "mg")
    assert second.observations.keys == ("parallax",)
    assert not hasattr(first, "outkeys") and not hasattr(first, "add_keys")
    options = dict(OPTIONS, linear_age=False)
    _, first_trace = log_density(first.model, (), dict(options, output_keys=("umag",)), PARAMS)
    _, second_trace = log_density(second.model, (), options, PARAMS)
    assert "umag" in first_trace and "umag" not in second_trace
    np.testing.assert_allclose(first_trace["obs"]["fn"].loc, [model(PARAMS, keys="teff")["teff"]])
    np.testing.assert_allclose(second_trace["obs"]["fn"].loc, [1 / PARAMS["distance"]])


def test_supplied_model_rejects_conflicting_construction_options(grid_path):
    model = MistModel.from_isochrones(grid_path)
    for extra in ({"path": grid_path}, {"legacy_interpolation": False}, {"legacy_interpolation": True}):
        with pytest.raises(ValueError, match="model cannot be combined"):
            MistFit(model=model, observations={}, **extra)
    with pytest.raises(TypeError, match="model must be a MistModel"):
        MistFit(model=model.grid, observations={})


def test_removed_grid_wrappers_are_not_production_exports():
    import jaxstar.mistfit as package
    from jaxstar.mistfit import mistfit as fitting_module
    for namespace in (package, fitting_module):
        assert not hasattr(namespace, "MistGridIso")
        assert not hasattr(namespace, "NamedMistGridIso")


@pytest.mark.parametrize("backend", ["legacy", "named"])
@pytest.mark.parametrize("x64", [False, True])
def test_all_outputs_match_independent_interpolation_and_units(grid_path, x64_context, x64, backend):
    with x64_context(x64):
        model = make_model(grid_path, backend)
        actual = jax.jit(lambda model, p: model(p))(model, PARAMS)
        raw = reference_fields(grid_path)
        expected = {**raw, "mass_init": raw["mass"], "mass_current": raw["star_mass"],
                    "feh": raw["feh_photosphere"], "feh_surface": raw["feh_photosphere"],
                    "rho": raw["star_mass"] / raw["radius"]**3,
                    "rho_initial": raw["mass"] / raw["radius"]**3,
                    **PARAMS, "age": 10**PARAMS["logage"] / 1e9, "parallax": 1 / PARAMS["distance"]}
        for band in BANDS:
            expected[band + "_abs"] = raw[band]
            expected[band] = raw[band] - 5 * np.log10(1 / PARAMS["distance"]) + 10
        assert isinstance(actual, GridResult)
        assert tuple(actual) == model.output_keys
        assert set(actual) == set(expected)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6 if not x64 else 2e-13)
            assert actual[key].dtype == (jnp.float64 if x64 else jnp.float32)
        # FeH input is initial, observed FeH is surface; mass names are distinct.
        assert actual["feh"] != actual["feh_init"]
        assert actual["mass"] != actual["star_mass"]


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_per_call_keys_jit_pytree_and_no_model_mutation(grid_path, backend):
    model = make_model(grid_path, backend)
    all_fields = model.grid.field_names
    compiled = jax.jit(lambda model, p, keys: model(p, keys=keys), static_argnames="keys")
    mass = compiled(model, PARAMS, "mass")
    selected = compiled(model, PARAMS, ("teff", "kmag_abs", "parallax", "mass_current"))
    assert tuple(mass) == ("mass",)
    assert tuple(selected) == ("teff", "kmag_abs", "parallax", "mass_current")
    np.testing.assert_allclose(selected["teff"], model(PARAMS, keys="teff")["teff"])
    np.testing.assert_allclose(jax.jit(model)(PARAMS)["mass"], mass["mass"])
    assert tuple(model(PARAMS, keys=["radius", "mass"])) == ("radius", "mass")
    assert model.grid.field_names == all_fields
    assert not hasattr(model, "set_keys")
    with pytest.raises(FrozenInstanceError):
        model.legacy_interpolation = True

    # Same schema, different numerical leaves: no grid values captured as
    # static self or reconstructed/reloaded during PyTree unflattening.
    changed = RectilinearGrid(
        axes={k: model.grid.axis(k) for k in model.grid.axis_names},
        fields={k: Field(model.grid.field(k).values + (1 if k == "mass" else 0),
                         model.grid.field(k).dims) for k in all_fields},
    )
    replacement = MistModel(changed, legacy_interpolation=(backend == "legacy"))
    np.testing.assert_allclose(compiled(replacement, PARAMS, "mass")["mass"], mass["mass"] + 1)


def test_distance_only_required_for_selected_distance_outputs(grid_path):
    model = make_model(grid_path)
    intrinsic = {k: v for k, v in PARAMS.items() if k != "distance"}
    selected = model(intrinsic, keys=("teff", "kmag_abs", "age", "feh_init"))
    np.testing.assert_allclose(selected["kmag_abs"], reference_fields(grid_path)["kmag"], rtol=1e-6)
    for key in ("parallax", "distance", "kmag"):
        with pytest.raises(ValueError, match="distance"):
            model(intrinsic, keys=key)
    # Coordinate-only selections need no interpolated field.
    np.testing.assert_allclose(model(intrinsic, keys="logage")["logage"], PARAMS["logage"])
    with pytest.raises(KeyError, match="unknown MistModel output"):
        model(PARAMS, keys="no_such_output")
    for keys in ((), ("mass", "mass"), (1,)):
        with pytest.raises(ValueError):
            model(PARAMS, keys=keys)
    with pytest.raises(ValueError, match="feh_init"):
        model({"logage": 9., "feh": 0., "eep": 350.}, keys="mass")


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_broadcast_vmap_and_prediction_gradients(grid_path, x64_context, backend):
    with x64_context():
        model = make_model(grid_path, backend)
        keys = ("teff", "mass_init", "dmdeep", "kmag", "kmag_abs", "age", "parallax", "rho")
        point = jnp.array(list(PARAMS.values()))

        def evaluate(model, point):
            result = model(dict(zip(PARAMS, point)), keys=keys)
            return jnp.stack([result[k] for k in keys], axis=-1)

        analytic = jax.jit(jax.jacrev(evaluate, argnums=1))(model, point)
        eps = np.array([1e-5, 1e-5, 1e-3, 1e-6])
        numerical = np.column_stack([(evaluate(model, point + step) - evaluate(model, point - step)) / (2 * size)
                                     for step, size in zip(np.diag(eps), eps)])
        np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-7)
        # Stored centered dm/dEEP intentionally differs from the segment slope.
        stored = model(PARAMS, keys="dmdeep")["dmdeep"]
        assert abs(stored - analytic[keys.index("mass_init"), 2]) > 1e-5
        batch = jnp.stack((point, point + jnp.array([.1, .1, 10., .01])))
        np.testing.assert_allclose(jax.jit(jax.vmap(evaluate, in_axes=(None, 0)))(model, batch),
                                   evaluate(model, batch.T), rtol=1e-12)
        result = model({**PARAMS, "feh_init": jnp.array([[-.2], [.1]]),
                        "distance": jnp.array([[.1, .2, .3]])}, keys=keys)
        assert all(value.shape == (2, 3) for value in result.values())


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_both_modes_keep_named_boundaries_and_nonfinite_field_behavior(grid_path, tmp_path, backend):
    with np.load(grid_path) as data:
        fields = {key: data[key] for key in data.files}
    fields["teff"][0, 0, 0] = np.nan
    fields["mass"][0, 0, 0] = -np.inf
    path = tmp_path / "nonfinite.npz"
    np.savez(path, **fields)
    model = make_model(path, backend)
    adapter = NamedMistGridIso(path)
    keys = ("teff", "mass", "kmag")
    for age, feh, eep in ((8., -.5, 300.), (8.2, -.3, 320.), (10., .5, 500.),
                          (7.9, .1, 370.), (9.2, .6, 370.), (9.2, .1, 510.),
                          (np.nan, .1, 370.)):
        expected = dict(adapter.values(age, feh, eep, keys=keys))
        expected["kmag"] = expected["kmag"] - 5 * jnp.log10(1 / PARAMS["distance"]) + 10
        actual = model({**PARAMS, "logage": age, "feh_init": feh, "eep": eep}, keys=keys)
        for key in keys:
            np.testing.assert_allclose(actual[key], jnp.where(jnp.isnan(expected[key]), -jnp.inf, expected[key]), rtol=1e-6)


def make_fit(cls, grid_path, backend):
    observations = MistObservations(
        ["teff", "logg", "feh", "gmag3", "jmag", "parallax"],
        [5400., 4.4, -.15, 10., 9.5, 8.4], [70., .1, .06, .04, .03, .1])
    if cls is ReferenceMistFit:
        fit = cls(grid_path, grid_backend=backend)
        fit.set_data(observations.keys, observations.values, observations.errors)
        return fit
    return cls(grid_path, legacy_interpolation=(backend == "legacy"), observations=observations)


@pytest.mark.parametrize("backend", ["legacy", "named"])
@pytest.mark.parametrize("linear_age,nodata,flat", list(product((False, True), repeat=3)))
def test_frozen_fit_sites_log_densities_and_gradients(grid_path, x64_context, backend, linear_age, nodata, flat):
    with x64_context():
        options = dict(OPTIONS, linear_age=linear_age, nodata=nodata, flat_age_marginal=flat,
                       prot=18., prot_err=.07, rho_obs=1.1, rho_err=.2, age_obs=2., age_err=.3)
        parameters = dict(PARAMS)
        if linear_age:
            parameters = {"age": 10**parameters.pop("logage") / 1e9, **parameters}
        point = jnp.array(list(parameters.values()))
        results, traces = [], []
        for cls in (ReferenceMistFit, MistFit):
            fit = make_fit(cls, grid_path, backend)

            def density(point):
                return log_density(fit.model, (), options, dict(zip(parameters, point)))[0]

            results.append(jax.jit(jax.value_and_grad(density))(point))
            traces.append(log_density(fit.model, (), options, parameters)[1])
        assert set(traces[0]) == set(traces[1])
        for name in traces[0]:
            old, new = traces[0][name], traces[1][name]
            assert old["type"] == new["type"]
            np.testing.assert_allclose(old["value"], new["value"], rtol=2e-13, atol=2e-13)
            if old["type"] == "sample":
                # Check each prior/likelihood contribution, not only their sum.
                np.testing.assert_allclose(old["fn"].log_prob(old["value"]),
                                           new["fn"].log_prob(new["value"]), rtol=2e-12, atol=2e-12)
        for old, new in zip(jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(results[1])):
            assert np.all(np.isfinite(new))
            np.testing.assert_allclose(old, new, rtol=3e-11, atol=3e-10)
        if nodata:
            assert "obs" not in traces[1]
        else:
            assert "obs" in traces[1]
        # Centered derivative is the prior input, independent of AD of mass.
        raw = reference_fields(grid_path)
        penalty = -30 * (1 / (np.exp(20 * (raw["mass"] - OPTIONS["massmin"])) + 1)
                         + 1 / (np.exp(-20 * (raw["mass"] - OPTIONS["massmax"])) + 1))
        expected = np.log(raw["dmdeep"]) + penalty
        if flat:
            expected -= np.log(min(raw["mmax"], OPTIONS["massmax"]) - raw["mmin"])
        np.testing.assert_allclose(traces[1]["logjac"]["fn"].log_factor, expected, rtol=1e-12)


@pytest.mark.parametrize("extra", [{}, {"prot": 18.}, {"rho_obs": 1.}, {"rho_err": .2},
                                   {"age_obs": 2.}, {"age_err": .3}])
def test_disabled_and_partial_optional_likelihoods(grid_path, extra):
    params = {"age": 10**PARAMS["logage"] / 1e9, **{k: v for k, v in PARAMS.items() if k != "logage"}}
    old = make_fit(ReferenceMistFit, grid_path, "legacy")
    new = make_fit(MistFit, grid_path, "legacy")
    old_value, old_trace = log_density(old.model, (), dict(OPTIONS, **extra), params)
    new_value, new_trace = log_density(new.model, (), dict(OPTIONS, **extra), params)
    assert set(old_trace) == set(new_trace)
    np.testing.assert_allclose(old_value, new_value, rtol=1e-6)


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_fit_key_changes_do_not_mutate_or_reuse_stale_physical_selection(grid_path, backend, monkeypatch, x64_context):
    # Key selection should not depend on float32 age-to-logage roundoff.
    with x64_context():
        fit = make_fit(MistFit, grid_path, backend)
        model = fit.physical_model
        monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("changing output keys must not reload data"))
        params = {"age": 10**PARAMS["logage"] / 1e9, **{k: v for k, v in PARAMS.items() if k != "logage"}}
        log_density(fit.model, (), OPTIONS, params)
        _, trace = log_density(fit.model, (), dict(OPTIONS, output_keys=(
            "umag", "kmag_abs", "mass_current", "age", "parallax", "feh")), params)
        assert "kmag_abs" in trace and "mass_current" in trace
        fit = MistFit(model=model, observations=MistObservations(
            ["kmag_abs", "mass_current", "feh_init", "age", "parallax"],
            [4., 1., -.1, 2., 8.4], [.1, .1, .1, .2, .1]))
        _, trace = log_density(fit.model, (), OPTIONS, params)
        assert fit.physical_model is model
        assert "umag" not in trace  # Additional outputs belonged only to the preceding call.
        expected = model(PARAMS, keys=fit.observations.keys)
        np.testing.assert_allclose(trace["obs"]["fn"].loc, [expected[k] for k in fit.observations.keys], rtol=1e-6)


@pytest.mark.parametrize("backend", ["legacy", "named"])
@pytest.mark.parametrize("linear_age", [False, True])
def test_additional_outputs_preserve_sites_density_and_nuts_gradient(grid_path, x64_context, backend, linear_age):
    with x64_context():
        fit = make_fit(MistFit, grid_path, backend)
        old = make_fit(ReferenceMistFit, grid_path, backend)
        # The frozen fitter can save raw fields using its historical setter.
        old.add_keys(["umag", "logl"])
        additional = ("umag", "logl", "rho", "mass_init", "mass_current", "kmag_abs",
                      "age", "logage", "distance", "parallax", "feh", "mass", "rho")
        options = dict(OPTIONS, linear_age=linear_age, flat_age_marginal=True,
                       prot=18., rho_obs=1.1, rho_err=.2, age_obs=2., age_err=.3)
        start = dict(PARAMS)
        if linear_age:
            start = {"age": 10**start.pop("logage") / 1e9, **start}
        default_value, default = log_density(fit.model, (), options, start)
        old_value, old_trace = log_density(old.model, (), options, start)
        results = []
        for selection in (None, (), additional):
            kwargs = dict(options, output_keys=selection)
            value, trace = log_density(fit.model, (), kwargs, start)
            np.testing.assert_allclose(value, default_value, rtol=1e-13, atol=1e-12)
            for name, site in default.items():
                np.testing.assert_array_equal(trace[name]["value"], site["value"])
                if site["type"] == "sample":
                    np.testing.assert_array_equal(trace[name]["fn"].log_prob(site["value"]),
                                                  site["fn"].log_prob(site["value"]))
            assert set(trace) == set(default) | set(selection or ())
            if selection:
                for key in ("umag", "logl"):
                    np.testing.assert_allclose(trace[key]["value"], old_trace[key]["value"], rtol=2e-12)
                np.testing.assert_allclose(value, old_value, rtol=2e-12)
                expected = fit.physical_model(PARAMS, keys=tuple(dict.fromkeys(additional)))
                for key in expected:
                    np.testing.assert_allclose(trace[key]["value"], expected[key], rtol=2e-12)
            info = initialize_model(jax.random.PRNGKey(13), fit.model,
                                    init_strategy=init_to_value(values=start), model_kwargs=kwargs)
            results.append(jax.jit(jax.value_and_grad(info.potential_fn))(info.param_info.z))
        for result in results[1:]:
            for base, value in zip(jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(result)):
                assert np.all(np.isfinite(value))
                np.testing.assert_allclose(value, base, rtol=1e-12, atol=1e-10)
        # Calling with extra keys did not change subsequent default traces.
        assert set(log_density(fit.model, (), options, start)[1]) == set(default)
        assert not hasattr(fit, "outkeys") and not hasattr(fit, "add_keys")


@pytest.mark.parametrize("selection", [None, (), [], "rho", ["rho", "mass", "rho"]])
def test_output_key_normalization_and_static_jit(grid_path, selection):
    fit = make_fit(MistFit, grid_path, "named")
    options = dict(OPTIONS, linear_age=False)
    default = log_density(fit.model, (), options, PARAMS)[1]
    value, trace = log_density(fit.model, (), dict(options, output_keys=selection), PARAMS)
    expected_keys = {selection} if isinstance(selection, str) else set(selection or ())
    assert set(trace) == set(default) | expected_keys
    # A key sequence passed through an outer JIT must be a static tuple.
    evaluate = jax.jit(lambda p, keys: log_density(
        fit.model, (), dict(options, output_keys=keys), p)[0], static_argnames="keys")
    keys = selection if selection is None or isinstance(selection, str) else tuple(selection)
    np.testing.assert_allclose(evaluate(PARAMS, keys), value, rtol=1e-6)


@pytest.mark.parametrize("selection,exception", [("", ValueError), ([1], ValueError),
                                                  ([[]], ValueError), ("not_an_output", KeyError)])
def test_invalid_additional_outputs_fail_without_changing_the_fit(grid_path, selection, exception):
    fit = make_fit(MistFit, grid_path, "named")
    options = dict(OPTIONS, linear_age=False)
    with pytest.raises(exception):
        log_density(fit.model, (), dict(options, output_keys=selection), PARAMS)
    assert np.isfinite(log_density(fit.model, (), options, PARAMS)[0])
    assert not hasattr(fit, "outkeys")


@pytest.mark.parametrize("linear_age", [False, True])
def test_actual_nuts_potential_matches_reference_and_finite_differences(grid_path, x64_context, linear_age):
    with x64_context():
        options = dict(OPTIONS, linear_age=linear_age, flat_age_marginal=True,
                       prot=18., rho_obs=1.1, rho_err=.2, age_obs=2., age_err=.3)
        start = dict(PARAMS)
        if linear_age:
            start["age"] = 10**start.pop("logage") / 1e9
        results = []
        for cls in (ReferenceMistFit, MistFit):
            fit = make_fit(cls, grid_path, "legacy")
            info = initialize_model(jax.random.PRNGKey(13), fit.model,
                                    init_strategy=init_to_value(values=start), model_kwargs=options)
            names = tuple(info.param_info.z)
            point = jnp.stack([info.param_info.z[name] for name in names])
            potential = lambda p: info.potential_fn(dict(zip(names, p)))
            value, grad = jax.jit(jax.value_and_grad(potential))(point)
            results.append((value, grad))
            eps = 1e-5
            fd = jnp.array([(potential(point + delta) - potential(point - delta)) / (2 * eps)
                            for delta in eps * np.eye(len(point))])
            np.testing.assert_allclose(grad, fd, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-12)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("backend", ["legacy", "named"])
def test_independent_four_chain_nuts_runs_on_tiny_grid(grid_path, backend):
    fit = make_fit(MistFit, grid_path, backend)
    options = dict(OPTIONS, nodata=True)
    sampler_options = dict(num_warmup=8, num_samples=8, max_tree_depth=3, progress_bar=False)
    # Exercise the actual four-chain default, not a mocked sampler.
    first = fit.run_nuts(jax.random.PRNGKey(42), model_kwargs=options, **sampler_options)
    expected_method = "parallel" if jax.local_device_count() >= 4 else "sequential"
    assert first.chain_method == expected_method
    saved = {key: np.array(value) for key, value in first.get_samples(group_by_chain=True).items()}
    assert all(value.shape == (4, 8) and np.all(np.isfinite(value)) for value in saved.values())
    defaults = set(saved)
    assert {"age", "logage", "feh_init", "eep", "distance", "mass", "star_mass", "kmag"} <= defaults
    selections = (["rho", "mass_init", "age", "parallax", "mass", "rho"], "kmag_abs", None)
    for i, selection in enumerate(selections):
        result = fit.run_hmc(jax.random.PRNGKey(43 + i), model_kwargs=options,
                             num_chains=1, output_keys=selection, **sampler_options)
        samples = result.get_samples()
        extras = {selection} if isinstance(selection, str) else set(selection or ())
        assert set(samples) == defaults | extras
        assert result is not first
        assert all(np.all(np.isfinite(value)) for value in samples.values())
        if "rho" in extras:
            np.testing.assert_allclose(samples["rho"], samples["star_mass"] / samples["radius"]**3, rtol=1e-6)
        if "kmag_abs" in extras:
            np.testing.assert_allclose(samples["kmag_abs"],
                                       samples["kmag"] + 5 * np.log10(samples["parallax"]) - 10, rtol=2e-6)
    for key, value in first.get_samples(group_by_chain=True).items():
        np.testing.assert_array_equal(value, saved[key])
    assert options == dict(OPTIONS, nodata=True)
    for removed in ("set_data", "setup_hmc", "mcmc", "samples", "outkeys", "add_keys"):
        assert not hasattr(fit, removed)


def test_legacy_float32_preserves_steps_computed_in_npz_precision(grid_path, tmp_path, x64_context):
    with np.load(grid_path) as data:
        fields = {key: data[key] for key in data.files}
    fields["logagrid"] = np.array([8., 8.05, 8.1], dtype=np.float64)
    fields["fgrid"] = np.array([.2, .3, .4], dtype=np.float64)
    path = tmp_path / "fractional_steps.npz"
    np.savez(path, **fields)
    params = {"age": 10**8.073 / 1e9, "feh_init": .337, "eep": 378.2, "distance": .12}
    with x64_context(False):
        old = make_fit(ReferenceMistFit, path, "legacy")
        new = make_fit(MistFit, path, "legacy")
        # This input distinguishes the NPZ step from subtracting cast nodes.
        assert new.physical_model.grid._axis_first_steps[0] != jnp.diff(new.physical_model.grid.axis("age"))[0]
        point = jnp.array(list(params.values()))
        results = [jax.jit(jax.value_and_grad(
            lambda p: log_density(fit.model, (), {}, dict(zip(params, p)))[0]))(point)
            for fit in (old, new)]
        for a, b in zip(jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(results[1])):
            np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
        old_trace = log_density(old.model, (), {}, params)[1]
        new_trace = log_density(new.model, (), {}, params)[1]
        for key in old.outkeys:
            np.testing.assert_allclose(old_trace[key]["value"], new_trace[key]["value"], rtol=2e-7)


@pytest.mark.parametrize("massmax", [1.45, 2.5])
def test_flat_age_marginal_preserves_mass_limits(grid_path, massmax):
    params = {"age": 10**PARAMS["logage"] / 1e9, **{k: v for k, v in PARAMS.items() if k != "logage"}}
    options = dict(OPTIONS, nodata=True, flat_age_marginal=True, massmax=massmax)
    traces = [log_density(make_fit(cls, grid_path, "legacy").model, (), options, params)[1]
              for cls in (ReferenceMistFit, MistFit)]
    np.testing.assert_allclose(traces[0]["logjac"]["fn"].log_factor,
                               traces[1]["logjac"]["fn"].log_factor, rtol=1e-6)
    # The saved deterministic mmax is the raw grid output, not the prior's cap.
    np.testing.assert_allclose(traces[1]["mmax"]["value"], reference_fields(grid_path)["mmax"], rtol=1e-6)


@pytest.mark.parametrize("dmdeep", [0., -.1, np.nan])
def test_invalid_prior_derivative_retains_negative_infinity(grid_path, tmp_path, dmdeep):
    with np.load(grid_path) as data:
        fields = {key: data[key] for key in data.files}
    fields["dmdeep"] = np.full_like(fields["dmdeep"], dmdeep)
    path = tmp_path / "invalid_derivative.npz"
    np.savez(path, **fields)
    params = {"age": 10**PARAMS["logage"] / 1e9, **{k: v for k, v in PARAMS.items() if k != "logage"}}
    for cls in (ReferenceMistFit, MistFit):
        fit = make_fit(cls, path, "legacy")
        _, trace = log_density(fit.model, (), dict(OPTIONS, nodata=True), params)
        assert np.isneginf(trace["logjac"]["fn"].log_factor)


@pytest.mark.full_grid
@pytest.mark.skipif(not os.environ.get(MISTGRID_ENV_VAR),
                    reason=f"set {MISTGRID_ENV_VAR} for the optional cached-grid check")
@pytest.mark.parametrize("x64", [False, True])
def test_full_grid_refactor_matches_frozen_fit(x64_context, x64):
    """Read only an explicitly supplied grid; never generate or download one."""
    with x64_context(x64):
        path = os.environ[MISTGRID_ENV_VAR]
        old = make_fit(ReferenceMistFit, path, "legacy")
        new = make_fit(MistFit, path, "legacy")
        options = dict(OPTIONS, linear_age=False, flat_age_marginal=True,
                       prot=18., rho_obs=1., rho_err=.2, age_obs=5., age_err=.3)
        # Interior MS, turnoff and giant coordinates in the packaged isochrone.
        for eep in (330.4, 435.5, 500.4):
            params = dict(logage=9.7, feh_init=.03, eep=eep, distance=.12)
            point = jnp.array(list(params.values()))
            results = [jax.jit(jax.value_and_grad(
                lambda p: log_density(fit.model, (), options, dict(zip(params, p)))[0]))(point)
                for fit in (old, new)]
            for a, b in zip(jax.tree_util.tree_leaves(results[0]), jax.tree_util.tree_leaves(results[1])):
                assert np.all(np.isfinite(b))
                np.testing.assert_allclose(a, b, rtol=1e-5 if not x64 else 1e-10, atol=1e-5 if not x64 else 1e-9)
            old_trace = log_density(old.model, (), options, params)[1]
            new_trace = log_density(new.model, (), options, params)[1]
            for key in old.outkeys:
                np.testing.assert_allclose(old_trace[key]["value"], new_trace[key]["value"],
                                           rtol=2e-6 if not x64 else 2e-12, atol=2e-6 if not x64 else 2e-12)
    jax.clear_caches()
