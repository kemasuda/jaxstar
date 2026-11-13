"""Trusted pixels stay narrow during inference and survive outlier padding."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm
from numpyro.infer.util import log_density

from jaxstar.specfit import (
    Observation, SpecFit, SpecModel, gaussian_mixture_log_likelihood, gaussian_mixture_responsibility,
    make_model_single, save_fit_result, load_fit_result,
)
from test_gaussian_mixture_spectral import reference
from test_specfit import case


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("fraction", [0., .15, 1.])
def test_guard_density_probability_and_jit_gradient(single, fraction, x64_context):
    with x64_context():
        wave = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        flux = np.array([[.8, np.nan, 1.02, .7], [.9, np.nan, 1.01, .6]])
        error = np.full((2, 4), .01)
        invalid = np.zeros((2, 4), bool)
        invalid[:, 1] = True
        guard = np.broadcast_to([True, True, False, False], (2, 4)).copy()
        if single:
            wave, flux, error, invalid, guard = [v[0] for v in (wave, flux, error, invalid, guard)]
        obs = Observation(wave, flux, error, invalid)
        mean = np.ones(obs.shape)
        options = dict(mismatch_fraction=fraction, mismatch_scale=.2, jitter=.004)
        with np.errstate(divide='ignore'):
            logp, q = reference(flux, mean, error, .004, fraction, .2)
        logp = np.where(guard, norm.logpdf(flux, mean, np.hypot(error, .004)), logp)
        q = np.where(invalid, np.nan, np.where(guard, 0., q))
        actual = gaussian_mixture_log_likelihood(obs, mean, **options, mask_guard=guard)
        np.testing.assert_allclose(actual, logp[~invalid].sum(), atol=1e-11)
        np.testing.assert_allclose(gaussian_mixture_responsibility(
            obs, mean, **options, mask_guard=guard), q, atol=1e-12, equal_nan=True)
        fn = lambda m: gaussian_mixture_log_likelihood(obs, m, **options, mask_guard=guard)
        value, gradient = jax.jit(jax.value_and_grad(fn))(jnp.asarray(mean))
        np.testing.assert_allclose(value, actual)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(gradient[guard & ~invalid],
                                   ((flux - mean) / (error**2 + .004**2))[guard & ~invalid])
        np.testing.assert_array_equal(gradient[invalid], 0.)
        np.testing.assert_allclose(gaussian_mixture_log_likelihood(
            obs, mean, **options, mask_guard=np.zeros(obs.shape, bool)),
            gaussian_mixture_log_likelihood(obs, mean, **options), rtol=0, atol=0)


def test_all_guard_has_no_mixture_parameter_gradient(x64_context):
    with x64_context():
        obs = Observation([1., 2.], [.8, 1.1], [.01, .02])
        def density(fraction, scale):
            return gaussian_mixture_log_likelihood(obs, jnp.ones(2),
                mismatch_fraction=fraction, mismatch_scale=scale, mask_guard=np.ones(2, bool))
        value, gradients = jax.jit(jax.value_and_grad(density, argnums=(0, 1)))(.15, .2)
        np.testing.assert_allclose(value, norm.logpdf(obs.flux, 1., obs.uncertainty).sum())
        np.testing.assert_array_equal(gradients, [0., 0.])


@pytest.mark.parametrize("guard", [True, [True], [1, 0], [0., 1.]])
def test_guard_requires_exact_shape_and_boolean_dtype(guard):
    obs = Observation([1., 2.], [1., 1.], [.01, .01])
    for function in (gaussian_mixture_log_likelihood, gaussian_mixture_responsibility):
        with pytest.raises(ValueError, match="mask_guard must be a boolean array"):
            function(obs, [1., 1.], mismatch_fraction=.1, mismatch_scale=.2, mask_guard=guard)


@pytest.mark.parametrize("single", [False, True])
def test_factory_prediction_padding_and_saved_guard(single, tmp_path, x64_context):
    with x64_context():
        fit, _, _ = case(single)
        flux = fit.observation.flux.copy()
        flux[..., [49, 50]] -= .4
        fit.observation = replace(fit.observation, flux=flux)
        guard = np.zeros(fit.observation.shape, bool)
        guard[..., [3, 50, 51]] = True  # Invalid pixel, outlier, neighbour.
        common = dict(resolving_power=70000., degree=1, use_gp=False,
            fixed=dict(rv=12., vsini=6., vmacro=3., q1=.36, q2=.3, jitter=.004))
        model = make_model_single(fit, **common, likelihood="gaussian_mixture",
            mismatch_fraction=.08, mismatch_scale=.1, mask_guard=guard)
        point = dict(teff=5800., continuum_coefficients=np.broadcast_to(
            [1.01, .01], fit.observation.shape[:-1] + (2,)).copy())
        prediction = fit.predict(point=point, model=model)
        q = prediction['mismatch_probability']
        assert np.all(q[..., 49] > .99)
        np.testing.assert_array_equal(q[..., [50, 51]], 0.)
        assert np.isnan(q[..., 3]).all()
        # A contaminated guarded pixel really contributes a narrow likelihood.
        _, trace = log_density(model, (fit.observation, fit.specmodel), {}, point)
        expected = gaussian_mixture_log_likelihood(fit.observation, prediction['model_flux'],
            mismatch_fraction=.08, mismatch_scale=.1, jitter=.004, mask_guard=guard)
        np.testing.assert_allclose(trace['spectrum']['fn'].log_factor, expected)
        flags = fit.mask_outliers(point=point, model=model, mask_velocity=10.)
        assert flags[..., 49].all()
        assert not flags[guard].any()
        assert fit.effective_mask[..., 3].all()
        unguarded = make_model_single(fit, **common, likelihood="gaussian_mixture",
                                      mismatch_fraction=.08, mismatch_scale=.1)
        flags_without = fit.mask_outliers(point=point, model=unguarded, mask_velocity=10.)
        assert flags_without[..., [50, 51]].all()
        fit.set_fit_mask(flags)
        save_fit_result(tmp_path, sf=fit, model=model, point=point)
        saved = load_fit_result(tmp_path)
        np.testing.assert_array_equal(saved.metadata['model']['settings']['mask_guard']['values'], guard)
        with pytest.raises(ValueError, match="mask_guard requires"):
            make_model_single(fit, **common, mask_guard=guard)


def mg_case(*, medium="air", single=False, **options):
    """Three sampled windows, with a bad pixel inside the first Mg guard."""
    from jaxstar.grid import Field, RectilinearGrid
    from jaxstar.specfit._data import _SpectralLibrary
    from jaxstar.specfit.libraries import _air_to_vacuum

    centers = np.array([5167.322, 5172.684, 5183.604])[:1 if single else 3]
    wave = centers[:, None] + np.linspace(-6, 6, 401)
    obs_wave = centers[:, None] + np.linspace(-1.8, 1.8, 101)
    beta = 12. / 299792.458
    if medium == "vacuum":
        wave, obs_wave = _air_to_vacuum(wave), _air_to_vacuum(obs_wave)
    wave = np.geomspace(wave[:, 0], wave[:, -1], 401).T
    obs_wave *= np.sqrt((1 + beta) / (1 - beta))
    regions = tuple(range(len(centers)))
    nodes = np.ones((2, len(centers), len(wave[0])))
    grid = RectilinearGrid(axes={"teff": np.array([5000., 6500.])}, fields={
        "flux": Field(nodes, ("teff",), payload_dims=("region", "pixel"))})
    model = SpecModel(_SpectralLibrary(grid, wave, regions, "fixture",
                                      wavelength_medium=medium), vmax=30.)
    if single:
        obs_wave = obs_wave[0]
    mask = np.zeros_like(obs_wave, bool)
    mask[..., 50] = True
    obs = Observation(obs_wave, np.ones_like(obs_wave), np.full_like(obs_wave, .01),
                      mask=mask, region=regions)
    return SpecFit(obs, model, **options)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_mg_matches_manual_windows_and_freezes_initial_rv(single, medium, x64_context):
    from jaxstar.specfit.libraries import _air_to_vacuum

    with x64_context():
        fit = mg_case(medium=medium, single=single, mg_guard=True)
        assert fit.guard_rv is None
        with pytest.raises(ValueError, match="check_ccf"):
            _ = fit.mask_guard
        fit.ccf = {"rv": 12.}
        beta = 12./299792.458
        rest = fit.observation.wavelength / np.sqrt((1+beta)/(1-beta))
        centers = np.array([5167.322, 5172.684, 5183.604])
        if medium == "air":
            manual = np.any(abs(rest[..., None]-centers) <= 1., axis=-1)
        else:
            manual = np.any((rest[..., None] >= _air_to_vacuum(centers-1.)) &
                            (rest[..., None] <= _air_to_vacuum(centers+1.)), axis=-1)
        manual &= ~fit.observation.mask
        np.testing.assert_array_equal(fit.mask_guard, manual)
        assert not fit.mask_guard.flags.writeable
        assert fit.guard_rv == 12.
        fit.ccf = {"rv": 100.}
        np.testing.assert_array_equal(fit.mask_guard, manual)
        fit.reset_fit_mask()
        np.testing.assert_array_equal(fit.mask_guard, manual)


def test_custom_union_copy_and_explicit_rv(x64_context):
    with x64_context():
        base = mg_case()
        custom = np.zeros(base.observation.shape, bool)
        custom[..., [0, 50]] = True  # Outside Mg and a fixed-invalid pixel.
        fit = SpecFit(base.observation, base.specmodel, mg_guard=True,
                      custom_guard=custom, guard_rv=12.)
        saved = custom.copy()
        custom[:] = False
        mg_only = SpecFit(base.observation, base.specmodel, mg_guard=True, guard_rv=12.)
        np.testing.assert_array_equal(fit.mask_guard,
            (saved | mg_only.mask_guard) & ~base.observation.mask)
        np.testing.assert_array_equal(fit.custom_guard, saved)
        assert not fit.custom_guard.flags.writeable
        assert fit.ccf is None
        assert fit.guard_rv == 12.
        no_guard = SpecFit(base.observation, base.specmodel)
        assert no_guard.mask_guard is None


def test_guard_configuration_validation(x64_context):
    with x64_context():
        base, _, _ = case()
        with pytest.raises(ValueError, match="wavelength_medium"):
            SpecFit(base.observation, base.specmodel, mg_guard=True)
        with pytest.raises(TypeError, match="mg_guard must be a bool"):
            SpecFit(base.observation, base.specmodel, mg_guard=1)
        with pytest.raises(ValueError, match="guard_rv requires"):
            SpecFit(base.observation, base.specmodel, guard_rv=0.)
        for invalid in [True, np.zeros(base.observation.shape), np.zeros(2, bool)]:
            with pytest.raises(ValueError, match="boolean array"):
                SpecFit(base.observation, base.specmodel, custom_guard=invalid)
        for invalid in [np.nan, np.inf, True, [0.], 299792.458]:
            with pytest.raises(ValueError, match="guard_rv must be"):
                mg_case(mg_guard=True, guard_rv=invalid)


@pytest.mark.parametrize("single", [False, True])
def test_specfit_guard_matches_legacy_density_probability_and_mask(single, tmp_path, x64_context):
    with x64_context():
        base, _, _ = case(single)
        flux = base.observation.flux.copy()
        flux[..., [49, 50]] -= .4
        obs = replace(base.observation, flux=flux)
        guard = np.zeros(obs.shape, bool)
        guard[..., [3, 50, 51]] = True
        fit = SpecFit(obs, base.specmodel, custom_guard=guard)
        legacy_fit = SpecFit(obs, base.specmodel)
        common = dict(resolving_power=70000., degree=1, use_gp=False,
            fixed=dict(rv=12., vsini=6., vmacro=3., q1=.36, q2=.3, jitter=.004))
        mixture = dict(likelihood="gaussian_mixture", mismatch_fraction=.08, mismatch_scale=.1)
        model = make_model_single(fit, **common, **mixture)
        legacy = make_model_single(legacy_fit, **common, **mixture, mask_guard=guard)
        point = dict(teff=5800., continuum_coefficients=np.broadcast_to(
            [1.01, .01], obs.shape[:-1] + (2,)).copy())
        actual, _ = log_density(model, (obs, base.specmodel), {}, point)
        expected, _ = log_density(legacy, (obs, base.specmodel), {}, point)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        np.testing.assert_allclose(
            fit.predict(model=model, point=point)['mismatch_probability'],
            legacy_fit.predict(model=legacy, point=point)['mismatch_probability'],
            rtol=0, atol=0, equal_nan=True)
        for velocity in [0., 10., "auto"]:
            actual = fit.mask_outliers(model=model, point=point, mask_velocity=velocity)
            expected = legacy_fit.mask_outliers(model=legacy, point=point, mask_velocity=velocity)
            np.testing.assert_array_equal(actual, expected)
            assert not actual[guard].any()
        # Final Gaussian models do not need a guard argument, yet saves retain it.
        final = make_model_single(fit, **common)
        assert final.settings['mask_guard'] is None
        save_fit_result(tmp_path, sf=fit, model=final, point={'teff': 5800.})
        saved = load_fit_result(tmp_path)
        np.testing.assert_array_equal(saved.mask_guard, fit.mask_guard)
        np.testing.assert_array_equal(saved.custom_guard, guard)
        restored = SpecFit(saved.observation, fit.specmodel, custom_guard=saved.mask_guard)
        restored.set_fit_mask(saved.fit_mask)
        restored_model = make_model_single(restored, **common, **mixture)
        np.testing.assert_allclose(
            restored.predict(model=restored_model, point=point)['mismatch_probability'],
            fit.predict(model=model, point=point)['mismatch_probability'],
            rtol=0, atol=0, equal_nan=True)


def test_mismatched_model_guard_rejected_before_inference(x64_context):
    with x64_context():
        fit, _, _ = case()
        guard = np.zeros(fit.observation.shape, bool); guard[..., 50] = True
        guarded = SpecFit(fit.observation, fit.specmodel, custom_guard=guard)
        options = dict(resolving_power=70000., use_gp=False,
            fixed=dict(rv=12., vsini=6., vmacro=3.), likelihood="gaussian_mixture",
            mismatch_fraction=.1, mismatch_scale=.1)
        model = make_model_single(fit, **options)
        with pytest.raises(ValueError, match="conflicts with SpecFit"):
            make_model_single(guarded, **options, mask_guard=~guard)
        with pytest.raises(ValueError, match="model mask_guard differs"):
            guarded.run_svi(model=model, rng_key=jax.random.PRNGKey(0), num_steps=1)
        with pytest.raises(ValueError, match="model mask_guard differs"):
            guarded.predict(model=model, point={})
        compatible = make_model_single(guarded, **options, mask_guard=guard)
        np.testing.assert_array_equal(compatible.settings['mask_guard'], guard)


def test_from_grid_forwards_guards(tmp_path, x64_context):
    from jaxstar.specfit import save_spectral_grid

    with x64_context():
        base = mg_case()
        path = tmp_path/'grid.npz'
        save_spectral_grid(path, base.specmodel.spectra)
        custom = np.zeros(base.observation.shape, bool); custom[..., 0] = True
        fit = SpecFit.from_grid(path, base.observation, mg_guard=True,
            custom_guard=custom, guard_rv=12., vmax=30.)
        direct = SpecFit(base.observation, base.specmodel, mg_guard=True,
                        custom_guard=custom, guard_rv=12.)
        np.testing.assert_array_equal(fit.mask_guard, direct.mask_guard)


def test_mg_settings_roundtrip_with_gaussian_model(tmp_path, x64_context):
    with x64_context():
        base = mg_case()
        custom = np.zeros(base.observation.shape, bool); custom[..., 0] = True
        fit = SpecFit(base.observation, base.specmodel, mg_guard=True,
                      guard_rv=12., custom_guard=custom)
        model = make_model_single(fit, resolving_power=70000., use_gp=False,
                                  fixed=dict(rv=12., vsini=6., vmacro=3.))
        save_fit_result(tmp_path, sf=fit, model=model, point={'teff':5800.})
        saved = load_fit_result(tmp_path)
        assert saved.metadata['mask_guard']['mg_guard'] is True
        assert saved.metadata['mask_guard']['guard_rv_km_s'] == 12.
        assert saved.metadata['model']['settings']['mask_guard'] is None
        restored = SpecFit(saved.observation, fit.specmodel, mg_guard=True,
            guard_rv=saved.metadata['mask_guard']['guard_rv_km_s'],
            custom_guard=saved.custom_guard)
        np.testing.assert_array_equal(restored.mask_guard, saved.mask_guard)
        np.testing.assert_array_equal(saved.mask_guard, fit.mask_guard)
        assert not saved.mask_guard.flags.writeable


def test_legacy_result_without_guard_arrays(tmp_path, x64_context):
    import hashlib
    import json

    with x64_context():
        fit, _, _ = case(single=True)
        guard = np.zeros(fit.observation.shape, bool); guard[50] = True
        model = make_model_single(fit, resolving_power=70000., use_gp=False,
            fixed=dict(rv=12., vsini=6., vmacro=3.), likelihood='gaussian_mixture',
            mismatch_fraction=.1, mismatch_scale=.1, mask_guard=guard)
        save_fit_result(tmp_path, sf=fit, model=model, point={'teff':5800.})
        path = tmp_path/'arrays.npz'
        with np.load(path) as data:
            old = {k:data[k] for k in data.files if k not in {'mask_guard','custom_guard'}}
        np.savez(path, **old)
        manifest_path = tmp_path/'result.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['arrays.npz'] = hashlib.sha256(path.read_bytes()).hexdigest()
        del manifest['metadata']['mask_guard']
        manifest_path.write_text(json.dumps(manifest))
        np.testing.assert_array_equal(load_fit_result(tmp_path).mask_guard, guard)
        del manifest['metadata']['model']['settings']['mask_guard']
        manifest_path.write_text(json.dumps(manifest))
        saved = load_fit_result(tmp_path)
        assert saved.mask_guard is None and saved.custom_guard is None


def test_prefit_wrapper_inherits_guard(monkeypatch, x64_context):
    with x64_context():
        base, _, _ = case(single=True)
        guard = np.zeros(base.observation.shape, bool); guard[50] = True
        fit = SpecFit(base.observation, base.specmodel, custom_guard=guard)
        calls = []
        def fake_svi(**kwargs):
            calls.append(kwargs)
            return kwargs['p_initial']
        monkeypatch.setattr(fit, 'run_svi', fake_svi)
        model, point = fit.run_mixture_prefit(rng_key=jax.random.PRNGKey(0),
            model_kwargs=dict(resolving_power=70000.,
                              fixed=dict(rv=12., vsini=6., vmacro=3.)),
            p_initial={'teff':5800.})
        assert len(calls) == 1 and calls[0]['model'] is model
        np.testing.assert_array_equal(model.settings['mask_guard'], guard)
        assert float(point['teff']) == 5800.


def test_sigma_clip_respects_custom_guard(x64_context):
    from test_mixture_masking import custom_case

    with x64_context():
        base, point, model, _ = custom_case()
        options = dict(point=point, model=model, method='sigma_clip', mask_v=10.)
        unprotected = base.mask_outliers(**options)
        assert unprotected[10]
        guard = np.zeros(base.observation.shape, bool); guard[10] = True
        fit = SpecFit(base.observation, base.specmodel, custom_guard=guard)
        assert not fit.mask_outliers(**options)[10]


@pytest.mark.parametrize("all_guarded", [False, True])
def test_residual_mixture_fits_only_unguarded_pixels(all_guarded, monkeypatch, x64_context):
    from test_mixture_masking import custom_case

    with x64_context():
        base, point, model, _ = custom_case()
        guard = np.full(base.observation.shape, all_guarded)
        guard[10] = True
        fit = SpecFit(base.observation, base.specmodel, custom_guard=guard)
        candidates = ~fit.observation.mask & ~guard
        calls = []

        def fit_residuals(residual, variance):
            calls.append(residual)
            np.testing.assert_array_equal(residual,
                (fit.observation.flux - point['baseline'])[candidates])
            np.testing.assert_array_equal(variance, np.full(candidates.sum(), .01**2))
            return dict(mismatch_fraction=.1, mismatch_scale=.1), np.full(residual.shape, .8)

        monkeypatch.setattr("jaxstar.specfit._outliers._fit_gaussian_mixture", fit_residuals)
        flags = fit.mask_outliers(point=point, model=model, extend_outlier_mask=False)
        assert len(calls) == int(not all_guarded)
        np.testing.assert_array_equal(flags, candidates)
        probability = fit.outlier_diagnostics['mismatch_probability']
        np.testing.assert_array_equal(probability[fit.mask_guard], 0.)
        assert np.isnan(probability[fit.observation.mask]).all()
