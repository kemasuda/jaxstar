"""Ragged native grids: field support, derivative stencils and inference."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from jaxstar.mistfit import MistFit, MistModel, plot_track
from jaxstar.mistfit.mistgrid.tracks import masked_age_derivative
from test_mist_tracks import track_path, track_precision, PARAMS, LATENTS, OPTIONS, native_fields


@pytest.fixture
def masked_path(track_path):
    data = native_fields(track_path)
    present = np.ones(data['age_gyr'].shape, dtype=bool)
    # A short native track, an absent mass/FeH pair, and a photometry-only hole.
    present[0, 0, 3:] = False
    present[-1, -1] = False
    for name, values in data.items():
        if values.shape[:3] == present.shape:
            values[~present] = np.nan
    data['absolute_magnitudes'][1, 1, 2] = np.nan
    np.savez(track_path, **data, track_present=present,
             metadata_json=np.array(json.dumps(dict(kind='track', format_version=2))))
    return track_path


def test_ragged_derivatives_do_not_bridge_gaps_or_reversals():
    age = np.array([[1., 4., 9., 16., np.nan, np.nan],
                    [1., 4., 9., 8., 25., 36.],
                    [1., 4., np.nan, 16., 25., 36.]])
    present = np.ones(age.shape, bool)
    present[0, 4:] = False
    actual = masked_age_derivative(age, np.arange(1., 7.), present)
    np.testing.assert_array_equal(actual[0], [4., 4., 6., 6., np.nan, np.nan])
    np.testing.assert_array_equal(actual[1], [4., 4., np.nan, np.nan, 14., 14.])
    np.testing.assert_array_equal(actual[2], [np.nan, np.nan, np.nan, np.nan, 10., 10.])


def test_masked_exact_nodes_and_field_specific_support(masked_path):
    model = MistModel.from_tracks(masked_path)
    p = dict(PARAMS, mass_init=1., feh_init=0., eep=400.)
    result = model(p, keys=('age', 'dtdeep', 'teff', 'kmag_abs'))
    assert all(np.isfinite(result[k]) for k in ('age', 'dtdeep', 'teff'))
    assert np.isneginf(result['kmag_abs'])
    # Zero-weight missing neighbors must not poison a valid node or face.
    valid = model(dict(p, eep=350.), keys='kmag_abs')['kmag_abs']
    assert np.isfinite(valid)
    assert np.isneginf(model(dict(p, eep=350.000001), keys='kmag_abs')['kmag_abs'])
    endpoint = dict(PARAMS, mass_init=.8, feh_init=-.4, eep=400.)
    assert np.isfinite(model(endpoint, keys='dtdeep')['dtdeep'])
    assert np.isneginf(model(dict(endpoint, eep=400.01), keys='dtdeep')['dtdeep'])
    # Native track exists at this face, even though the next FeH plane is absent.
    assert np.isfinite(model(dict(p, mass_init=1.4, feh_init=0.), keys='age')['age'])
    assert np.isneginf(model(dict(p, mass_init=1.4, feh_init=.01), keys='age')['age'])
    assert 'track_present' not in model.output_keys


def test_masked_jit_vmap_pytree_and_gradients(masked_path):
    model = MistModel.from_tracks(masked_path)
    leaves, tree = jax.tree_util.tree_flatten(model)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    def evaluate(m, p):
        return jnp.stack(tuple(m(dict(zip(PARAMS, p)), keys=('age', 'radius', 'kmag', 'dtdeep')).values()))
    points = jnp.array([[1., 0., 350., .15], [1., 0., 399., .15],
                        [.8, -.4, 420., .15], [1.41, .4, 400., .15]])
    values = jax.jit(jax.vmap(evaluate, in_axes=(None, 0)))(restored, points)
    assert np.isfinite(values[0]).all()
    assert np.isneginf(values[1, 2]) and np.isfinite(values[1, 0])
    assert np.isneginf(values[2:]).all()
    grads = jax.jit(jax.vmap(jax.jacrev(evaluate, argnums=1), in_axes=(None, 0)))(model, points)
    assert np.isfinite(grads).all()
    assert np.all(np.asarray(grads)[2:] == 0)


@pytest.mark.parametrize('nodata,observe_mag,finite', [(True, True, True), (False, False, True), (False, True, False)])
def test_missing_saved_outputs_do_not_change_density(masked_path, nodata, observe_mag, finite):
    model = MistModel.from_tracks(masked_path)
    observations = {'parallax': (6.7, .1), 'teff': (6100., 120.)}
    if observe_mag:
        observations['kmag'] = (8., .1)
    fit = MistFit(model=model, observations=observations)
    latents = dict(LATENTS, mass=1., feh_init=0., eep=399.)
    options = dict(OPTIONS, nodata=nodata)
    def density(vector, extra=()):
        return log_density(fit.model, (), dict(options, output_keys=extra), dict(zip(latents, vector)))[0]
    point = jnp.array(list(latents.values()))
    value, gradient = jax.jit(jax.value_and_grad(density))(point)
    assert bool(np.isfinite(value)) == finite
    assert np.isfinite(gradient).all()
    extra = density(point, ('kmag_abs', 'rho'))
    np.testing.assert_array_equal(value, extra)


@pytest.mark.parametrize('options', [dict(prot=15.), dict(rho_obs=1., rho_err=.1), dict(age_obs=2., age_err=.1)])
def test_optional_constraints_reject_missing_support_without_nan(masked_path, options):
    model = MistModel.from_tracks(masked_path)
    fit = MistFit(model=model, observations={'parallax': (6.7, .1)})
    latents = dict(LATENTS, mass=1.39, feh_init=.3, eep=399.)
    def density(p):
        return log_density(fit.model, (), dict(OPTIONS, massmax=1.4, **options), dict(zip(latents, p)))[0]
    val, grad = jax.jit(jax.value_and_grad(density))(jnp.array(list(latents.values())))
    assert np.isneginf(val)
    assert np.isfinite(grad).all()


def test_plot_preserves_missing_segments_and_age_endpoints(masked_path):
    import matplotlib.pyplot as plt
    model = MistModel.from_tracks(masked_path)
    fig = plot_track(model, 1., 0., x='eep', y='kmag_abs', n_points=15)
    x, y = fig.axes[0].lines[0].get_data()
    assert np.isnan(y).any()
    assert np.isnan(y[(x > 350.) & (x < 450.)]).all()
    assert np.isfinite(y[(x <= 350.) | (x >= 450.)]).all()
    fig2 = plot_track(model, .8, -.4, x='age', y='eep', age_range=(1., 3.), n_points=10)
    ages, eeps = fig2.axes[0].lines[0].get_data()
    np.testing.assert_allclose(ages[[0, -1]], [1., 3.], rtol=1e-12)
    assert (eeps <= 400.).all()
    with pytest.raises(ValueError, match='no supported'):
        plot_track(model, 1.4, .4, x='eep', y='mass')
    plt.close(fig); plt.close(fig2)


@pytest.mark.parametrize('fault', ['mask_missing', 'mask_dtype', 'value_outside', 'inf', 'metadata_missing'])
def test_masked_schema_rejects_ambiguous_or_corrupt_fields(masked_path, fault):
    data = native_fields(masked_path)
    if fault == 'mask_missing':
        del data['track_present']
    elif fault == 'mask_dtype':
        data['track_present'] = data['track_present'].astype(float)
    elif fault == 'value_outside':
        data['logR'][-1, -1, 0] = 0.
    elif fault == 'inf':
        data['logR'][0, 0, 0] = np.inf
    else:
        del data['metadata_json']
    np.savez(masked_path, **data)
    with pytest.raises(ValueError):
        MistModel.from_tracks(masked_path)
