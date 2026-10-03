"""Measured-data validation, immutability, precision and JAX composition."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstar.specfit import Observation, SpecModel


def arrays(dtype=np.float32, regions=None):
    wave = np.array([5000., 5001., 5002.], dtype=dtype)
    flux = np.array([1.1, .8, -.1], dtype=dtype)
    error = np.full(3, .02, dtype=dtype)
    if regions is not None:
        wave = np.stack([wave + 1000*i for i in range(regions)])
        flux = np.broadcast_to(flux, wave.shape).copy()
        error = np.broadcast_to(error, wave.shape).copy()
    return wave, flux, error


def test_single_region_values_shapes_properties_and_default_mask():
    wave, flux, error = arrays()
    obs = Observation(wave, flux, error)
    assert obs.shape == (3,) and obs.ndim == 1
    assert obs.n_regions == 1 and obs.n_pixels == 3
    for actual, expected in zip((obs.wavelength, obs.flux, obs.uncertainty), (wave, flux, error)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(obs.mask, np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(obs.valid, np.ones(3, dtype=bool))
    assert obs.mask.dtype == bool
    assert obs.region is obs.order is obs.exposure is None


def test_rectangular_regions_and_distinct_region_order_metadata():
    wave, flux, error = arrays(regions=2)
    region, order = ['blue', 'red'], np.array([8, 8], dtype=np.int32)
    obs = Observation(wave, flux, error, [[False, True, False], [True, False, False]],
                      region=region, order=order, exposure='epoch1')
    assert obs.shape == (2, 3) and obs.ndim == 2
    assert obs.n_regions == 2 and obs.n_pixels == 3
    assert obs.region == ('blue', 'red') and obs.order == (8, 8)
    assert all(type(label) is int for label in obs.order)
    assert obs.exposure == 'epoch1'
    np.testing.assert_array_equal(obs.valid, [[True, False, True], [False, True, True]])
    region[0], order[0] = 'changed', 9
    assert obs.region == ('blue', 'red') and obs.order == (8, 8)


@pytest.mark.parametrize('shape', [(), (0,), (2, 0), (0, 3), (1, 1, 3)])
def test_unsupported_or_empty_shapes(shape):
    value = np.ones(shape)
    with pytest.raises(ValueError, match='nonempty.*shape'):
        Observation(value, value, value)


@pytest.mark.parametrize('field', ['wavelength', 'flux', 'uncertainty', 'mask'])
def test_shape_mismatches_are_not_broadcast_or_reshaped(field):
    wave, flux, error = arrays()
    values = dict(wavelength=wave, flux=flux, uncertainty=error, mask=np.zeros(3, dtype=bool))
    values[field] = np.full((1, 3), 1, dtype=bool if field == 'mask' else float)
    with pytest.raises(ValueError, match='shape'):
        Observation(**values)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf, -np.inf])
def test_uncertainty_must_be_finite_and_positive_on_usable_pixels(bad):
    wave, flux, error = arrays()
    error[1] = bad
    with pytest.raises(ValueError, match='uncertainty.*finite and positive.*usable'):
        Observation(wave, flux, error)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_flux_must_be_finite_on_usable_pixels(bad):
    wave, flux, error = arrays()
    flux[1] = bad
    with pytest.raises(ValueError, match='flux.*finite.*usable'):
        Observation(wave, flux, error)


def test_masked_invalid_flux_uncertainty_are_retained_without_replacement():
    wave, flux, error = arrays()
    flux[1:], error[1:] = [np.nan, np.inf], [-1, np.nan]
    obs = Observation(wave, flux, error, [False, True, True])
    np.testing.assert_array_equal(obs.flux, flux)
    np.testing.assert_array_equal(obs.uncertainty, error)
    np.testing.assert_array_equal(obs.valid, [True, False, False])
    # No requirement to have at least one usable pixel at this data-only layer.
    all_masked = Observation(wave, [np.nan]*3, [0, -1, np.inf], [True]*3)
    assert not np.any(all_masked.valid)


@pytest.mark.parametrize('mask', [[0, 1, 0], np.array([0., 1., 0.]), jnp.array([0, 1, 0])])
def test_numeric_binary_mask_conversion_is_explicit_and_not_inverted(mask):
    obs = Observation(*arrays(), mask=mask)
    assert obs.mask.dtype == bool
    np.testing.assert_array_equal(obs.mask, [False, True, False])


@pytest.mark.parametrize('mask', [[0, 2, 0], [0, -1, 0], [0, .5, 0], [0, np.nan, 0],
                                  [0, np.inf, 0], ['False', 'True', 'False'], [0j, 1j, 0j]])
def test_unsafe_mask_casts_are_rejected(mask):
    with pytest.raises(ValueError, match='mask must be Boolean.*0/1'):
        Observation(*arrays(), mask=mask)


@pytest.mark.parametrize('wave,message', [
    ([5000, np.nan, 5002], 'finite positive'),
    ([5000, np.inf, 5002], 'finite positive'),
    ([0, 1, 2], 'finite positive'),
    ([-1, 1, 2], 'finite positive'),
    ([5000, 5000, 5002], 'strictly increasing'),
    ([5002, 5001, 5000], 'strictly increasing'),
    (np.array([5002, 5001, 5000], dtype=np.uint32), 'strictly increasing'),
])
def test_wavelength_contract_includes_masked_pixels(wave, message):
    with pytest.raises(ValueError, match=message):
        Observation(wave, [np.nan]*3, [np.nan]*3, [True]*3)


def test_each_region_is_checked_independently_and_one_pixel_is_supported():
    wave, flux, error = arrays(regions=2)
    wave[1] = wave[1, ::-1]
    with pytest.raises(ValueError, match='strictly increasing'):
        Observation(wave, flux, error)
    assert Observation([5000.], [1.], [.1]).n_pixels == 1
    # Regions are not reordered or required to appear in increasing order.
    obs = Observation([[6000., 6001.], [5000., 5001.]], [[1., 1.]]*2, [[.1, .1]]*2)
    np.testing.assert_array_equal(obs.wavelength[:, 0], [6000., 5000.])


@pytest.mark.parametrize('field', ['wavelength', 'flux', 'uncertainty'])
@pytest.mark.parametrize('bad', [[True]*3, ['1']*3, [1+1j]*3, np.array([1, 2, 3], dtype=object)])
def test_non_real_numeric_arrays_are_rejected(field, bad):
    values = dict(zip(('wavelength', 'flux', 'uncertainty'), arrays()))
    values[field] = bad
    with pytest.raises(TypeError, match=field + '.*real numeric'):
        Observation(**values)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_precision_and_array_inputs(dtype, backend, x64_context):
    with x64_context(dtype == np.float64):
        before = jax.config.x64_enabled
        inputs = arrays(dtype)
        if backend == 'jax':
            inputs = tuple(jnp.asarray(value) for value in inputs)
        obs = Observation(*inputs)
        for original, actual in zip(inputs, (obs.wavelength, obs.flux, obs.uncertainty)):
            assert actual.dtype == dtype
            np.testing.assert_array_equal(actual, original)
            if backend == 'jax':
                assert actual is original
                assert actual.device == original.device
        assert jax.config.x64_enabled == before


def test_numpy_precision_preserved_with_x64_disabled_and_mixed_dtypes(x64_context):
    with x64_context(False):
        wave, flux, error = arrays(np.float64)
        obs = Observation(wave, flux.astype(np.float32), error)
        assert obs.wavelength.dtype == obs.uncertainty.dtype == np.float64
        assert obs.flux.dtype == np.float32
        assert not jax.config.x64_enabled
        # Transforms follow the caller's setting, rather than silently enabling x64.
        result = jax.jit(lambda o: o.flux.sum())(obs)
        assert result.dtype == np.float32
        assert not jax.config.x64_enabled


def test_integer_and_list_inputs_are_not_unnecessarily_promoted():
    obs = Observation([5000, 5001], [10, 20], [1, 2])
    assert obs.wavelength.dtype.kind == obs.flux.dtype.kind == obs.uncertainty.dtype.kind == 'i'
    np.testing.assert_array_equal(obs.flux, [10, 20])


def test_immutable_fields_and_independent_readonly_numpy_snapshots():
    wave, flux, error = arrays()
    mask = np.array([False, True, False])
    obs = Observation(wave, flux, error, mask)
    for name in ('wavelength', 'flux', 'uncertainty', 'mask'):
        value = getattr(obs, name)
        with pytest.raises(FrozenInstanceError):
            setattr(obs, name, value)
        with pytest.raises(ValueError, match='read-only'):
            value[0] = 0
    with pytest.raises(FrozenInstanceError):
        obs.exposure = 'changed'
    original = tuple(np.array(getattr(obs, name)) for name in ('wavelength', 'flux', 'uncertainty', 'mask'))
    wave[:], flux[:], error[:], mask[:] = 0, 0, 0, True
    for name, expected in zip(('wavelength', 'flux', 'uncertainty', 'mask'), original):
        np.testing.assert_array_equal(getattr(obs, name), expected)


def test_immutable_jax_inputs():
    inputs = tuple(jnp.asarray(value) for value in arrays())
    obs = Observation(*inputs)
    with pytest.raises(TypeError):
        obs.flux[0] = 0
    modified = inputs[1].at[0].set(0)
    assert modified[0] == 0 and obs.flux[0] != 0


def test_pytree_roundtrip_jit_dynamic_arrays_and_device_put():
    obs = Observation(*arrays(), mask=[False, True, False], region=('segment',), order=(8,), exposure='epoch1')
    leaves, structure = jax.tree_util.tree_flatten(obs)
    assert len(leaves) == 4 and all(hasattr(leaf, 'shape') for leaf in leaves)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert restored.region == ('segment',) and restored.order == (8,) and restored.exposure == 'epoch1'
    for name in ('wavelength', 'flux', 'uncertainty', 'mask'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(obs, name))
    compiled = jax.jit(lambda o: jnp.where(o.valid, o.flux, 0).sum())
    np.testing.assert_allclose(compiled(obs), obs.flux[obs.valid].sum())
    changed = Observation(obs.wavelength, obs.flux*2, obs.uncertainty, obs.mask,
                          region=obs.region, order=obs.order, exposure=obs.exposure)
    np.testing.assert_allclose(compiled(changed), compiled(obs)*2)
    device_obs = jax.device_put(obs)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(device_obs))
    output = jax.jit(lambda o: o)(device_obs)
    assert output.region == obs.region and output.order == obs.order and output.exposure == obs.exposure
    np.testing.assert_array_equal(output.mask, obs.mask)
    program = jax.make_jaxpr(lambda o: jnp.where(o.valid, o.flux, 0).sum())(obs)
    assert not program.consts


@pytest.mark.parametrize('name', ['region', 'order'])
@pytest.mark.parametrize('labels', [8, 'blue', (), ('a', 'b'), (('nested',),), (True,), (1.5,), (None,)])
def test_invalid_identifier_metadata(name, labels):
    with pytest.raises(ValueError, match=name):
        Observation(*arrays(), **{name: labels})


@pytest.mark.parametrize('label', ['', 1, ('epoch1',), ['epoch1']])
def test_invalid_exposure_metadata(label):
    with pytest.raises(ValueError, match='exposure.*string'):
        Observation(*arrays(), exposure=label)


def test_no_physical_parameters_and_direct_specmodel_composition():
    from test_specmodel import make_library, observations, parameters

    assert set(Observation.__dataclass_fields__) == {'wavelength', 'flux', 'uncertainty', 'mask', 'region', 'order', 'exposure'}
    with pytest.raises(TypeError, match='resolving_power'):
        Observation(*arrays(), resolving_power=70000.)
    library = make_library(np.float32)
    model, params, wave = SpecModel(library), parameters(), observations(library)
    obs = Observation(wave, np.ones_like(wave), np.full_like(wave, .01))
    np.testing.assert_array_equal(model(params, obs.wavelength), model(params, wave))
    predict = jax.jit(lambda m, p, o: m(p, o.wavelength))
    np.testing.assert_allclose(predict(model, params, obs), model(params, wave), rtol=2e-6, atol=2e-7)
