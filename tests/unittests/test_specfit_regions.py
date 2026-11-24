"""Native line extraction and small model grids with all atmosphere nodes."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from jaxstar.specfit import (Observation, SpecFit, SpecModel, extract_regions,
                             load_spectral_grid, save_spectral_grid, single_star_params)
from jaxstar.specfit import storage
from test_specfit import case


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def test_extract_native_pixels_masks_padding_and_source_mapping():
    wave = np.stack((np.linspace(5000, 5010, 101), np.linspace(5005, 5015, 101)))
    flux = np.arange(wave.size, dtype=float).reshape(wave.shape)
    error = np.full(wave.shape, .01)
    mask = np.zeros(wave.shape, bool)
    mask[1, 20] = True
    flux[mask], error[mask] = np.nan, np.inf
    obs = Observation(wave, flux, error, mask, region=(10, 11), exposure="one")
    bounds = [[5006.9, 5007.4], [5001., 5001.2], [5007., 5007.1]]
    extracted, mapping = extract_regions(obs, wavelength_ranges=bounds,
                                        regions=("a", "b", "c"), source_regions=(11, 10, 11))
    assert extracted.region == ("a", "b", "c") and extracted.exposure == "one"
    assert mapping.source_regions == (11, 10, 11)
    np.testing.assert_array_equal(mapping.source_indices, [1, 0, 1])
    assert not mapping.pixel_indices.flags.writeable
    assert np.any(mapping.pixel_indices == -1)
    for r, source in enumerate(mapping.source_indices):
        valid = mapping.pixel_indices[r] >= 0
        original = np.flatnonzero((wave[source] >= bounds[r][0]) & (wave[source] <= bounds[r][1]))
        np.testing.assert_array_equal(mapping.pixel_indices[r, valid], original)
        for name in ("wavelength", "flux", "uncertainty", "mask"):
            np.testing.assert_array_equal(getattr(extracted, name)[r, valid], getattr(obs, name)[source, original])
        assert extracted.mask[r, ~valid].all()
        assert np.isnan(extracted.flux[r, ~valid]).all()
        assert set(extracted.wavelength[r]).issubset(set(wave[source]))
    fit_mask = np.zeros(obs.shape, bool)
    fit_mask[1, 21] = True
    assert mapping.take(fit_mask, fill_value=True)[0, 2]
    np.testing.assert_array_equal(obs.flux, flux)
    with pytest.raises(ValueError, match="original Observation.shape"):
        mapping.take(fit_mask[0], fill_value=True)


def test_ambiguous_missing_and_invalid_windows_fail():
    wave = np.tile(np.linspace(5000., 5002., 21), (2, 1))
    obs = Observation(wave, np.ones_like(wave), np.ones_like(wave), region=(8, 9))
    with pytest.raises(ValueError, match="found 2"):
        extract_regions(obs, wavelength_ranges=[[5000.5, 5001.]])
    with pytest.raises(ValueError, match="found 0"):
        extract_regions(obs, wavelength_ranges=[[4999., 5001.]], source_regions=[8])
    for ranges in ([], [5000, 5001], [[5001, 5000]], [[5000, np.nan]]):
        with pytest.raises(ValueError):
            extract_regions(obs, wavelength_ranges=ranges)
    with pytest.raises(ValueError, match="at least one observed pixel"):
        extract_regions(obs, wavelength_ranges=[[5000.01, 5000.02]], source_regions=[8])
    one = Observation(wave[0], np.ones(21), np.ones(21))
    lines, mapping = extract_regions(one, wavelength_ranges=[[5000.2, 5000.4]])
    assert lines.ndim == 2 and mapping.source_regions == (0,) and lines.region == (0,)


def test_common_crop_selects_source_labels_before_jax_and_preserves_nodes(tmp_path, monkeypatch):
    fit, _, _ = case()
    original = fit.specmodel.spectra
    # Two overlapping orders with distinct fluxes, and float32 flux under x64.
    from jaxstar.grid import Field, RectilinearGrid
    wave = np.asarray(original.wavelength).copy()
    wave[1] = wave[0]
    values = np.asarray(original.grid.field("flux").values, dtype=np.float32).copy()
    values[..., 1, :] *= .99
    spectra = replace(original, wavelength=wave, grid=RectilinearGrid(
        axes={"teff": original.grid.axis("teff")},
        fields={"flux": Field(values, ("teff",), payload_dims=("region", "pixel"))}))
    path = save_spectral_grid(tmp_path / 'grid.npz', spectra)
    bounds = [[5000.1, 5000.5], [5000.2, 5000.9], [5000.1, 5000.5]]
    with pytest.raises(ValueError, match="found 2"):
        load_spectral_grid(path, wavelength=bounds, wav_margin=.7, crop=True)
    shapes = []
    real_grid = storage.RectilinearGrid
    def recording_grid(**kwargs):
        shapes.append(kwargs["fields"]["flux"].values.shape)
        return real_grid(**kwargs)
    monkeypatch.setattr(storage, 'RectilinearGrid', recording_grid)
    selected = load_spectral_grid(path, wavelength=bounds, wav_margin=.7, crop=True,
                                 source_regions=(9, 8, 9), regions=('first', 'second', 'third'))
    assert len(shapes) == 1 and shapes[0][-2] == 3 and shapes[0][-1] < wave.shape[1] / 2
    assert selected.grid.field('flux').values.dtype == np.float32
    np.testing.assert_array_equal(selected.grid.axis('teff'), original.grid.axis('teff'))
    for r, source in enumerate((1, 0, 1)):
        start = np.searchsorted(wave[source], selected.wavelength[r, 0])
        size = selected.wavelength.shape[1]
        np.testing.assert_array_equal(selected.grid.field('flux').values[..., r, :],
                                      values[..., source, start:start + size])
        assert selected.wavelength[r, 0] <= bounds[r][0] - .7
        assert selected.wavelength[r, -1] >= bounds[r][-1] + .7


@pytest.mark.parametrize('sampling', [{}, {'pixels': 211}, {'velocity_step': .7}])
def test_crop_full_forward_agree_for_free_atmosphere_and_region_rv(tmp_path, sampling):
    fit, _, _ = case()
    path = save_spectral_grid(tmp_path / 'grid.npz', fit.specmodel.spectra)
    obs, mapping = extract_regions(fit.observation,
        wavelength_ranges=[[5000.0, 5000.6], [6000.0, 6000.9], [5000.2, 5000.6]],
        regions=('line1', 'line2', 'line3'), source_regions=(8, 9, 8))
    full = SpecFit.from_grid(path, obs, source_regions=mapping.source_regions,
                            wav_margin=.9, vmax=30.)
    cropped = SpecFit.from_grid(path, obs, source_regions=mapping.source_regions,
                               crop=True, wav_margin=.9, vmax=30., **sampling)
    assert cropped.specmodel.spectra.wavelength.shape[1] < full.specmodel.spectra.wavelength.shape[1]
    if 'velocity_step' in sampling:
        spacing = np.diff(np.log(cropped.specmodel.spectra.wavelength), axis=-1) * 299792.458
        np.testing.assert_allclose(spacing, .7, atol=1e-8)
    if sampling:
        # Check interpolation independently at every retained atmosphere node.
        # Resampling changes the numerical broadening grid; unlike a native
        # crop, it need not reproduce the original coarse-grid model exactly.
        original = fit.specmodel.spectra
        target = np.asarray(cropped.specmodel.spectra.wavelength)
        expected_nodes = np.stack([
            np.stack([np.interp(target[r], original.wavelength[source], node[source])
                      for r, source in enumerate(mapping.source_indices)])
            for node in np.asarray(original.grid.field('flux').values)])
        np.testing.assert_allclose(cropped.specmodel.spectra.grid.field('flux').values,
                                   expected_nodes, atol=1e-14)
    for teff in (5000., 5800., 6500.):
        params = single_star_params({'teff': teff}, vsini=6., vmacro=3., q1=.36, q2=.3,
                                    rv=np.array([12., 9., 6.]), resolving_power=70000.,
                                    depth_scale=np.array([.8, 1.2, 1.]))
        expected = full.specmodel(params, obs.wavelength)
        actual = jax.jit(lambda p: cropped.specmodel(p, obs.wavelength))(params)
        np.testing.assert_allclose(actual, expected, atol=1e-3 if sampling else 2e-10)


def test_insufficient_cropped_model_margin_fails_at_evaluation(tmp_path):
    fit, _, _ = case()
    path = save_spectral_grid(tmp_path / 'grid.npz', fit.specmodel.spectra)
    obs, mapping = extract_regions(fit.observation, wavelength_ranges=[[4999.4, 5000.9]])
    sf = SpecFit.from_grid(path, obs, crop=True, wav_margin=.05, vmax=30.,
                          source_regions=mapping.source_regions)
    params = single_star_params({'teff': 5800.}, vsini=6., vmacro=3., q1=.36, q2=.3,
                                rv=12., resolving_power=70000.)
    with pytest.raises(ValueError, match='insufficient model wavelength coverage'):
        sf.specmodel(params, obs.wavelength)


@pytest.mark.parametrize('variable', [False, True])
def test_cropped_fixed_ip_matches_full_model_with_order_mapping(tmp_path, variable):
    fit, _, _ = case()
    path = save_spectral_grid(tmp_path / 'grid.npz', fit.specmodel.spectra)
    obs, mapping = extract_regions(fit.observation,
        wavelength_ranges=[[5000., 5000.6], [6000., 6000.8], [5000.1, 5000.5]],
        regions=('a', 'b', 'c'))
    sigma = np.array([1.1, 1.8])[mapping.source_indices]
    constant = dict(mu1_kms=0., mu2_kms=.2, sigma1_kms=sigma,
                    sigma2_kms=2.3, fraction2=.1)
    centers = np.array([5000., 6000.])[mapping.source_indices]
    def calibration(wave):
        return {**constant, 'sigma1_kms': sigma[:, None] * (wave / centers[:, None])}
    options = dict(source_regions=mapping.source_regions, wav_margin=1.4,
                   vmax=30., fixed_ip=calibration if variable else constant)
    full = SpecFit.from_grid(path, obs, **options)
    small = SpecFit.from_grid(path, obs, crop=True, **options)
    params = single_star_params({'teff': 5800.}, vsini=6., vmacro=3., q1=.36, q2=.3,
                                rv=np.array([12., 8., 6.]), depth_scale=np.array([1.2, .8, 1.]))
    np.testing.assert_allclose(small.specmodel(params, obs.wavelength),
                              full.specmodel(params, obs.wavelength), atol=2e-10)


@pytest.mark.parametrize('options, message', [
    ({'crop': True}, 'requires wavelength'),
    ({'source_regions': [8]}, 'requires wavelength'),
    ({'crop': 'yes'}, 'crop must be a bool'),
])
def test_invalid_crop_configuration(tmp_path, options, message):
    with pytest.raises((ValueError, TypeError), match=message):
        load_spectral_grid(tmp_path/'missing.npz', **options)
