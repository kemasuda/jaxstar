"""Offline directory selection, model factories and observed RV coverage."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from jaxstar.specfit import (
    Observation, SpecFit, SpecModel, select_grid_paths,
    load_coelho, load_bosz, load_tlusty, resample_spectral_grid,
    save_spectral_grid,
)


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def write_grids(root, library="coelho", dtype=np.float64):
    axes = dict(tgrid=[5500., 6000.], ggrid=[4.])
    if library == "coelho":
        axes.update(fgrid=[0.], agrid=[0.])
    elif library == "bosz":
        axes.update(mgrid=[0.], agrid=[0.], cgrid=[0.], vgrid=[1.])
    else:
        axes.update(zgrid=[0.])
    shape = tuple(len(axis) for axis in axes.values())
    paths, wave = [], []
    for i, center in enumerate((5000., 6000.)):
        source = (center * np.linspace(.998, 1.002, 401)).astype(dtype)
        profile = np.exp(-.5 * ((source / center - 1) / .00004)**2)
        flux = np.stack([1 - depth * profile for depth in (.2, .5)])
        path = root / ("z_normed.npz" if i == 0 else "a_normed.npz")
        np.savez(path, **{name: np.asarray(a, dtype=dtype) for name, a in axes.items()},
                 wavgrid=source, flux=flux.reshape(shape + (401,)).astype(dtype))
        paths.append(path)
        wave.append(center * np.linspace(.999, 1.0005, 31))
    return paths, np.asarray(wave, dtype=dtype)


def observation(wave, regions=None):
    mask = np.zeros(wave.shape, bool)
    mask[..., 0] = True
    return Observation(wave, np.ones_like(wave), np.full_like(wave, .01),
                       mask=mask, region=regions, exposure="fixture")


def params(model):
    return {"components": ({
        "atmosphere": {name: np.mean(model.spectra.grid.axis(name))
                       for name in model.spectra.grid.axis_names},
        "broadening": dict(vsini=2., vmacro=1., u1=.5, u2=.2), "rv": 0.,
    },), "instrument": {"resolving_power": 70000.}}


def test_selection_uses_stored_wavelength_and_input_order(tmp_path):
    paths, wave = write_grids(tmp_path)
    selected = select_grid_paths(tmp_path, wave[::-1], regions=("red", "blue"))
    assert selected == paths[::-1]  # deliberately unrelated filenames
    assert select_grid_paths(tmp_path, wave[0]) == paths[:1]
    assert select_grid_paths(tmp_path, wave[[0, 0]], regions=(8, 8)) == [paths[0]] * 2
    # Selection does not load flux arrays (this object payload cannot be read
    # with allow_pickle=False) and includes an exactly satisfied margin.
    np.savez(tmp_path / "only.npz", wavgrid=np.array([10., 15., 20.]),
             flux=np.array([object()], dtype=object))
    assert select_grid_paths(tmp_path, [13., 17.], pattern="only.npz") == [tmp_path / "only.npz"]
    assert select_grid_paths(tmp_path, [10., 20.], wav_margin=0.,
                             pattern="only.npz") == [tmp_path / "only.npz"]


def test_missing_ambiguous_and_malformed_candidates(tmp_path):
    with pytest.raises(FileNotFoundError, match="No grids matching"):
        select_grid_paths(tmp_path, [5000., 5001.])
    paths, wave = write_grids(tmp_path)
    with pytest.raises(ValueError, match="Region 'blue'.*found 0") as error:
        select_grid_paths(tmp_path, wave[:1], regions=("blue",), wav_margin=6.)
    assert paths[0].name in str(error.value)
    assert "Angstrom" in str(error.value)
    with np.load(paths[0]) as data:
        np.savez(tmp_path / "overlap.npz", wavgrid=data["wavgrid"])
    with pytest.raises(ValueError, match="found 2") as error:
        select_grid_paths(tmp_path, wave[0])
    assert "overlap.npz" in str(error.value)
    assert select_grid_paths(tmp_path, wave, pattern="*_normed.npz") == paths
    np.savez(tmp_path / "bad.npz", wavelength=[5000., 5001.])
    with pytest.raises(ValueError, match="bad.npz.*wavgrid"):
        select_grid_paths(tmp_path, wave, pattern="bad.npz")
    np.savez(tmp_path / "bad.npz", wavgrid=[5002., 5001.])
    with pytest.raises(ValueError, match="bad.npz.*strictly increasing"):
        select_grid_paths(tmp_path, wave, pattern="bad.npz")


@pytest.mark.parametrize("wave,options,message", [
    ([], {}, "non-empty"),
    ([[[5000., 5001.]]], {}, "shape"),
    ([5001., 5000.], {}, "strictly increasing"),
    ([5000., np.nan], {}, "finite"),
    ([0., 1.], {}, "positive"),
    ([5000., 5001.], {"wav_margin": -1.}, "wav_margin"),
    ([5000., 5001.], {"wav_margin": np.inf}, "wav_margin"),
    ([5000., 5001.], {"regions": (8, 9)}, "one identifier"),
])
def test_invalid_selection_requests(tmp_path, wave, options, message):
    with pytest.raises(ValueError, match=message):
        select_grid_paths(tmp_path, wave, **options)


@pytest.mark.parametrize("library,loader", [
    ("coelho", load_coelho), ("bosz", load_bosz), ("tlusty", load_tlusty),
])
@pytest.mark.parametrize("single", [False, True])
def test_factories_match_manual_setup(tmp_path, library, loader, single):
    paths, wave = write_grids(tmp_path, library)
    paths, wave = (paths[:1], wave[0]) if single else (paths[::-1], wave[::-1])
    labels = (8,) if single else (9, 8)
    options = dict(library=library, pattern="*_normed.npz", wav_margin=3.,
                   pixels=501, vmax=30., wavelength_medium="air", flux_kind="normalized")
    manual = SpecModel(resample_spectral_grid(
        loader(paths, regions=labels, wavelength_medium="air", flux_kind="normalized"),
        pixels=501), vmax=30.)
    model = SpecModel.from_grid(tmp_path, wave, regions=np.asarray(labels), **options)
    obs = observation(wave, labels)
    fit = SpecFit.from_grid(tmp_path, obs, **options)
    assert fit.observation is obs
    assert fit.regions == labels
    point = params(manual)
    expected = manual(point, wave)
    for result in (model, fit.specmodel):
        assert result.spectra.sources == tuple(str(path) for path in paths)
        assert result.spectra.regions == labels
        assert result.spectra.wavelength_medium == "air"
        assert result.spectra.flux_kind == "normalized"
        assert result.spectra.wavelength.shape == (len(paths), 501)
        np.testing.assert_allclose(result(point, wave), expected, atol=1e-12)
    np.testing.assert_array_equal(fit.rv_bounds, SpecFit(obs, manual).rv_bounds)


def test_velocity_sampling_unlabeled_observation_and_jit(tmp_path):
    paths, wave = write_grids(tmp_path)
    fit = SpecFit.from_grid(tmp_path, observation(wave), velocity_step=2., vmax=30.)
    assert fit.regions == (0, 1)
    expected = resample_spectral_grid(load_coelho(paths), velocity_step=2.)
    np.testing.assert_array_equal(fit.specmodel.spectra.wavelength, expected.wavelength)
    model, point = fit.specmodel, params(fit.specmodel)
    np.testing.assert_allclose(jax.jit(lambda m, p, w: m(p, w))(model, point, wave),
                               model(point, wave), atol=1e-12)
    with pytest.raises(ValueError, match="exactly one"):
        SpecModel.from_grid(tmp_path, wave)
    with pytest.raises(ValueError, match="exactly one"):
        SpecModel.from_grid(tmp_path, wave, pixels=500, velocity_step=2.)
    with pytest.raises(ValueError, match="library must"):
        SpecModel.from_grid(tmp_path, wave, library="unknown", pixels=500)
    with pytest.raises(TypeError, match="Observation"):
        SpecFit.from_grid(tmp_path, None, pixels=500)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rv_bounds_describe_actual_broadened_coverage(tmp_path, dtype):
    _, wave = write_grids(tmp_path, dtype=dtype)
    fit = SpecFit.from_grid(tmp_path, observation(wave), pixels=501, vmax=30.)
    model, point = fit.specmodel, params(fit.specmodel)
    assert fit.rv_bounds.shape == (2, 2)
    assert not fit.rv_bounds.flags.writeable
    assert np.all(fit.rv_bounds[:, 0] < 0)
    assert np.all(fit.rv_bounds[:, 1] > 0)
    # Independent endpoint check using the forward relativistic Doppler factor.
    half = model.velocity_grid.shape[-1] // 2
    safe = np.asarray(model.spectra.wavelength, dtype=np.float64)[:, half:-half]
    beta = fit.rv_bounds / 299792.458
    factor = np.sqrt((1 + beta) / (1 - beta))
    np.testing.assert_allclose(safe[:, -1] * factor[:, 0], wave[:, -1], atol=1e-10)
    np.testing.assert_allclose(safe[:, 0] * factor[:, 1], wave[:, 0], atol=1e-10)
    compiled = jax.jit(lambda rv: model(
        {**point, "components": ({**point["components"][0], "rv": rv},)}, wave))
    for side, inward in ((0, .1), (1, -.1)):
        rv = fit.rv_bounds[:, side] + inward
        assert np.isfinite(compiled(rv)).all()
        outside = rv.copy()
        outside[0] = fit.rv_bounds[0, side] - inward
        point["components"][0]["rv"] = outside
        with pytest.raises(ValueError, match="insufficient model wavelength coverage"):
            model(point, wave)
    # Masks do not alter the full array evaluated by SpecModel.
    more_masked = replace(fit.observation, mask=np.ones_like(wave, dtype=bool))
    np.testing.assert_array_equal(SpecFit(more_masked, model).rv_bounds, fit.rv_bounds)


def test_empty_rv_interval_and_custom_broadening(tmp_path):
    paths, wave = write_grids(tmp_path)
    model = SpecModel(resample_spectral_grid(load_coelho(paths), pixels=501), vmax=30.)
    full_wave = np.asarray(model.spectra.wavelength)
    fit = SpecFit(observation(full_wave), model)
    assert np.all(fit.rv_bounds[:, 0] > fit.rv_bounds[:, 1])
    with pytest.raises(ValueError, match="coverage"):
        model(params(model), full_wave)

    def unbroadened(wavelength, flux, **kwargs):
        return wavelength, flux

    custom = SpecFit.from_grid(tmp_path, observation(wave), pixels=501,
                               broadening_operator=unbroadened)
    assert custom.rv_bounds is None
    point = params(custom.specmodel)
    assert np.isfinite(custom.specmodel(point, wave)).all()
    point["components"][0]["rv"] = 2000.
    with pytest.raises(ValueError, match="coverage"):
        custom.specmodel(point, wave)


@pytest.mark.parametrize("library,loader", [
    ("coelho", load_coelho), ("bosz", load_bosz), ("tlusty", load_tlusty),
])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_common_npz_factories_match_selected_legacy_models(tmp_path, library, loader, single, dtype):
    paths, wave = write_grids(tmp_path, library, dtype)
    spectra = resample_spectral_grid(loader(
        paths, regions=("blue", "red"), wavelength_medium="air", flux_kind="normalized"),
        pixels=501)
    path = save_spectral_grid(tmp_path / "common.npz", spectra)
    indices = [1] if single else [1, 0, 1]
    selected_wave = wave[1] if single else wave[indices]
    labels = (9,) if single else (9, 8, 9)
    obs = observation(selected_wave, labels)
    # No library/sampling/medium/product declarations: use saved information.
    fit = SpecFit.from_grid(path, obs, vmax=30.)
    model = SpecModel.from_grid(path, selected_wave, regions=np.asarray(labels), vmax=30.)
    manual = SpecModel(resample_spectral_grid(loader(
        [paths[i] for i in indices], regions=labels, wavelength_medium="air",
        flux_kind="normalized"), pixels=501), vmax=30.)
    point = params(manual)
    expected = manual(point, selected_wave)
    assert fit.observation is obs
    assert fit.regions == labels
    for result in (fit.specmodel, model):
        assert result.spectra.regions == labels
        assert result.spectra.library == library
        assert result.spectra.wavelength_medium == "air"
        assert result.spectra.flux_kind == "normalized"
        assert result.spectra.sources == spectra.sources
        assert result.spectra.grid.field("flux").values.dtype == dtype
        np.testing.assert_array_equal(result.spectra.wavelength, manual.spectra.wavelength)
        np.testing.assert_array_equal(result.spectra.grid.field("flux").values,
                                      manual.spectra.grid.field("flux").values)
        np.testing.assert_allclose(result(point, selected_wave), expected, atol=1e-12)
    np.testing.assert_array_equal(fit.rv_bounds, SpecFit(obs, manual).rv_bounds)


@pytest.mark.parametrize("sampling", [{"pixels": 601}, {"velocity_step": 2.}])
def test_common_npz_optional_resampling_and_saved_labels(tmp_path, sampling):
    paths, wave = write_grids(tmp_path, "bosz")
    # A non-log-uniform prepared artifact needs explicit resampling to fit.
    path = save_spectral_grid(tmp_path / "common.npz", load_bosz(paths, regions=(8, 9)))
    with pytest.raises(ValueError, match="log-uniform"):
        SpecFit.from_grid(path, observation(wave))
    fit = SpecFit.from_grid(path, observation(wave[::-1]), vmax=30., **sampling)
    assert fit.regions == (9, 8)
    manual = resample_spectral_grid(load_bosz(paths[::-1], regions=(9, 8)), **sampling)
    np.testing.assert_array_equal(fit.specmodel.spectra.wavelength, manual.wavelength)
    np.testing.assert_array_equal(fit.specmodel.spectra.grid.field("flux").values,
                                  manual.grid.field("flux").values)
    with pytest.raises(ValueError, match="exactly one"):
        SpecFit.from_grid(path, observation(wave), pixels=500, velocity_step=2.)


def test_common_npz_declarations_masks_and_jit(tmp_path):
    paths, wave = write_grids(tmp_path, "bosz")
    spectra = resample_spectral_grid(load_bosz(
        paths, wavelength_medium="air", flux_kind="normalized"), pixels=501)
    path = save_spectral_grid(tmp_path / "common.npz", spectra)
    fit = SpecFit.from_grid(path, observation(wave), library="bosz",
                            wavelength_medium="air", flux_kind="normalized", vmax=30.)
    model, point = fit.specmodel, params(fit.specmodel)
    np.testing.assert_allclose(jax.jit(lambda m, p, w: m(p, w))(model, point, wave),
                               model(point, wave), atol=1e-12)
    for options in ({"library": "coelho"}, {"wavelength_medium": "vacuum"},
                    {"flux_kind": "unnormalized"}):
        with pytest.raises(ValueError, match="conflicts with saved"):
            SpecFit.from_grid(path, observation(wave), **options)
    full = np.asarray(spectra.wavelength)
    obs = observation(full)
    # The first pixel is masked, but selection still requires its margin.
    with pytest.raises(ValueError, match="found 0"):
        SpecFit.from_grid(path, obs, wav_margin=.01)
    np.testing.assert_array_equal(
        SpecFit.from_grid(path, obs, wav_margin=0., vmax=30.).rv_bounds,
        SpecFit.from_grid(path, replace(obs, mask=np.ones_like(full, dtype=bool)),
                          wav_margin=0., vmax=30.).rv_bounds)


def test_common_npz_custom_broadening_and_missing_path(tmp_path):
    paths, wave = write_grids(tmp_path)
    spectra = replace(resample_spectral_grid(load_coelho(paths), pixels=501),
                      library="custom_synthesis")
    path = save_spectral_grid(tmp_path / "common.npz",
                              spectra)

    def unbroadened(wavelength, flux, **kwargs):
        return wavelength, flux

    fit = SpecFit.from_grid(path, observation(wave), broadening_operator=unbroadened)
    assert fit.specmodel.spectra.library == "custom_synthesis"
    assert fit.rv_bounds is None
    assert np.isfinite(fit.specmodel(params(fit.specmodel), wave)).all()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SpecFit.from_grid(tmp_path / "missing.npz", observation(wave))
    with pytest.raises(ValueError, match="not a common prepared"):
        SpecFit.from_grid(paths[0], observation(wave[0]))
    with pytest.raises(ValueError, match="wavelength is required"):
        SpecModel.from_grid(path, None)
