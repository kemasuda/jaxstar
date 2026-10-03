"""Offline synthetic tests of native files and prepared spectral-library grids."""

from dataclasses import FrozenInstanceError
from itertools import product
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from jaxstar.grid import RectilinearGrid
from jaxstar.specfit import (
    load_bosz, load_coelho, load_tlusty,
    prepare_bosz, prepare_coelho, prepare_tlusty,
    load_spectral_grid, save_spectral_grid,
    is_log_uniform, resample_spectral_grid,
)


SCHEMAS = {
    "coelho": (("teff", "tgrid"), ("logg", "ggrid"), ("feh", "fgrid"), ("alpha", "agrid")),
    "bosz": (("teff", "tgrid"), ("logg", "ggrid"), ("mh", "mgrid"),
             ("alpha", "agrid"), ("carbon", "cgrid"), ("vmic", "vgrid")),
    "tlusty": (("teff", "tgrid"), ("logg", "ggrid"), ("logZ", "zgrid")),
}
LOADERS = {"coelho": load_coelho, "bosz": load_bosz, "tlusty": load_tlusty}


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


@pytest.fixture(params=tuple(SCHEMAS))
def prepared(request, tmp_path):
    library = request.param
    values = {
        "teff": [5000, 5250, 6000], "logg": [3.5, 4.5], "feh": [-0.5, 0],
        "mh": [-0.5, 0], "alpha": [0, 0.4], "carbon": [0], "vmic": [0, 2],
        "logZ": np.log10([0.1, 0.5, 1]),
    }
    axes = {name: np.asarray(values[name], dtype=np.float64) for name, _ in SCHEMAS[library]}
    dtype = np.float64 if library == "coelho" else np.float32
    base = np.zeros(tuple(len(axis) for axis in axes.values()), dtype=np.float64)
    gradients = []
    for i, axis in enumerate(axes.values()):
        shape = [1] * len(axes)
        shape[i] = len(axis)
        step = axis[1] - axis[0] if len(axis) > 1 else 1
        base += (i + 1) * 0.1 * ((axis - axis[0]) / step).reshape(shape)
        gradients.append((i + 1) * 0.1 / step if len(axis) > 1 else 0)
    paths, payloads = [], []
    for region in range(2):
        wavelength = np.array([5000, 5000.2, 5000.6, 5001, 5001.7, 5002.3]) + region * 1000
        flux = (base[..., None] + 0.7 + 0.1 * region + 0.01 * np.arange(6)).astype(dtype)
        payload = {key: axes[name] for name, key in SCHEMAS[library]}
        payload.update(wavgrid=wavelength, flux=flux)
        path = tmp_path / f"{library}_{region}.npz"
        np.savez(path, **dict(reversed(tuple(payload.items()))))
        paths.append(path)
        payloads.append(payload)
    return library, axes, paths, payloads, np.array(gradients)


def test_prepared_axes_payload_wavelength_and_exact_nodes(prepared):
    name, axes, paths, data, _ = prepared
    library = LOADERS[name](paths[::-1], regions=(8, 8), wavelength_medium="vacuum")
    assert isinstance(library.grid, RectilinearGrid)
    assert library.grid.axis_names == tuple(axes)
    assert library.grid.axis_kinds[0] == "nonuniform"
    assert library.regions == (8, 8)
    assert library.wavelength_unit == "angstrom"
    assert library.wavelength_medium == "vacuum"
    assert library.flux_kind == "as_stored"
    assert library.library == name
    assert library.sources == tuple(str(path) for path in paths[::-1])
    field = library.grid.field("flux")
    assert field.dims == tuple(axes)
    assert field.payload_dims == ("region", "pixel")
    assert field.values.shape == tuple(len(axis) for axis in axes.values()) + (2, 6)
    for axis_name, axis in axes.items():
        np.testing.assert_array_equal(library.grid.axis(axis_name), axis)
    np.testing.assert_array_equal(library.wavelength, np.stack([d["wavgrid"] for d in data[::-1]]))
    point = {axis_name: axis[-1] for axis_name, axis in axes.items()}
    result = library.grid.interpolate(point)["flux"]
    index = tuple(-1 for _ in axes)
    np.testing.assert_array_equal(result, np.stack([d["flux"][index] for d in data[::-1]]))
    assert result.dtype == data[0]["flux"].dtype


def test_prepared_interior_batch_jit_and_coordinate_gradients(prepared):
    name, axes, paths, data, gradients = prepared
    library = LOADERS[name](paths)
    point = jnp.array([a[0] + 0.3 * (a[1] - a[0]) if len(a) > 1 else a[0] for a in axes.values()])
    lower = jnp.array([a[0] for a in axes.values()])

    def evaluate(lib, parameters):
        return lib.grid.interpolate({n: parameters[..., i] for i, n in enumerate(axes)})["flux"]

    expected = np.stack([d["flux"][tuple(0 for _ in axes)] for d in data])
    increment = sum(0.03 * (i + 1) for i, a in enumerate(axes.values()) if len(a) > 1)
    for function in (evaluate, jax.jit(evaluate)):
        np.testing.assert_allclose(function(library, point), expected + increment, rtol=2e-6)
        result = function(library, jnp.stack((point, lower)))
        assert result.shape == (2, 2, 6)
        np.testing.assert_allclose(result, np.stack((expected + increment, expected)), rtol=2e-6)
    gradient = jax.jit(jax.grad(lambda p: evaluate(library, p).sum()))(point)
    np.testing.assert_allclose(gradient, gradients * 12, rtol=2e-5, atol=1e-7)
    outside = point.at[0].set(axes["teff"][0] - 1)
    assert np.all(np.isneginf(evaluate(library, outside)))


def test_prepared_pixel_windows_are_exact_and_not_resampled(prepared):
    name, _, paths, data, _ = prepared
    library = LOADERS[name](paths, pixel_slices=(slice(1, 5), slice(2, 6)))
    np.testing.assert_array_equal(library.wavelength, np.stack((data[0]["wavgrid"][1:5], data[1]["wavgrid"][2:6])))
    np.testing.assert_array_equal(library.grid.field("flux").values,
                                  np.stack((data[0]["flux"][..., 1:5], data[1]["flux"][..., 2:6]), axis=-2))
    assert library.wavelength.size == 8  # no atmosphere-node duplication


def test_private_carrier_is_an_immutable_dynamic_pytree(prepared):
    name, axes, paths, _, _ = prepared
    library = LOADERS[name](paths)
    leaves, structure = jax.tree_util.tree_flatten(library)
    assert len(leaves) == len(axes) + 3
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert restored.regions == library.regions
    np.testing.assert_array_equal(restored.wavelength, library.wavelength)
    changed = jax.tree_util.tree_map(lambda leaf: leaf * 2, library)
    result = jax.jit(lambda lib: lib.wavelength.sum())(changed)
    np.testing.assert_allclose(result, 2 * library.wavelength.sum())
    with pytest.raises(FrozenInstanceError):
        library.regions = ("changed",)


@pytest.mark.parametrize("problem,message", [
    ("missing", "missing entries"), ("flux-shape", "flux shape"),
    ("int-flux", "floating-point"), ("wave-shape", "dimensional"),
    ("wave-duplicate", "strictly increasing"), ("wave-negative", "positive"),
    ("wave-nan", "finite"), ("axis-shape", "one-dimensional"),
])
def test_malformed_prepared_file_fails(tmp_path, problem, message):
    data = dict(tgrid=np.array([5000.]), ggrid=np.array([4.]), zgrid=np.array([0.]),
                wavgrid=np.array([5000., 5001., 5003.]), flux=np.ones((1, 1, 1, 3), dtype=np.float32))
    if problem == "missing":
        del data["zgrid"]
    elif problem == "flux-shape":
        data["flux"] = np.ones((1, 1, 1, 2))
    elif problem == "int-flux":
        data["flux"] = data["flux"].astype(int)
    elif problem == "wave-shape":
        data["wavgrid"] = data["wavgrid"][None]
    elif problem == "wave-duplicate":
        data["wavgrid"][1] = data["wavgrid"][0]
    elif problem == "wave-negative":
        data["wavgrid"][0] = -1
    elif problem == "wave-nan":
        data["wavgrid"][0] = np.nan
    elif problem == "axis-shape":
        data["tgrid"] = data["tgrid"][None]
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises((ValueError, TypeError), match=message):
        load_tlusty(path)


@pytest.mark.parametrize("problem,message", [
    ("axes", "identical atmosphere"), ("dtype", "flux dtype"),
    ("pixels", "pixel count"), ("wave-dtype", "wavelength dtype"),
])
def test_inconsistent_regions_fail(prepared, problem, message):
    name, _, paths, data, _ = prepared
    second = data[1].copy()
    if problem == "axes":
        second["tgrid"] = second["tgrid"] + 1
    elif problem == "dtype":
        second["flux"] = second["flux"].astype(np.float32 if name == "coelho" else np.float64)
    elif problem == "pixels":
        second["wavgrid"] = second["wavgrid"][:-1]
        second["flux"] = second["flux"][..., :-1]
    elif problem == "wave-dtype":
        second["wavgrid"] = second["wavgrid"].astype(np.float32)
    np.savez(paths[1], **second)
    with pytest.raises(ValueError, match=message):
        LOADERS[name](paths)


@pytest.mark.parametrize("kwargs,message", [
    ({"pixel_slices": [slice(None)]}, "one slice"),
    ({"pixel_slices": slice(None, None, -1)}, "positive step"),
    ({"pixel_slices": slice(2, 2)}, "non-empty"),
    ({"regions": [0]}, "one region identifier"),
    ({"regions": [[0], [1]]}, "strings or integers"),
    ({"wavelength_medium": "guess"}, "wavelength_medium"),
    ({"flux_kind": "absolute"}, "flux_kind"),
])
def test_loading_configuration_fails_clearly(prepared, kwargs, message):
    name, _, paths, _, _ = prepared
    with pytest.raises(ValueError, match=message):
        LOADERS[name](paths, **kwargs)


def test_coelho_missing_alpha_has_only_an_explicit_zero_singleton(tmp_path):
    path = tmp_path / "no_alpha.npz"
    np.savez(path, tgrid=[5000.], ggrid=[4.], fgrid=[0.], wavgrid=[5000., 5001.],
             flux=np.array([[[[0.8, 0.9]]]], dtype=np.float32))
    library = load_coelho(path)
    np.testing.assert_array_equal(library.grid.axis("alpha"), [0.])
    value = library.grid.interpolate(dict(teff=5000, logg=4, feh=0, alpha=0))["flux"]
    np.testing.assert_allclose(value, [[0.8, 0.9]])
    assert np.all(np.isneginf(library.grid.interpolate(dict(teff=5000, logg=4, feh=0, alpha=0.4))["flux"]))
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["flux"] = np.ones((1, 1, 1, 2, 2))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="multiple alpha"):
        load_coelho(path)


def test_loaders_follow_existing_jax_dtype_canonicalization(prepared, x64_context):
    name, _, paths, _, _ = prepared
    with x64_context(False):
        library = LOADERS[name](paths)
        assert library.grid.field("flux").values.dtype == jnp.float32
        assert library.wavelength.dtype == jnp.float32


def write_coelho(path, flux):
    hdu = fits.PrimaryHDU(np.stack((flux, 100 * flux)))
    hdu.header["CRVAL1"] = 5000.
    hdu.header["CD1_1"] = 0.5
    hdu.writeto(path)


@pytest.mark.parametrize("normalized", [True, False])
def test_raw_coelho_rows_native_sampling_and_vacuum_conversion(tmp_path, normalized):
    axes = dict(teff=[5000., 5500.], logg=[4.], feh=[0.], alpha=[0.])
    base = np.linspace(0.7, 1, 7)
    for i, t in enumerate(axes["teff"]):
        write_coelho(tmp_path / f"{int(t)}_40_p00p00.ms.fits", base + 0.1 * i)
    air = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5004)],
                          wavelength_medium="air", normalized=normalized)
    vacuum = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5006)],
                             normalized=normalized)
    np.testing.assert_array_equal(air.wavelength, [5000 + np.arange(7) * 0.5])
    assert np.all(vacuum.wavelength > air.wavelength)
    np.testing.assert_allclose(vacuum.wavelength[0, 0], 5001.394848638, rtol=1e-12)
    np.testing.assert_allclose(air.grid.field("flux").values[0, 0, 0, 0, 0],
                               base * (1 if normalized else 100), rtol=1e-6)
    assert vacuum.grid.field("flux").values.dtype == jnp.float32
    assert vacuum.flux_kind == ("normalized" if normalized else "unnormalized")


def test_raw_coelho_requires_explicit_source_replacement(tmp_path):
    alias = tmp_path / "4250_50_p05p04.ms.fits"
    write_coelho(alias, np.linspace(0.7, 1, 7))
    axes = dict(teff=[4250.], logg=[5.], feh=[0.5], alpha=[0.])
    with pytest.raises(FileNotFoundError):
        prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5006)])
    library = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5006)],
                              source_overrides={(4250., 5., 0.5, 0.): alias})
    assert library.sources == (str(alias),)


def write_gzip(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        np.savetxt(stream, data)


@pytest.mark.parametrize("normalized", [True, False])
def test_raw_bosz_naming_continuum_and_singleton_axes(tmp_path, normalized):
    axes = dict(teff=[5000.], logg=[2.5, 4.], mh=[0.], alpha=[0.], carbon=[0.], vmic=[2.])
    wav = np.array([5000., 5000.2, 5000.7, 5001.8, 5003.])
    for g in axes["logg"]:
        geom = "ms" if g <= 3 else "mp"
        path = tmp_path / "m+0.00" / f"bosz2024_{geom}_t5000_g+{g:.1f}_m+0.00_a+0.00_c+0.00_v2_rorig_noresam.txt.gz"
        write_gzip(path, np.stack((wav, np.linspace(1, 2, 5), np.full(5, 2)), axis=1))
    library = prepare_bosz(tmp_path, axes=axes, wavelength_ranges=[(4999, 5004)],
                           wavelength_medium="air", normalized=normalized)
    np.testing.assert_array_equal(library.wavelength, wav[None])
    expected = np.linspace(1, 2, 5) / (2 if normalized else 1)
    np.testing.assert_allclose(library.grid.field("flux").values[0, 0, 0, 0, 0, 0, 0], expected, rtol=1e-6)
    assert library.grid.axis_kinds[4] == "singleton"


def test_bosz_missing_files_do_not_trigger_logg4_fallback(tmp_path):
    axes = dict(teff=[5000.], logg=[2.5], mh=[0.], alpha=[0.], carbon=[0.], vmic=[2.])
    replacement = tmp_path / "m+0.00" / "bosz2024_mp_t5000_g+4.0_m+0.00_a+0.00_c+0.00_v2_rorig_noresam.txt.gz"
    write_gzip(replacement, [[5000, 1, 2], [5001, 2, 2]])
    with pytest.raises(FileNotFoundError):
        prepare_bosz(tmp_path, axes=axes, wavelength_ranges=[(4999, 5003)], wavelength_medium="air")
    library = prepare_bosz(tmp_path, axes=axes, wavelength_ranges=[(4999, 5003)],
                           wavelength_medium="air", source_overrides={(5000., 2.5, 0., 0., 0., 2.): replacement})
    assert library.sources == (str(replacement),)


def test_bosz_native_uv_is_already_vacuum(tmp_path):
    axes = dict(teff=[5000.], logg=[4.], mh=[0.], alpha=[0.], carbon=[0.], vmic=[2.])
    path = tmp_path / "m+0.00" / "bosz2024_mp_t5000_g+4.0_m+0.00_a+0.00_c+0.00_v2_rorig_noresam.txt.gz"
    write_gzip(path, [[1500, 1, 2], [1600, 2, 2]])
    library = prepare_bosz(tmp_path, axes=axes, wavelength_ranges=[(1499, 1601)])
    np.testing.assert_array_equal(library.wavelength, [[1500, 1600]])
    with pytest.raises(ValueError, match="2000 Angstrom"):
        prepare_bosz(tmp_path, axes=axes, wavelength_ranges=[(1499, 1601)], wavelength_medium="air")


def write_tlusty(directory, temperature, gravity, base, *, ostar=False):
    stem = f"{'G' if ostar else 'BG'}{temperature}g{int(round(gravity * 100)):03d}v2"
    wav = np.array([5000., 5000.2, 5000.7, 5001.1, 5001.9, 5002.4, 5003.])
    continuum = 2 + 0.1 * (wav - 5000)
    spectrum = directory / f"{stem}.7.gz"
    cont = directory / f"{stem}.17.gz"
    write_gzip(spectrum, np.stack((wav, base * continuum), axis=1))
    write_gzip(cont, [[4999, 1.9], [5001, 2.1], [5004, 2.4]])
    return spectrum, cont, wav, continuum


@pytest.mark.parametrize("kind", ["normalized", "median_scaled", "unnormalized"])
def test_raw_tlusty_pairs_discovery_and_flux_products(tmp_path, kind):
    base = np.linspace(0.7, 1, 7)
    for z, t in product([0.5, 1.0], [30000, 32500]):
        directory = tmp_path / f"Z{z}_{'vt2' if t == 30000 else 'ostar'}"
        _, _, wav, cont = write_tlusty(directory, t, 3.5, base, ostar=t > 30000)
    # Deliberately unusable OSTAR overlap: must be skipped at <=30000 K.
    write_gzip(tmp_path / "Z1.0_ostar" / "G30000g350v2.7.gz", [[1, 1], [2, 2]])
    library = prepare_tlusty(tmp_path, wavelength_ranges=[(4999, 5004)], flux_kind=kind)
    np.testing.assert_array_equal(library.grid.axis("teff"), [30000., 32500.])
    np.testing.assert_array_equal(library.grid.axis("logg"), [3.5])
    np.testing.assert_allclose(library.grid.axis("logZ"), np.log10([0.5, 1.0]))
    np.testing.assert_array_equal(library.wavelength, wav[None])
    expected = base if kind == "normalized" else base * cont
    if kind == "median_scaled":
        expected /= np.median(expected)
    np.testing.assert_allclose(library.grid.field("flux").values[0, 0, 0, 0], expected, rtol=2e-6)
    assert library.flux_kind == kind


def test_tlusty_incomplete_cartesian_grid_requires_explicit_replacement(tmp_path):
    write_tlusty(tmp_path / "Z1.0_vt2", 30000, 3, np.ones(7))
    write_tlusty(tmp_path / "Z1.0_vt2", 30000, 4, np.ones(7))
    spectrum, continuum, _, _ = write_tlusty(tmp_path / "Z1.0_ostar", 35000, 4, np.ones(7), ostar=True)
    with pytest.raises(ValueError, match="missing TLUSTY atmosphere nodes"):
        prepare_tlusty(tmp_path, wavelength_ranges=[(4999, 5004)])
    library = prepare_tlusty(tmp_path, wavelength_ranges=[(4999, 5004)],
                              source_overrides={(35000., 3., 0.): (spectrum, continuum)})
    assert library.grid.field("flux").values.shape == (2, 2, 1, 1, 7)


def test_native_region_packing_and_explicit_preparation_sampling(tmp_path):
    axes = dict(teff=[5000.], logg=[4.], feh=[0.], alpha=[0.])
    write_coelho(tmp_path / "5000_40_p00p00.ms.fits", np.arange(7, dtype=float) + 1)
    bounds = [(4999, 5001), (5001, 5003)]
    with pytest.raises(ValueError, match="different lengths"):
        prepare_coelho(tmp_path, axes=axes, wavelength_ranges=bounds, wavelength_medium="air")
    library = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=bounds,
                              pixels=5, wavelength_medium="air", regions=("blue", "red"))
    np.testing.assert_allclose(library.wavelength, [np.linspace(5000, 5000.5, 5), np.linspace(5001.5, 5002.5, 5)])
    np.testing.assert_allclose(library.grid.field("flux").values[0, 0, 0, 0],
                               [np.linspace(1, 2, 5), np.linspace(4, 6, 5)])
    assert library.regions == ("blue", "red")


def test_preparation_requires_shared_native_samples_or_explicit_sampling(tmp_path):
    axes = dict(teff=[5000., 5500.], logg=[4.], feh=[0.], alpha=[0.])
    for t in axes["teff"]:
        write_coelho(tmp_path / f"{int(t)}_40_p00p00.ms.fits", np.ones(7))
    with fits.open(tmp_path / "5500_40_p00p00.ms.fits", mode="update") as hdus:
        hdus[0].header["CD1_1"] = 0.6
    with pytest.raises(ValueError, match="samples differ"):
        prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5005)], wavelength_medium="air")
    library = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5005)],
                              pixels=8, wavelength_medium="air")
    np.testing.assert_allclose(library.wavelength, [np.linspace(5000, 5003, 8)])
    np.testing.assert_array_equal(library.grid.field("flux").values, np.ones((2, 1, 1, 1, 1, 8)))


def test_tlusty_missing_continuum_coverage_fails(tmp_path):
    _, continuum, _, _ = write_tlusty(tmp_path / "Z1.0_vt2", 30000, 3.5, np.ones(7))
    write_gzip(continuum, [[5001, 2], [5002, 2]])
    with pytest.raises(ValueError, match="continuum coverage"):
        prepare_tlusty(tmp_path, wavelength_ranges=[(4999, 5004)])


@pytest.mark.parametrize("pixels", [0, 1, 2.5, True])
def test_invalid_preparation_sampling_fails(tmp_path, pixels):
    with pytest.raises(ValueError, match="pixels"):
        prepare_coelho(tmp_path, axes=dict(teff=[5000], logg=[4], feh=[0], alpha=[0]),
                        wavelength_ranges=[(5000, 5010)], pixels=pixels)


def assert_same_spectra(expected, actual):
    for attribute in ("regions", "library", "wavelength_medium", "flux_kind", "sources", "wavelength_unit"):
        assert getattr(actual, attribute) == getattr(expected, attribute)
    assert actual.grid.axis_names == expected.grid.axis_names
    assert actual.grid.axis_kinds == expected.grid.axis_kinds
    assert actual.grid.boundary == expected.grid.boundary
    for name in expected.grid.axis_names:
        np.testing.assert_array_equal(actual.grid.axis(name), expected.grid.axis(name))
        assert actual.grid.axis(name).dtype == expected.grid.axis(name).dtype
    left, right = expected.grid.field("flux"), actual.grid.field("flux")
    assert right.dims == left.dims
    assert right.payload_dims == left.payload_dims
    assert right.values.dtype == left.values.dtype
    assert actual.wavelength.dtype == expected.wavelength.dtype
    np.testing.assert_array_equal(actual.wavelength, expected.wavelength)
    np.testing.assert_array_equal(right.values, left.values)
    np.testing.assert_array_equal(actual.grid.fill_value, expected.grid.fill_value)


def test_legacy_prepared_files_can_be_migrated_once_to_common_storage(prepared, tmp_path):
    name, axes, paths, _, _ = prepared
    legacy = LOADERS[name](paths, regions=(8, "red"), wavelength_medium="vacuum", flux_kind="normalized")
    path = save_spectral_grid(tmp_path / "common.npz", legacy)
    restored = load_spectral_grid(path)
    assert_same_spectra(legacy, restored)
    point = {name: (axis[0] + axis[-1]) / 2 for name, axis in axes.items()}
    np.testing.assert_array_equal(restored.grid.interpolate(point)["flux"],
                                  legacy.grid.interpolate(point)["flux"])
    with pytest.raises(ValueError, match="legacy jaxspec"):
        load_spectral_grid(paths[0])
    with pytest.raises(ValueError, match="missing entries"):
        LOADERS[name](path)


@pytest.mark.parametrize("name,kind", [
    ("coelho", "normalized"), ("coelho", "unnormalized"),
    ("bosz", "normalized"), ("bosz", "unnormalized"),
    ("tlusty", "normalized"), ("tlusty", "unnormalized"), ("tlusty", "median_scaled"),
])
def test_raw_preparation_common_artifact_round_trip_in_a_fresh_process(tmp_path, name, kind):
    raw = tmp_path / "raw"
    raw.mkdir()
    if name == "coelho":
        axes = dict(teff=[5000., 5500.], logg=[4.], feh=[0.], alpha=[0.])
        for i, t in enumerate(axes["teff"]):
            write_coelho(raw / f"{int(t)}_40_p00p00.ms.fits", np.linspace(0.7, 1, 7) + 0.1 * i)
        spectra = prepare_coelho(raw, axes=axes, wavelength_ranges=[(5000, 5003), (5003, 5005)],
                                 pixels=4, regions=(8, "red"), normalized=kind == "normalized")
    elif name == "bosz":
        axes = dict(teff=[5000., 5500.], logg=[4.], mh=[0.], alpha=[0.], carbon=[0.], vmic=[2.])
        wav = np.array([5000., 5000.2, 5000.7, 5001.8, 5003., 5003.4, 5004., 5005.])
        for i, t in enumerate(axes["teff"]):
            path = raw / "m+0.00" / f"bosz2024_mp_t{int(t)}_g+4.0_m+0.00_a+0.00_c+0.00_v2_rorig_noresam.txt.gz"
            write_gzip(path, np.stack((wav, np.linspace(1, 2, 8) + i * 0.1, np.full(8, 2)), axis=1))
        spectra = prepare_bosz(raw, axes=axes, wavelength_ranges=[(4999, 5002), (5002, 5006)],
                               regions=(8, "red"), wavelength_medium="air", normalized=kind == "normalized")
    else:
        for i, t in enumerate([30000, 32500]):
            write_tlusty(raw / f"Z1.0_{'vt2' if i == 0 else 'ostar'}", t, 3.5,
                         np.linspace(0.7, 1, 7) + i * 0.1, ostar=i > 0)
        spectra = prepare_tlusty(raw, wavelength_ranges=[(4999, 5001), (5001, 5002.5)],
                                 regions=(8, "red"), flux_kind=kind, wavelength_medium="vacuum")

    artifact = save_spectral_grid(tmp_path / "prepared.npz", spectra)
    # Original source paths no longer exist. The fresh process has only the
    # common artifact; it cannot reuse a prepared object or reopen those paths.
    raw.rename(tmp_path / "unavailable_raw")
    assert all(not Path(source).exists() for source in spectra.sources)
    point = {axis: float((spectra.grid.axis(axis)[0] + spectra.grid.axis(axis)[-1]) / 2)
             for axis in spectra.grid.axis_names}
    snapshot = tmp_path / "fresh_process.npz"
    script = """
import json
import sys
import jax
import numpy as np
from jaxstar.specfit import load_spectral_grid
spectra = load_spectral_grid(sys.argv[1])
point = json.loads(sys.argv[3])
result = jax.jit(lambda s: s.grid.interpolate(point)['flux'])(spectra)
metadata = {name: getattr(spectra, name) for name in
            ('regions', 'library', 'wavelength_medium', 'flux_kind', 'sources', 'wavelength_unit')}
metadata.update(axis_names=spectra.grid.axis_names, axis_kinds=spectra.grid.axis_kinds,
                flux_dims=spectra.grid.field('flux').dims, boundary=spectra.grid.boundary,
                payload_dims=spectra.grid.field('flux').payload_dims)
np.savez(sys.argv[2], metadata=json.dumps(metadata), result=np.asarray(result),
         flux=np.asarray(spectra.grid.field('flux').values), wavelength=np.asarray(spectra.wavelength),
         **{f'axis_{i}': np.asarray(spectra.grid.axis(n)) for i, n in enumerate(spectra.grid.axis_names)})
"""
    environment = dict(os.environ, JAX_ENABLE_X64="1",
                       PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    process = subprocess.run([sys.executable, "-c", script, str(artifact), str(snapshot), json.dumps(point)],
                             env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    with np.load(snapshot, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        for attribute in ("regions", "library", "wavelength_medium", "flux_kind", "sources", "wavelength_unit"):
            expected = getattr(spectra, attribute)
            assert metadata[attribute] == (list(expected) if isinstance(expected, tuple) else expected)
        assert metadata["axis_names"] == list(spectra.grid.axis_names)
        assert metadata["axis_kinds"] == list(spectra.grid.axis_kinds)
        assert metadata["flux_dims"] == list(spectra.grid.field("flux").dims)
        assert metadata["payload_dims"] == ["region", "pixel"]
        assert metadata["boundary"] == spectra.grid.boundary
        for i, axis in enumerate(spectra.grid.axis_names):
            np.testing.assert_array_equal(data[f"axis_{i}"], spectra.grid.axis(axis))
            assert data[f"axis_{i}"].dtype == spectra.grid.axis(axis).dtype
        np.testing.assert_array_equal(data["wavelength"], spectra.wavelength)
        np.testing.assert_array_equal(data["flux"], spectra.grid.field("flux").values)
        assert data["flux"].dtype == spectra.grid.field("flux").values.dtype
        assert data["wavelength"].dtype == spectra.wavelength.dtype
        np.testing.assert_array_equal(data["result"], jax.jit(lambda s: s.grid.interpolate(point)["flux"])(spectra))
    assert_same_spectra(spectra, load_spectral_grid(artifact))


@pytest.mark.parametrize("name,kind", [
    ("coelho", "normalized"), ("coelho", "unnormalized"),
    ("bosz", "normalized"), ("bosz", "unnormalized"),
    ("tlusty", "normalized"), ("tlusty", "unnormalized"), ("tlusty", "median_scaled"),
])
@pytest.mark.parametrize("control", ["pixels", "velocity_step"])
def test_direct_raw_log_sampling_products_nodes_and_queries(tmp_path, name, kind, control):
    bounds = np.array([[5000.15, 5001.2], [5001.4, 5002.6]])
    options = dict(sampling="log", regions=(8, "red"))
    if control == "pixels":
        options["pixels"] = 9
    else:
        # The same physical velocity spacing applies to both regions. Equal
        # log spans permit rectangular packing without changing the requested dv.
        bounds[1, 1] = bounds[1, 0] * bounds[0, 1] / bounds[0, 0]
        options["velocity_step"] = 1.0
    natives = []
    if name == "coelho":
        axes = dict(teff=[5000., 5500.], logg=[4.], feh=[0.], alpha=[0.])
        for i, t in enumerate(axes["teff"]):
            wav = 5000 + np.arange(7) * (0.5 + 0.1 * i)
            flux = 0.7 + 0.02 * (wav - 5000) + 0.1 * i
            path = tmp_path / f"{int(t)}_40_p00p00.ms.fits"
            write_coelho(path, flux)
            with fits.open(path, mode="update") as hdus:
                hdus[0].header["CD1_1"] = 0.5 + 0.1 * i
            natives.append((wav, flux * (1 if kind == "normalized" else 100)))
        spectra = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=bounds,
                                 wavelength_medium="air", normalized=kind == "normalized", **options)
    elif name == "bosz":
        axes = dict(teff=[5000., 5500.], logg=[4.], mh=[0.], alpha=[0.], carbon=[0.], vmic=[2.])
        for i, t in enumerate(axes["teff"]):
            wav = np.array([5000., 5000.2, 5000.7, 5001.8, 5003.]) + 0.03 * i
            flux = 0.7 + 0.02 * (wav - 5000) + 0.1 * i
            path = tmp_path / "m+0.00" / f"bosz2024_mp_t{int(t)}_g+4.0_m+0.00_a+0.00_c+0.00_v2_rorig_noresam.txt.gz"
            write_gzip(path, np.stack((wav, flux * 2, np.full(len(wav), 2)), axis=1))
            natives.append((wav, flux * (1 if kind == "normalized" else 2)))
        spectra = prepare_bosz(tmp_path, axes=axes, wavelength_ranges=bounds,
                               wavelength_medium="air", normalized=kind == "normalized", **options)
    else:
        for i, t in enumerate([30000, 32500]):
            base = np.linspace(0.7, 1, 7) + i * 0.1
            _, _, wav, continuum = write_tlusty(tmp_path / f"Z1.0_{'vt2' if i == 0 else 'ostar'}",
                                               t, 3.5, base, ostar=i > 0)
            natives.append((wav, base if kind == "normalized" else base * continuum))
        spectra = prepare_tlusty(tmp_path, wavelength_ranges=bounds, flux_kind=kind,
                                 wavelength_medium="air", **options)

    assert is_log_uniform(spectra)
    expected_count = (9 if control == "pixels" else
                      int(np.floor(np.log(bounds[0, 1] / bounds[0, 0]) * 299792.458)) + 1)
    assert spectra.wavelength.shape == (2, expected_count)
    assert spectra.wavelength.dtype == np.float64
    assert spectra.grid.field("flux").values.dtype == np.float32
    assert spectra.regions == (8, "red")
    assert spectra.flux_kind == kind
    assert spectra.wavelength_medium == "air"
    step = np.diff(np.log(spectra.wavelength), axis=-1)
    np.testing.assert_allclose(step, np.broadcast_to(step[:, :1], step.shape), rtol=1e-8)
    if control == "velocity_step":
        np.testing.assert_allclose(step * 299792.458, options["velocity_step"], rtol=1e-8)
    expected_nodes = []
    for i, (wav, values) in enumerate(natives):
        rows = []
        for r, (lo, hi) in enumerate(bounds):
            scaling = np.median(values[(wav > lo) & (wav < hi)]) if kind == "median_scaled" else 1
            rows.append(np.interp(spectra.wavelength[r], wav, values / scaling))
        expected = np.array(rows).astype(np.float32)
        expected_nodes.append(expected)
        point = {axis: spectra.grid.axis(axis)[i if axis == "teff" else 0] for axis in spectra.grid.axis_names}
        np.testing.assert_array_equal(spectra.grid.interpolate(point)["flux"], expected)

    def evaluate(lib, teff):
        point = {axis: lib.grid.axis(axis)[0] for axis in lib.grid.axis_names}
        return lib.grid.interpolate({**point, "teff": teff})["flux"]

    temperatures = spectra.grid.axis("teff")
    midpoint = temperatures[0] + 0.3 * (temperatures[-1] - temperatures[0])
    expected = 0.7 * expected_nodes[0] + 0.3 * expected_nodes[1]
    np.testing.assert_allclose(jax.jit(evaluate)(spectra, midpoint), expected, rtol=2e-6)
    batch = jax.jit(evaluate)(spectra, jnp.array([temperatures[0], midpoint]))
    np.testing.assert_allclose(batch, np.stack((expected_nodes[0], expected)), rtol=2e-6)
    gradient = jax.grad(lambda temperature: evaluate(spectra, temperature).sum())(midpoint)
    span = temperatures[-1] - temperatures[0]
    expected_gradient = (expected_nodes[1].astype(np.float64) - expected_nodes[0]).sum() / span
    # Median-scaled nodes can have almost equal integrated flux. Account for
    # cancellation of float32 corner reductions as the sample count grows.
    reduction_roundoff = (2 * np.finfo(np.float32).eps
                          * np.maximum(np.abs(expected_nodes[0]), np.abs(expected_nodes[1])).sum(dtype=np.float64)
                          / span)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=reduction_roundoff)
    artifact = save_spectral_grid(tmp_path / "log.npz", spectra)
    assert_same_spectra(spectra, load_spectral_grid(artifact))
    assert is_log_uniform(load_spectral_grid(artifact))


def test_log_preparation_preserves_vacuum_conversion_and_requires_coverage(tmp_path):
    axes = dict(teff=[5000.], logg=[4.], feh=[0.], alpha=[0.])
    write_coelho(tmp_path / "5000_40_p00p00.ms.fits", np.ones(7))
    vacuum = prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(5001.5, 5004.1)],
                            sampling="log", velocity_step=1.0)
    assert vacuum.wavelength_medium == "vacuum"
    assert is_log_uniform(vacuum)
    np.testing.assert_allclose(vacuum.wavelength[0, 0], 5001.5, rtol=0, atol=1e-12)
    assert 0 <= np.log(5004.1 / vacuum.wavelength[0, -1]) < 1 / 299792.458
    np.testing.assert_allclose(np.diff(np.log(vacuum.wavelength[0])) * 299792.458, 1.0, rtol=1e-8)
    np.testing.assert_array_equal(vacuum.grid.field("flux").values, 1)
    with pytest.raises(ValueError, match="does not cover requested log range"):
        prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(4999, 5003)],
                        sampling="log", velocity_step=1.0, wavelength_medium="air")


@pytest.mark.parametrize("control", ["velocity_step", "pixels"])
def test_legacy_conversion_and_fresh_process_loading(prepared, tmp_path, control):
    name, _, paths, _, _ = prepared
    # Unequal legacy log spans yield unequal counts at one dv. Convert one
    # region for normal fitting use; count mode separately exercises packing.
    selected_paths = paths[:1] if control == "velocity_step" else paths
    regions = (8,) if control == "velocity_step" else (8, "red")
    legacy = LOADERS[name](selected_paths, regions=regions, wavelength_medium="vacuum", flux_kind="unnormalized")
    options = {"velocity_step": 1.0} if control == "velocity_step" else {"pixels": 11}
    converted = resample_spectral_grid(legacy, **options)
    assert is_log_uniform(converted)
    for attribute in ("regions", "library", "wavelength_medium", "flux_kind", "sources", "wavelength_unit"):
        assert getattr(converted, attribute) == getattr(legacy, attribute)
    assert converted.grid.axis_names == legacy.grid.axis_names
    for axis in legacy.grid.axis_names:
        np.testing.assert_array_equal(converted.grid.axis(axis), legacy.grid.axis(axis))
    values = np.asarray(legacy.grid.field("flux").values)
    for index in np.ndindex(values.shape[:-2]):
        for r in range(len(regions)):
            expected = np.interp(converted.wavelength[r], legacy.wavelength[r], values[index + (r,)])
            np.testing.assert_allclose(converted.grid.field("flux").values[index + (r,)], expected, rtol=1e-7)
    artifact = save_spectral_grid(tmp_path / "converted.npz", converted)
    script = """
import sys
import numpy as np
from jaxstar.specfit import load_spectral_grid, is_log_uniform
spectra = load_spectral_grid(sys.argv[1])
assert is_log_uniform(spectra)
point = {n: (spectra.grid.axis(n)[0] + spectra.grid.axis(n)[-1]) / 2 for n in spectra.grid.axis_names}
np.savez(sys.argv[2], wavelength=np.asarray(spectra.wavelength),
         flux=np.asarray(spectra.grid.field('flux').values), result=np.asarray(spectra.grid.interpolate(point)['flux']))
"""
    environment = dict(os.environ, JAX_ENABLE_X64="1", PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    output = tmp_path / "fresh.npz"
    process = subprocess.run([sys.executable, "-c", script, str(artifact), str(output)],
                             env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["wavelength"], converted.wavelength)
        np.testing.assert_array_equal(data["flux"], converted.grid.field("flux").values)
        point = {n: (converted.grid.axis(n)[0] + converted.grid.axis(n)[-1]) / 2 for n in converted.grid.axis_names}
        np.testing.assert_array_equal(data["result"], converted.grid.interpolate(point)["flux"])
