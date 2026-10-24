"""Native logg neighbours, explicit edge copies and early input checks."""

import gzip
import json

import numpy as np
import pytest
from astropy.io import fits

from jaxstar.specfit import (
    check_spectral_inputs, prepare_bosz, prepare_coelho,
    save_spectral_grid, load_spectral_grid,
)


def write_node(root, g, *, t=6000, v=1, m=0., a=0., offset=0., continuum=2.):
    geometry = "mp" if g > 3 else "ms"
    path = root / f"m{m:+.2f}" / (
        f"bosz2024_{geometry}_t{t}_g{g:+.1f}_m{m:+.2f}_a{a:+.2f}_c+0.00_v{v}_rorig_noresam.txt.gz")
    path.parent.mkdir(exist_ok=True)
    wave = np.linspace(4999, 5004, 251) + offset
    flux = .3 + .02*g*g + .03*v + .04*np.sin(3*(wave-5000))
    with gzip.open(path, "wt") as stream:
        np.savetxt(stream, np.column_stack((wave, flux*continuum, np.full(wave.size, continuum))))
    return path, wave, flux


def axes(gravities):
    return dict(teff=[6000], logg=gravities, mh=[0.], alpha=[0.], carbon=[0.], vmic=[1])


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("sampling", ["native", "linear", "log"])
def test_missing_logg_averages_resampled_native_flux(tmp_path, x64_context, normalized, sampling):
    grid_axes = axes([3.5, 4.])
    grid_axes["vmic"] = [0, 1, 2]
    raw = {}
    for v in grid_axes["vmic"]:
        raw[v, 3.] = write_node(tmp_path, 3., v=v, continuum=2.)
        raw[v, 4.] = write_node(tmp_path, 4., v=v, continuum=5.,
                                offset=0. if sampling == "native" else .007)
    settings = dict(axes=grid_axes, wavelength_ranges=[(5000, 5001), (5002, 5003)],
                    wavelength_medium="air", normalized=normalized,
                    sampling=sampling, missing_nodes="logg")
    if sampling != "native":
        settings["pixels"] = 32
    report = check_spectral_inputs("bosz", tmp_path, axes=grid_axes, missing_nodes="logg")
    assert report["ready"] and not report["unresolved"]
    assert len(report["replacements"]) == 3
    for entry in report["replacements"]:
        assert entry["method"] == "linear_logg"
        assert [s["node"]["logg"] for s in entry["sources"]] == [3., 4.]
        assert [s["weight"] for s in entry["sources"]] == [.5, .5]
        for source in entry["sources"]:
            assert {k: v for k, v in source["node"].items() if k != "logg"} == {
                k: v for k, v in entry["node"].items() if k != "logg"}
    json.dumps(report, allow_nan=False)
    with x64_context():
        spectra = prepare_bosz(tmp_path, **settings)
        assert set(spectra.sources) == {str(value[0]) for value in raw.values()}
        assert spectra.grid.field("flux").values.dtype == np.float32
        assert spectra.wavelength.dtype == np.float64
        wave = np.asarray(spectra.wavelength)
        values = np.asarray(spectra.grid.field("flux").values)
        for i, v in enumerate(grid_axes["vmic"]):
            expected = []
            for g, continuum in ((3., 2.), (4., 5.)):
                _, x, y = raw[v, g]
                if not normalized:
                    y = y * continuum
                expected.append(np.stack([np.interp(row, x, y) for row in wave]))
            np.testing.assert_allclose(values[0, 0, 0, 0, 0, i], (expected[0]+expected[1])/2, rtol=1e-6)
            np.testing.assert_allclose(values[0, 1, 0, 0, 0, i], expected[1], rtol=1e-6)
        restored = load_spectral_grid(save_spectral_grid(tmp_path / "grid.npz", spectra))
        np.testing.assert_array_equal(restored.grid.field("flux").values, values)
        assert restored.sources == spectra.sources


@pytest.mark.parametrize("upper_exists,target_exists,method", [
    (False, False, "copy_lower_logg"),
    (True, False, "linear_logg"),
    (False, True, None),
])
def test_upper_edge_uses_native_target_then_bracket_then_lower(tmp_path, x64_context, upper_exists, target_exists, method):
    write_node(tmp_path, 4.5)
    # Different Teff, chemistry or vmic cannot supply the missing upper neighbour.
    write_node(tmp_path, 5.5, t=6250)
    write_node(tmp_path, 5.5, m=.25)
    write_node(tmp_path, 5.5, v=2)
    if upper_exists:
        write_node(tmp_path, 5.5)
    if target_exists:
        write_node(tmp_path, 5.)
    grid_axes = axes([4.5, 5.])
    report = check_spectral_inputs("bosz", tmp_path, axes=grid_axes, missing_nodes="logg")
    assert report["ready"]
    assert [entry["method"] for entry in report["replacements"]] == ([] if method is None else [method])
    with x64_context():
        spectra = prepare_bosz(tmp_path, axes=grid_axes, wavelength_ranges=[(5000, 5003)],
                               pixels=64, sampling="log", wavelength_medium="air", missing_nodes="logg")
        flux = np.asarray(spectra.grid.field("flux").values)[0, :, 0, 0, 0, 0, 0]
        if method == "copy_lower_logg":
            np.testing.assert_array_equal(flux[0], flux[1])
        else:
            assert not np.array_equal(flux[0], flux[1])


@pytest.mark.parametrize("gravities,available,missing_count", [
    ([3., 3.5], [3.5], 1),  # No copying downwards from the upper neighbour.
    ([4., 4.5, 5.], [4.], 2),  # No interior copy or recursive filling.
    ([4., 5.], [4.], 1),  # No jumping over the absent immediate lower neighbour.
])
def test_unresolved_neighbors_are_reported_before_raw_reads(tmp_path, monkeypatch, gravities, available, missing_count):
    for g in available:
        write_node(tmp_path, g)
    grid_axes = axes(gravities)
    monkeypatch.setattr(np, "loadtxt", lambda *a, **k: pytest.fail("read before input check finished"))
    report = check_spectral_inputs("bosz", tmp_path, axes=grid_axes, missing_nodes="logg")
    assert not report["ready"] and len(report["unresolved"]) == missing_count
    with pytest.raises(FileNotFoundError, match=f"failed for {missing_count} node.*no spectra were read"):
        prepare_bosz(tmp_path, axes=grid_axes, wavelength_ranges=[(5000, 5003)],
                     pixels=32, sampling="log", missing_nodes="logg")


@pytest.mark.parametrize("library", ["bosz", "coelho"])
def test_strict_mode_lists_all_missing_inputs_before_opening_first(tmp_path, monkeypatch, library):
    if library == "bosz":
        write_node(tmp_path, 4.)
        grid_axes, prepare = axes([4.]), prepare_bosz
        monkeypatch.setattr(np, "loadtxt", lambda *a, **k: pytest.fail("unexpected spectrum read"))
    else:
        (tmp_path / "6000_40_p00p00.ms.fits").touch()
        grid_axes, prepare = dict(teff=[6000], logg=[4.], feh=[0.], alpha=[0.]), prepare_coelho
        monkeypatch.setattr(fits, "open", lambda *a, **k: pytest.fail("unexpected spectrum read"))
    grid_axes["teff"] = [6000, 6250, 6500]
    report = check_spectral_inputs(library, tmp_path, axes=grid_axes)
    assert not report["ready"] and len(report["unresolved"]) == 2
    with pytest.raises(FileNotFoundError, match="no spectra were read") as error:
        prepare(tmp_path, axes=grid_axes, wavelength_ranges=[(5000, 5003)], pixels=32, sampling="log")
    assert "6250" in str(error.value) and "6500" in str(error.value)


def test_existing_corrupt_file_is_not_replaced(tmp_path):
    write_node(tmp_path, 3.)
    write_node(tmp_path, 4.)
    path, _, _ = write_node(tmp_path, 3.5)
    with gzip.open(path, "wt") as stream:
        stream.write("corrupt spectrum\n")
    grid_axes = axes([3.5])
    report = check_spectral_inputs("bosz", tmp_path, axes=grid_axes, missing_nodes="logg")
    assert report["ready"] and report["replacements"] == []
    with pytest.raises(ValueError):
        prepare_bosz(tmp_path, axes=grid_axes, wavelength_ranges=[(5000, 5003)],
                     sampling="log", pixels=16, missing_nodes="logg")


def test_policy_is_explicit_and_bosz_only(tmp_path):
    grid_axes = axes([3.5])
    write_node(tmp_path, 3.)
    write_node(tmp_path, 4.)
    assert not check_spectral_inputs("bosz", tmp_path, axes=grid_axes)["ready"]
    with pytest.raises(ValueError, match="missing_nodes"):
        check_spectral_inputs("bosz", tmp_path, axes=grid_axes, missing_nodes="nearest")
    with pytest.raises(ValueError, match="only for BOSZ"):
        check_spectral_inputs("coelho", tmp_path, axes=dict(teff=[6000], logg=[4], feh=[0], alpha=[0]),
                              missing_nodes="logg")
