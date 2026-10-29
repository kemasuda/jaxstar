"""Coelho Teff neighbours preserve composition and fail early if incomplete."""

from itertools import product
import json

from astropy.io import fits
import numpy as np
import pytest

from jaxstar.specfit import check_spectral_inputs, prepare_coelho


def write_node(root, node, *, offset=0.):
    t, g, f, a = node
    path = root / f"{t}_{int(g*10):02d}_{'p' if f >= 0 else 'm'}{int(abs(f)*10):02d}p{int(a*10):02d}.ms.fits"
    wave = 4999. + np.arange(251)*.02 + offset
    flux = .6 + .02*((t-4000)/250)**2 + .01*g + .2*a + .03*np.sin(3*(wave-5000))
    # Distinct scaling at each temperature catches averaging the wrong row.
    unnormalized = flux * (2 + .1*(t-4000)/250)
    hdu = fits.PrimaryHDU(np.stack((flux, unnormalized)))
    hdu.header["CRVAL1"], hdu.header["CD1_1"] = wave[0], .02
    hdu.writeto(path)
    return path, wave, flux, unnormalized


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("sampling", ["native", "linear", "log"])
def test_known_missing_nodes_average_teff_at_fixed_composition(
        tmp_path, x64_context, normalized, sampling):
    axes = dict(teff=[4250, 4750, 5250], logg=[3., 4., 5.], feh=[.2, .5], alpha=[0., .4])
    nodes = set(product(*axes.values()))
    missing = {(4250, 5., .5, 0.), (5250, 3., .2, 0.),
               *((4750, g, .5, .4) for g in axes["logg"])}
    contributors = {
        node: [(node[0]-250, *node[1:]), (node[0]+250, *node[1:])] if node in missing else [node]
        for node in nodes}
    sources = {source for values in contributors.values() for source in values}
    raw = {node: write_node(tmp_path, node,
                           offset=.007 if node[0] == 5000 and sampling != "native" else 0.)
           for node in sources}
    report = check_spectral_inputs("coelho", tmp_path, axes=axes, missing_nodes="teff")
    assert report["ready"] and not report["unresolved"]
    assert report["requested_nodes"] == 36 and len(report["replacements"]) == 5
    for entry in report["replacements"]:
        assert entry["method"] == "linear_teff"
        assert [source["weight"] for source in entry["sources"]] == [.5, .5]
        assert [source["node"]["teff"] for source in entry["sources"]] == [
            entry["node"]["teff"]-250, entry["node"]["teff"]+250]
        for source in entry["sources"]:
            assert all(source["node"][key] == entry["node"][key] for key in ("logg", "feh", "alpha"))
    json.dumps(report, allow_nan=False)
    settings = dict(axes=axes, wavelength_ranges=[(5000, 5001), (5002, 5003)],
                    wavelength_medium="air", normalized=normalized,
                    missing_nodes="teff", sampling=sampling)
    if sampling != "native":
        settings["pixels"] = 32
    with x64_context():
        spectra = prepare_coelho(tmp_path, **settings)
        assert spectra.grid.field("flux").values.dtype == np.float32
        assert set(spectra.sources) == {str(raw[node][0]) for node in sources}
        assert set(spectra.sources) == set(report["sources"])
        wave = np.asarray(spectra.wavelength)
        for node in nodes:
            expected = []
            for source in contributors[node]:
                _, x, y, unnormalized = raw[source]
                expected.append([np.interp(row, x, y if normalized else unnormalized) for row in wave])
            actual = spectra.grid.interpolate(dict(zip(axes, node)))["flux"]
            np.testing.assert_allclose(actual, np.mean(expected, axis=0), rtol=1e-6)


@pytest.mark.parametrize("temperatures,available", [
    ([4250], [4000]),  # No lower-only copy, even at the requested upper edge.
    ([4250], [4500]),  # No upper-only copy.
    ([4250], [3750, 4750]),  # No wider-neighbour search.
    ([4250, 4500], [4000, 4750]),  # No recursive filling of adjacent holes.
])
def test_missing_teff_neighbours_fail_before_any_fits_reads(tmp_path, monkeypatch, temperatures, available):
    for t in available:
        write_node(tmp_path, (t, 5., .5, 0.))
    # Other compositions/gravities cannot replace native neighbours or targets.
    for t in (4000, 4250, 4500, 4750):
        for g, f, a in ((4., .5, 0.), (5., 0., 0.), (5., .5, .4)):
            write_node(tmp_path, (t, g, f, a))
    axes = dict(teff=temperatures, logg=[5.], feh=[.5], alpha=[0.])
    monkeypatch.setattr(fits, "open", lambda *a, **k: pytest.fail("read before all input checks passed"))
    report = check_spectral_inputs("coelho", tmp_path, axes=axes, missing_nodes="teff")
    assert not report["ready"] and len(report["unresolved"]) == len(temperatures)
    assert all(entry["missing_neighbors"] for entry in report["unresolved"])
    with pytest.raises(FileNotFoundError, match="no spectra were read") as error:
        prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(5000, 5003)],
                       sampling="log", pixels=32, missing_nodes="teff")
    for entry in report["unresolved"]:
        assert entry["missing_path"] in str(error.value)
        assert all(path in str(error.value) for path in entry["missing_neighbors"])


def test_existing_corrupt_coelho_file_is_not_replaced(tmp_path):
    for t in (4000, 4500):
        write_node(tmp_path, (t, 5., .5, 0.))
    (tmp_path / "4250_50_p05p00.ms.fits").write_text("not a FITS spectrum\n")
    axes = dict(teff=[4250], logg=[5.], feh=[.5], alpha=[0.])
    report = check_spectral_inputs("coelho", tmp_path, axes=axes, missing_nodes="teff")
    assert report["ready"] and report["replacements"] == []
    with pytest.raises(OSError):
        prepare_coelho(tmp_path, axes=axes, wavelength_ranges=[(5000, 5003)],
                       sampling="log", pixels=32, missing_nodes="teff")


def test_teff_policy_is_explicit_coelho_only_and_excludes_overrides(tmp_path):
    for t in (4000, 4500):
        write_node(tmp_path, (t, 5., .5, 0.))
    axes = dict(teff=[4250], logg=[5.], feh=[.5], alpha=[0.])
    assert not check_spectral_inputs("coelho", tmp_path, axes=axes)["ready"]
    with pytest.raises(ValueError, match="cannot be combined with source_overrides"):
        check_spectral_inputs("coelho", tmp_path, axes=axes, missing_nodes="teff",
                              source_overrides={(4250, 5., .5, 0.): "4000_50_p05p00.ms.fits"})
    with pytest.raises(ValueError, match="only for Coelho"):
        check_spectral_inputs("bosz", tmp_path, missing_nodes="teff",
                              axes=dict(teff=[4250], logg=[5.], mh=[.5], alpha=[0.], carbon=[0.], vmic=[1]))
