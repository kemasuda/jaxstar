"""OB2025 pairing, family selection, normalization and reusable provenance."""

from dataclasses import replace
import gzip
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxstar.specfit import (SpecModel, check_spectral_inputs, load_spectral_grid,
                            prepare_tlusty, resample_spectral_grid, save_spectral_grid)
from jaxstar.specfit.libraries import _air_to_vacuum


@pytest.fixture(autouse=True)
def precision(x64_context):
    with x64_context():
        yield


def write_pair(root, family, teff, *, logg=4., code="100", vt=2, value=.8, suffix=""):
    directory = root / f"{family}_z{code}v{vt}"
    directory.mkdir(exist_ok=True, parents=True)
    stem = f"z{code}t{teff}g{round(logg*100):03d}v{vt}{suffix}"
    wave = np.linspace(4998., 5006., 801)
    continuum = 2 + .01 * (wave - 5000)
    spectrum = directory / f"{stem}.spec.gz"
    cont = directory / f"{stem}.cont.gz"
    with gzip.open(spectrum, "wt") as stream:
        np.savetxt(stream, np.column_stack((wave, value * continuum)))
    # Coarser continuum, with the repeated terminal row found in real files.
    idx = [0, 400, 800, 800]
    with gzip.open(cont, "wt") as stream:
        np.savetxt(stream, np.column_stack((wave[idx], continuum[idx])))
    return spectrum, cont


def test_family_switch_fixed_vt_and_metadata_survive_loading_and_jit(tmp_path):
    write_pair(tmp_path, "BSTAR", 29000, value=.7)
    write_pair(tmp_path, "OSTAR", 30000, value=.9)
    # These alternatives must never enter the preparation or axis discovery.
    write_pair(tmp_path, "BSTAR", 30000, value=99.)
    write_pair(tmp_path, "OSTAR", 27500, value=99.)
    write_pair(tmp_path, "OSTAR", 30000, vt=10, value=99.)
    write_pair(tmp_path, "OSTAR", 30000, value=99., suffix="old")
    spectra = prepare_tlusty(tmp_path, wavelength_ranges=[(5000., 5003.)],
                             sampling="log", velocity_step=3.)
    assert spectra.grid.axis_names == ("teff", "logg", "logZ")
    np.testing.assert_array_equal(spectra.grid.axis("teff"), [29000, 30000])
    assert spectra.wavelength_medium == "vacuum"
    assert spectra.grid.field("flux").values.dtype == np.float32
    np.testing.assert_allclose(np.diff(np.log(spectra.wavelength[0])), 3./299792.458,
                               atol=2e-15, rtol=0)
    np.testing.assert_allclose(spectra.grid.field("flux").values[:, 0, 0, 0],
                               np.broadcast_to([[.7], [.9]], (2, spectra.wavelength.shape[1])))
    assert len(spectra.sources) == 4
    assert spectra.provenance["synthesis_vmic_kms"] == 2
    assert spectra.provenance["atmosphere_vmic_kms"] == {"BSTAR": 2., "OSTAR": 10.}
    assert all("v10" not in Path(p).name and "old" not in Path(p).name for p in spectra.sources)

    path = save_spectral_grid(tmp_path / "grid.npz", spectra)
    # from_grid selects an observed wavelength region without access to raw files.
    model = SpecModel.from_grid(path, wavelength=np.array([5000.5, 5001., 5002.]),
                                wav_margin=.1, vmax=10.)
    restored = model.spectra
    assert restored.provenance == spectra.provenance
    restored.provenance["atmosphere_vmic_kms"]["OSTAR"] = -99
    assert restored.provenance == spectra.provenance  # immutable nested metadata
    expected = restored.grid.interpolate(dict(teff=29500., logg=4., logZ=0.))["flux"]
    compiled = jax.jit(lambda lib: lib.grid.interpolate(
        dict(teff=29500., logg=4., logZ=0.))["flux"])(restored)
    np.testing.assert_allclose(compiled, .8)
    np.testing.assert_allclose(compiled, expected)
    assert resample_spectral_grid(restored, pixels=5).provenance == spectra.provenance


def test_preflight_reports_all_missing_pairs_without_reading_any_flux(tmp_path, monkeypatch):
    _, cont = write_pair(tmp_path, "BSTAR", 29000)
    cont.unlink()
    write_pair(tmp_path, "OSTAR", 30000)[0].write_bytes(b"")
    axes = dict(teff=[29000, 30000, 32500], logg=[4.], logZ=[0.])

    def no_read(*args, **kwargs):
        pytest.fail("preflight must finish before any spectrum or continuum is opened")

    monkeypatch.setattr(np, "loadtxt", no_read)
    report = check_spectral_inputs("tlusty", tmp_path, axes=axes)
    assert not report["ready"] and report["requested_nodes"] == 3
    assert len(report["unresolved"]) == 3
    missing = [p for entry in report["unresolved"] for p in entry["missing_paths"]]
    assert len(missing) == 4
    assert str(cont) in missing
    with pytest.raises(FileNotFoundError, match="no spectra were read") as error:
        prepare_tlusty(tmp_path, axes=axes, wavelength_ranges=[(5000, 5003)])
    assert all(p in str(error.value) for p in missing)


def test_missing_ostar_does_not_fall_back_to_bstar_or_neighbor_logg(tmp_path):
    write_pair(tmp_path, "BSTAR", 30000, logg=3.)
    write_pair(tmp_path, "OSTAR", 30000, logg=4.)
    axes = dict(teff=[30000], logg=[3., 4.], logZ=[0.])
    report = check_spectral_inputs("tlusty", tmp_path, axes=axes)
    assert not report["ready"] and not report["replacements"]
    assert "OSTAR_z100v2" in report["unresolved"][0]["missing_path"]
    with pytest.raises(ValueError, match="'error' or 'lowest_logg'"):
        check_spectral_inputs("tlusty", tmp_path, axes=axes, missing_nodes="logg")


@pytest.mark.parametrize("kind", ["normalized", "unnormalized", "median_scaled"])
def test_missing_lower_logg_uses_boundary_at_each_temperature_and_metallicity(tmp_path, kind):
    axes = dict(teff=[35000, 50000], logg=[3., 3.25, 3.5, 3.75, 4.],
                logZ=[np.log10(.5), 0.])
    minima = {(35000, "050"): 3., (35000, "100"): 3.25,
              (50000, "050"): 3.75, (50000, "100"): 4.}
    for (teff, code), lower in minima.items():
        for g in axes["logg"]:
            if g >= lower:
                write_pair(tmp_path, "OSTAR", teff, logg=g, code=code, value=.1*g)
    report = check_spectral_inputs("tlusty", tmp_path, axes=axes, missing_nodes="lowest_logg")
    assert report["ready"] and len(report["replacements"]) == 8
    assert report["requested_nodes"] == 20 and len(report["sources"]) == 24
    for entry in report["replacements"]:
        assert entry["method"] == "copy_lowest_logg"
        target, source = entry["node"], entry["sources"][0]
        assert target["teff"] == source["node"]["teff"]
        assert target["logZ"] == source["node"]["logZ"]
        assert target["logg"] < source["node"]["logg"]
        assert source["weight"] == 1.

    spectra = prepare_tlusty(tmp_path, axes=axes, wavelength_ranges=[(5000, 5003)],
                             sampling="log", velocity_step=3., flux_kind=kind,
                             missing_nodes="lowest_logg")
    for i, teff in enumerate(axes["teff"]):
        for k, code in enumerate(("050", "100")):
            lower = minima[teff, code]
            values = np.asarray(spectra.grid.field("flux").values)[i, :, k]
            boundary_index = axes["logg"].index(lower)
            for j, g in enumerate(axes["logg"]):
                if g < lower:
                    np.testing.assert_array_equal(values[j], values[boundary_index])
                if kind == "normalized":
                    np.testing.assert_allclose(values[j], .1*max(g, lower), atol=2e-8)
    assert spectra.provenance["missing_nodes"] == "lowest_logg"
    assert spectra.provenance["replacements"] == report["replacements"]
    saved = save_spectral_grid(tmp_path / "copied.npz", spectra)
    assert load_spectral_grid(saved).provenance == spectra.provenance


def test_lower_boundary_can_be_outside_requested_axis_without_crossing_families(tmp_path):
    # At 30000 K only OSTAR may supply the boundary, even if BSTAR has lower g.
    write_pair(tmp_path, "BSTAR", 30000, logg=3., value=99.)
    write_pair(tmp_path, "OSTAR", 30000, logg=4., value=.8)
    # A different vt and an old file must not lower the discovered boundary.
    write_pair(tmp_path, "OSTAR", 30000, logg=3., vt=10, value=99.)
    write_pair(tmp_path, "OSTAR", 30000, logg=3., suffix="old", value=99.)
    axes = dict(teff=[30000], logg=[3., 3.5], logZ=[0.])
    spectra = prepare_tlusty(tmp_path, axes=axes, wavelength_ranges=[(5000, 5003)],
                             missing_nodes="lowest_logg")
    np.testing.assert_allclose(spectra.grid.field("flux").values, .8)
    assert len(spectra.sources) == 2
    np.testing.assert_array_equal(spectra.grid.axis("logg"), [3., 3.5])
    assert all(e["sources"][0]["node"]["logg"] == 4. for e in spectra.provenance["replacements"])


@pytest.mark.parametrize("problem", ["interior", "upper", "missing-continuum", "empty", "broken-boundary", "no-family"])
def test_lowest_logg_policy_does_not_hide_invalid_or_non_lower_gaps(tmp_path, monkeypatch, problem):
    pair = write_pair(tmp_path, "OSTAR", 35000, logg=3.5)
    write_pair(tmp_path, "OSTAR", 35000, logg=4.)
    target = 3.
    if problem == "interior":
        target = 3.75
    elif problem == "upper":
        target = 4.25
    elif problem in {"missing-continuum", "empty"}:
        pair = write_pair(tmp_path, "OSTAR", 35000, logg=3.)
        if problem == "missing-continuum":
            pair[1].unlink()
        else:
            pair[0].write_bytes(b"")
    elif problem == "broken-boundary":
        pair[1].unlink()
    axes = dict(teff=[35000 if problem != "no-family" else 36000], logg=[target], logZ=[0.])
    monkeypatch.setattr(np, "loadtxt", lambda *a, **kw: pytest.fail("must fail before reading flux"))
    report = check_spectral_inputs("tlusty", tmp_path, axes=axes, missing_nodes="lowest_logg")
    assert not report["ready"] and not report["replacements"]
    with pytest.raises(FileNotFoundError, match="no spectra were read"):
        prepare_tlusty(tmp_path, axes=axes, wavelength_ranges=[(5000, 5003)],
                       missing_nodes="lowest_logg")


def test_nominal_low_metallicity_alias_is_explicit(tmp_path):
    write_pair(tmp_path, "BSTAR", 29000, code="0032", value=.7)
    write_pair(tmp_path, "OSTAR", 30000, code="0033", value=.9)
    spectra = prepare_tlusty(tmp_path, wavelength_ranges=[(5000, 5003)])
    np.testing.assert_allclose(spectra.grid.axis("logZ"), [np.log10(.032)])
    assert spectra.provenance["metallicity_z_over_sun"]["0033"] == .032
    with pytest.raises(ValueError, match="native Z/Z_sun"):
        check_spectral_inputs("tlusty", tmp_path,
                              axes=dict(teff=[30000], logg=[4.], logZ=[np.log10(.033)]))


@pytest.mark.parametrize("kind", ["normalized", "unnormalized", "median_scaled"])
def test_vacuum_to_air_preserves_flux_product(tmp_path, kind):
    write_pair(tmp_path, "OSTAR", 35000)
    vacuum = prepare_tlusty(tmp_path, wavelength_ranges=[(4990, 5010)], flux_kind=kind)
    air = prepare_tlusty(tmp_path, wavelength_ranges=[(4990, 5010)],
                         wavelength_medium="air", flux_kind=kind)
    np.testing.assert_allclose(_air_to_vacuum(np.asarray(air.wavelength)),
                               vacuum.wavelength, rtol=0, atol=1e-10)
    assert np.all(air.wavelength < vacuum.wavelength)
    np.testing.assert_array_equal(air.grid.field("flux").values,
                                  vacuum.grid.field("flux").values)
    if kind == "normalized":
        np.testing.assert_allclose(air.grid.field("flux").values, .8)
    elif kind == "unnormalized":
        np.testing.assert_allclose(air.grid.field("flux").values[0, 0, 0],
                                   .8 * (2 + .01 * (vacuum.wavelength-5000)), rtol=1e-6)


@pytest.mark.parametrize("data,match", [
    ([[4998, 2], [5006, 2], [5006, 3]], "conflicting continuum"),
    ([[4998, 2], [5006, 0]], "finite and positive"),
    ([[5001, 2], [5002, 2]], "continuum coverage"),
])
def test_bad_continua_fail_without_arbitrary_merging(tmp_path, data, match):
    _, cont = write_pair(tmp_path, "OSTAR", 35000)
    with gzip.open(cont, "wt") as stream:
        np.savetxt(stream, data)
    with pytest.raises(ValueError, match=match):
        prepare_tlusty(tmp_path, wavelength_ranges=[(5000, 5003)])


def test_empty_gzip_content_has_a_clear_error(tmp_path):
    _, cont = write_pair(tmp_path, "OSTAR", 35000)
    with gzip.open(cont, "wt"):
        pass
    with pytest.warns(UserWarning, match="no data"):
        with pytest.raises(ValueError, match="continuum contains no samples"):
            prepare_tlusty(tmp_path, wavelength_ranges=[(5000, 5003)])


def test_mixed_formats_require_explicit_selection(tmp_path):
    write_pair(tmp_path, "OSTAR", 35000)
    (tmp_path / "Z1.0_ostar").mkdir()
    with pytest.raises(ValueError, match="mixed TLUSTY formats"):
        prepare_tlusty(tmp_path, wavelength_ranges=[(5000, 5003)])
    spectra = prepare_tlusty(tmp_path, wavelength_ranges=[(5000, 5003)], source_format="ob2025")
    assert spectra.provenance["release"] == "TLUSTY OB2025"


def test_optional_provenance_is_backward_compatible(tmp_path):
    write_pair(tmp_path, "OSTAR", 35000)
    original = prepare_tlusty(tmp_path, wavelength_ranges=[(5000, 5003)])
    path = save_spectral_grid(tmp_path / "new.npz", original)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(arrays["metadata"].item())
    del metadata["provenance"]
    arrays["metadata"] = np.array(json.dumps(metadata))
    legacy = tmp_path / "old-format1.npz"
    np.savez(legacy, **arrays)
    restored = load_spectral_grid(legacy)
    assert restored.provenance == {}
    np.testing.assert_array_equal(restored.grid.field("flux").values,
                                  original.grid.field("flux").values)
    with pytest.raises(ValueError, match="JSON object"):
        replace(original, provenance_json="[]")
