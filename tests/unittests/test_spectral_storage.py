"""Common artifact validation without any source-library axis conventions."""

import json

import jax
import numpy as np
import pytest

from jaxstar.grid import Axis, Field, RectilinearGrid
from jaxstar.specfit import load_spectral_grid, save_spectral_grid
from jaxstar.specfit._data import _SpectralLibrary


@pytest.fixture
def artifact(tmp_path, x64_context):
    with x64_context():
        # Arbitrary producer/axes, field axis order distinct from grid order,
        # an explicitly nonuniform kernel and a nondefault boundary fill.
        grid = RectilinearGrid(
            axes={"abundance_X": Axis([-0.5, 0., 0.5], "nonuniform"),
                  "temperature": [4000., 4300., 5000.]},
            fields={"flux": Field(np.arange(54, dtype=np.float64).reshape(3, 3, 2, 3),
                                  ("temperature", "abundance_X"), payload_dims=("region", "pixel"))},
            fill_value=-123.,
        )
        spectra = _SpectralLibrary(grid, [[5000., 5000.2, 5001.], [6000., 6000.8, 6002.]],
                                   ("赤", 7), "custom_synthesis", "vacuum", "unnormalized", ("offline-input",))
        path = save_spectral_grid(tmp_path / "custom.npz", spectra)
        yield path, spectra


def test_common_schema_has_no_library_registry_or_native_axis_keys(artifact):
    path, spectra = artifact
    restored = load_spectral_grid(path)
    assert restored.library == "custom_synthesis"
    assert restored.regions == ("赤", 7)
    assert restored.sources == ("offline-input",)
    assert restored.grid.axis_names == spectra.grid.axis_names
    assert restored.grid.axis_kinds == spectra.grid.axis_kinds
    assert restored.grid.field("flux").dims == ("temperature", "abundance_X")
    point = {"temperature": 4550., "abundance_X": 0.2}
    np.testing.assert_array_equal(restored.grid.interpolate(point)["flux"], spectra.grid.interpolate(point)["flux"])
    assert np.all(restored.grid.interpolate({**point, "temperature": 6000.})["flux"] == -123.)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(restored))
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"metadata", "axis_0", "axis_1", "flux", "wavelength", "fill_value"}
        assert all(data[key].dtype.kind != "O" for key in data.files)
        metadata = json.loads(data["metadata"].item())
        assert metadata["format"] == "jaxstar.spectral_grid"
        assert metadata["version"] == 1


def test_save_uses_exact_path_and_requires_explicit_overwrite(artifact, tmp_path):
    path, spectra = artifact
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_spectral_grid(path, spectra)
    assert path.read_bytes() == original
    save_spectral_grid(path, spectra, overwrite=True)
    np.testing.assert_array_equal(load_spectral_grid(path).wavelength, spectra.wavelength)
    exact_path = tmp_path / "no_extension"
    assert save_spectral_grid(exact_path, spectra) == exact_path
    assert exact_path.exists()
    assert not exact_path.with_suffix(".npz").exists()
    with pytest.raises(TypeError, match="loader/preparer result"):
        save_spectral_grid(tmp_path / "bad.npz", spectra.grid)


@pytest.mark.parametrize("select", [False, True])
@pytest.mark.parametrize("problem,message", [
    ("missing-array", "missing.*arrays"), ("wrong-format", "format"),
    ("version", "unsupported.*version"), ("boolean-version", "unsupported.*version"),
    ("invalid-json", "JSON"), ("metadata-shape", "scalar JSON"),
    ("missing-metadata", "missing.*metadata"), ("axis-names", "non-empty and unique"),
    ("axis-kinds", "axis kind"), ("source-types", "list of strings"),
    ("flux-dims", "every grid axis"), ("payload-dims", "payload dimensions"),
    ("wave-shape", "shape must match"), ("unit", "angstrom"),
    ("regions", "one region identifier"), ("library", "provenance label"),
])
def test_malformed_common_artifact_fails(artifact, tmp_path, problem, message, select):
    path, _ = artifact
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(arrays["metadata"].item())
    if problem == "missing-array":
        del arrays["axis_1"]
    elif problem == "wrong-format":
        metadata["format"] = "something_else"
    elif problem == "version":
        metadata["version"] = 2
    elif problem == "boolean-version":
        metadata["version"] = True
    elif problem == "missing-metadata":
        del metadata["flux_kind"]
    elif problem == "axis-names":
        metadata["axis_names"] = ["repeated", "repeated"]
    elif problem == "axis-kinds":
        metadata["axis_kinds"] = ["nonuniform"]
    elif problem == "source-types":
        metadata["sources"] = [7]
    elif problem == "flux-dims":
        metadata["flux_dims"] = ["temperature"]
    elif problem == "payload-dims":
        metadata["payload_dims"] = ["order", "pixel"]
    elif problem == "wave-shape":
        arrays["wavelength"] = arrays["wavelength"][:, :-1]
    elif problem == "unit":
        metadata["wavelength_unit"] = "nm"
    elif problem == "regions":
        metadata["regions"] = ["red"]
    elif problem == "library":
        metadata["library"] = ""
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    if problem == "invalid-json":
        arrays["metadata"] = np.asarray("{")
    elif problem == "metadata-shape":
        arrays["metadata"] = arrays["metadata"][None]
    bad = tmp_path / "malformed.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match=message):
        load_spectral_grid(bad, **({"wavelength": [5000., 5000.1], "wav_margin": 0.}
                                  if select else {}))


@pytest.mark.parametrize("select", [False, True])
def test_common_loading_follows_jax_precision_setting(artifact, x64_context, select):
    path, _ = artifact
    with x64_context(False):
        spectra = load_spectral_grid(path, **({"wavelength": [6000., 6002.], "wav_margin": 0.}
                                              if select else {}))
        assert spectra.grid.field("flux").values.dtype == np.float32
        assert spectra.wavelength.dtype == np.float32


@pytest.mark.parametrize("labels", [None, ("first", "second", "first")])
def test_wavelength_selection_preserves_values_schema_and_row_association(artifact, labels):
    path, original = artifact
    indices = [1, 0, 1]
    # Requests may use only part of a saved region; saved samples stay intact.
    wave = [[6000.4, 6001.6], [5000.2, 5000.8], [6000.5, 6001.5]]
    result = load_spectral_grid(path, wavelength=wave, wav_margin=.1, regions=labels)
    assert result.regions == ((7, "赤", 7) if labels is None else labels)
    for key in ("library", "wavelength_medium", "wavelength_unit", "flux_kind", "sources"):
        assert getattr(result, key) == getattr(original, key)
    assert result.grid.axis_names == original.grid.axis_names
    assert result.grid.axis_kinds == original.grid.axis_kinds
    assert result.grid.field("flux").dims == original.grid.field("flux").dims
    assert result.grid.boundary == original.grid.boundary
    assert result.grid.fill_value == original.grid.fill_value
    np.testing.assert_array_equal(result.wavelength, np.asarray(original.wavelength)[indices])
    np.testing.assert_array_equal(result.grid.field("flux").values,
                                  np.asarray(original.grid.field("flux").values)[..., indices, :])
    point = {"temperature": 4550., "abundance_X": .2}
    np.testing.assert_array_equal(result.grid.interpolate(point)["flux"],
                                  np.asarray(original.grid.interpolate(point)["flux"])[indices])


def test_selection_reaches_jax_only_after_host_subset(artifact, monkeypatch):
    from jaxstar.specfit import storage

    path, original = artifact
    construct = storage.RectilinearGrid
    inputs = []

    def checked_grid(**kwargs):
        flux = kwargs["fields"]["flux"].values
        assert isinstance(flux, np.ndarray)
        assert flux.shape == (3, 3, 1, 3)
        inputs.append(flux)
        return construct(**kwargs)

    monkeypatch.setattr(storage, "RectilinearGrid", checked_grid)
    result = load_spectral_grid(path, wavelength=[6000., 6002.], wav_margin=0.)
    assert len(inputs) == 1
    assert result.regions == (7,)
    np.testing.assert_array_equal(result.wavelength, np.asarray(original.wavelength)[1:])
    # Exactly sufficient margins also count as coverage.
    assert load_spectral_grid(path, wavelength=[6000.5, 6001.5],
                              wav_margin=.5).regions == (7,)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_coverage_failure_precedes_flux_loading(artifact, tmp_path, monkeypatch, ambiguous):
    path, _ = artifact
    if ambiguous:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["wavelength"][1] = arrays["wavelength"][0]
        path = tmp_path / "overlapping.npz"
        np.savez(path, **arrays)
    with np.load(path) as data:
        npz_type = type(data)
    getitem = npz_type.__getitem__

    def forbid_flux(self, key):
        assert key != "flux", "coverage must be checked before reading flux"
        return getitem(self, key)

    monkeypatch.setattr(npz_type, "__getitem__", forbid_flux)
    wave = [5000.2, 5000.8] if ambiguous else [4999., 5000.5]
    with pytest.raises(ValueError, match=f"Region 'obs'.*found {2 if ambiguous else 0}") as error:
        load_spectral_grid(path, wavelength=wave, regions=("obs",), wav_margin=.1)
    assert str(path) in str(error.value)
    assert "row 0 (region '赤')" in str(error.value)
    assert "Angstrom" in str(error.value)


@pytest.mark.parametrize("wave,options,message", [
    ([], {}, "non-empty"),
    ([[[5000., 5001.]]], {}, "shape"),
    ([5001., 5000.], {}, "strictly increasing"),
    ([5000., np.nan], {}, "finite"),
    ([0., 1.], {}, "positive"),
    ([5000., 5001.], {"wav_margin": -1.}, "wav_margin"),
    ([5000., 5001.], {"wav_margin": np.inf}, "wav_margin"),
    ([5000., 5001.], {"regions": (8, 9)}, "one identifier"),
    (None, {"regions": (8,)}, "regions requires wavelength"),
])
def test_invalid_wavelength_selection(artifact, wave, options, message):
    path, _ = artifact
    with pytest.raises(ValueError, match=message):
        load_spectral_grid(path, wavelength=wave, **options)
