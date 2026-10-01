"""Common artifact validation without any source-library axis conventions."""

import json

import jax
import numpy as np
import pytest

from jaxstar.grid import Axis, Field, RectilinearGrid
from jaxstar.specfit import load_spectral_grid, save_spectral_grid
from jaxstar.specfit._data import _SpectralLibrary


@pytest.fixture
def artifact(tmp_path):
    with jax.experimental.enable_x64():
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
def test_malformed_common_artifact_fails(artifact, tmp_path, problem, message):
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
        load_spectral_grid(bad)


def test_common_loading_follows_jax_precision_setting(artifact):
    path, _ = artifact
    with jax.experimental.disable_x64():
        spectra = load_spectral_grid(path)
        assert spectra.grid.field("flux").values.dtype == np.float32
        assert spectra.wavelength.dtype == np.float32
