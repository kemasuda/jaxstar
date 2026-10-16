"""Offline v2.5 acquisition, schema, chemistry and model integration checks."""

import io
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxstar.mistfit import MistFit, MistModel
from jaxstar.mistfit.mistgrid import _tracks_v25 as v25, cache, create_tracks
from jaxstar.mistfit.mistgrid.sources import (
    bolometric_correction_archive, isochrone_archive, track_archives, track_metallicities,
)
from jaxstar.mistfit.results import _grid_record
from test_mist_track_builder import FEHS, MASSES, bc_files, track_text, write_archive


def native_track(mass, feh, *, rotation=.4, end=5):
    text = track_text(mass, feh, rotation=rotation, end=end)
    text = text.replace("version number = 1.2", "version number = 2.5")
    text = text.replace(" 2 8 YES", " 2 9 YES")
    text = text.replace("log_surf_z surface_h1", "log_surf_cell_z surface_h1 surface_fe56")
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            values = line.split()
            values[6] = str(np.log10(.01685)+feh)
            line = " ".join(values)+" 0.001"
        lines.append(line)
    return "\n".join(lines)+"\n"


def bc_v2_files():
    files = {}
    for text in bc_files().values():
        lines = text.splitlines()
        for afe in (0., .4):
            out = ["# lgTef logg Fe_H a_Fe Av Rv " + " ".join(v25.BANDS)]
            for row in lines[1:]:
                values = list(map(float, row.split()))
                feh = values[2]
                # Make the other alpha slice unmistakably different.
                bands = np.array(values[5:]) + 100*afe
                out.append(" ".join(map(str, (np.log10(values[0]), *values[1:3],
                                               afe, *values[3:5], *bands))))
            files[f"feh{feh:+.2f}_afe{afe:+.1f}.UBVRIplus"] = "\n".join(out)+"\n"
    return files


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(cache, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))


def test_v25_catalogue_and_cache_identity():
    all_sources = []
    for afe in (-.2, 0., .2, .4, .6):
        for rotation in (0., .4):
            sources = track_archives(version="2.5", afe=afe, rotation=rotation)
            assert len(sources) == (16 if afe == .6 else 17)
            assert all("/tarballs_v2.5/eeps/" in s.url for s in sources)
            all_sources.extend(sources)
    assert len({s.cache_key for s in all_sources}) == 168
    source, = track_archives(version="2.5", afe=-.2, feh=[-2.75], rotation=0.)
    assert source.filename == "MIST_v2.5_feh_m275_afe_m2_vvcrit0.0_EEPS.txz"
    assert track_metallicities(version="2.5")[0::16] == (-4., .5)
    with pytest.raises(ValueError, match="Unavailable"):
        track_archives(version="2.5", afe=.6, feh=[.5])
    bc = bolometric_correction_archive(version="2.5")
    assert bc.url == "https://mist.science/BC_tables/v2/UBVRIplus.txz"
    assert bc == bolometric_correction_archive(version="2.5", afe=.6)
    assert bc.cache_key != bolometric_correction_archive().cache_key
    with pytest.raises(ValueError, match="1.2"):
        isochrone_archive(version="2.5")


@pytest.mark.parametrize("rotation", [0., .4])
def test_v25_build_load_predict_and_provenance(tmp_path, x64_context, rotation):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_archive(inputs / "UBVRIplus.txz", bc_v2_files())
    for feh, source in zip(FEHS, track_archives(version="2.5", rotation=rotation, feh=FEHS)):
        write_archive(inputs / source.filename, {
            f"tracks/{round(mass*100):05d}M.track.eep": native_track(mass, feh, rotation=rotation)
            for mass in MASSES})
    path = create_tracks.create_track_grid(tmp_path / "v25.npz", version="2.5", rotation=rotation,
                                          feh=FEHS, local_dir=inputs, raw_cache_dir=tmp_path / "raw",
                                          offline=True)
    with np.load(path) as data:
        # No implicit small-domain restriction: all fixture masses/EEPs survive.
        np.testing.assert_array_equal(data["eep"], np.arange(1, 6))
        np.testing.assert_array_equal(data["initial_mass"], MASSES)
        mass, feh, eep = np.meshgrid(MASSES, FEHS, np.arange(1, 6), indexing="ij")
        np.testing.assert_allclose(data["bc_metallicity"], feh, atol=1e-15)
        np.testing.assert_allclose(data["cmd_surface_feh"], feh-np.log10(.7/.73465), atol=1e-15)
        bc = feh[..., None]*.4 + (.7*(3.73+.005*eep)+4.15*.2)[..., None] + np.arange(14)*.03
        expected = 4.74-2.5*(.2*eep+.1*mass)[..., None]-bc
        np.testing.assert_allclose(data["absolute_magnitudes"], expected, atol=4e-15)
        metadata = json.loads(data["metadata_json"].item())
        assert metadata["mist_version"] == "2.5" and metadata["Zsun"] == .01685
        assert metadata["chemistry"]["log_z_column"] == "log_surf_cell_z"
        assert set(metadata["implementation_sha256"]) == {"create_tracks.py", "_tracks_v12.py", "_tracks_v25.py"}
        assert all(r["source"]["version"] == "2.5" for r in metadata["inputs"])
    with x64_context():
        model = MistModel.from_tracks(path)
        leaves, treedef = jax.tree_util.tree_flatten(model)
        clone = jax.tree_util.tree_unflatten(treedef, leaves)
        assert clone.metadata == model.metadata
        record = model.metadata
        record["chemistry"]["log_z_column"] = "mutated"
        assert model.metadata["chemistry"]["log_z_column"] == "log_surf_cell_z"
        assert _grid_record(model)["metadata"]["mist_version"] == "2.5"
        point = dict(mass_init=.9, feh_init=0., eep=3.)
        result = jax.jit(lambda m, p: m(p, keys=("jmag_abs", "feh", "dtdeep")))(model, point)
        np.testing.assert_allclose(result["feh"], -np.log10(.7/.73465), atol=1e-15)
        assert result["dtdeep"] > 0
        gradient = jax.grad(lambda m: model({**point, "mass_init": m}, keys=("jmag_abs",))["jmag_abs"])(.9)
        np.testing.assert_allclose(gradient, -.25, atol=1e-14)


def test_v25_parser_requires_version_and_named_composition():
    text = native_track(1., 0.)
    for bad, match in [(text.replace("2.5", "1.2", 1), "v2.5"),
                       (text.replace("log_surf_cell_z", "log_surf_z"), "log_surf_cell_z"),
                       (text.replace("surface_fe56", "old_column"), "surface_fe56")]:
        with pytest.raises(ValueError, match=match):
            v25.read_track(io.BytesIO(bad.encode()))


def test_v25_header_mass_roundoff_does_not_split_native_axis(tmp_path):
    path = tmp_path / "tracks.txz"
    text = native_track(85., 0.).replace("# 85.0 5", "# 84.999999999 5")
    write_archive(path, {"08500M.track.eep": text})
    bc = tmp_path / "bc.txz"
    write_archive(bc, bc_v2_files())
    axes, fields = v25.read_bc_archive(bc)
    args = (path, 0., .4, 0., None, None, axes, fields, True, v25)
    tracks, records = create_tracks._read_archive(*args)
    assert tuple(tracks) == (85.,)
    assert records[0]["header_initial_mass"] == 84.999999999
    write_archive(path, {"08500M.track.eep": text.replace("84.999999999", "84.999")})
    with pytest.raises(ValueError, match="Initial mass disagrees"):
        create_tracks._read_archive(*args)


def test_v25_bc_alpha_isolation_and_schema_validation(tmp_path):
    path = tmp_path / "bc.txz"
    files = bc_v2_files()
    write_archive(path, files)
    axes, fields = v25.read_bc_archive(path)
    other_axes, other = v25.read_bc_archive(path, afe=.4)
    np.testing.assert_allclose(axes[1], np.log10([4000., 5000., 6000., 7000., 8000.]))
    np.testing.assert_array_equal(other_axes[0], axes[0])
    np.testing.assert_allclose(other-fields, 40.)
    with pytest.raises(ValueError, match="four BC metallicities"):
        v25.read_bc_archive(path, afe=.2)
    name = next(iter(files))
    files[name] = files[name].replace("lgTef", "Teff")
    write_archive(path, files)
    with pytest.raises(ValueError, match="column header"):
        v25.read_bc_archive(path)


def test_v25_surface_composition_cap_and_invalid_values():
    data = {"log_surf_cell_z": np.log10(.01685)+np.array([0., 1., np.nan, -4.]),
            "surface_h1": np.array([.73465, .7, .7, 0.])}
    metal, surface = v25.surface_composition(data, 0.)
    np.testing.assert_allclose(metal, [0., .1, np.nan, -4.], equal_nan=True)
    np.testing.assert_allclose(surface, [0., .1-np.log10(.7/.73465), np.nan, np.nan], equal_nan=True)


def test_frozen_official_v25_photometry_and_surface_feh():
    # Independent official outputs, not expectations generated by the builder.
    # Provenance and reproduction command: benchmarks/mist_v25/README.md.
    path = Path(__file__).parents[1] / "data/mist_v25/official_reference.npz"
    with np.load(path, allow_pickle=False) as data:
        for i, row in enumerate(data["raw"]):
            feh, log_z, hydrogen, log_t, log_g, log_l = row
            metal, _ = v25.surface_composition({"log_surf_cell_z": np.array([log_z]),
                                                 "surface_h1": np.array([hydrogen])}, feh)
            axes = tuple(data[key][i] for key in ("bc_z", "bc_t", "bc_g"))
            query = np.array([[metal[0], log_t, log_g]])
            result = 4.74-2.5*log_l-v25.track_bc(axes, data["bc_values"][i], query)[0]
            np.testing.assert_allclose(result, data["expected_magnitudes"][i], atol=5e-4, rtol=0.)
        raw = data["iso_raw"]
        _, feh = v25.surface_composition({"log_surf_cell_z": raw[:, 1], "surface_h1": raw[:, 2]}, raw[:, 0])
        np.testing.assert_allclose(feh, data["iso_expected"], atol=5e-6, rtol=0.)


def structure_track(mass, feh):
    """Synthetic named columns, including an unusable optional diagnostic."""
    text = native_track(mass, feh)
    lines = text.splitlines()
    columns = lines[6].lstrip("#").split()
    added = [name for name in v25.STRUCTURE_COLUMNS if name not in columns]
    lines[4] = lines[4].replace(" 2 9 YES", f" 2 {len(columns)+len(added)} YES")
    lines[6] += " " + " ".join(added)
    for i in range(7, len(lines)):
        values = {name: 10.+j+mass+feh+(i-6) for j, name in enumerate(added)}
        values["phase"] = 0.
        values["conv_env_turnover_time_g"] = 86400.*(mass+feh+i-6)
        if mass == MASSES[0] and feh == FEHS[0] and i == 9:
            values["e_thermal"] = np.nan
        lines[i] += " " + " ".join(str(values[name]) for name in added)
    return "\n".join(lines)+"\n"


def test_optional_structure_preserves_fields_and_fit_with_missing_diagnostic(tmp_path, x64_context):
    from numpyro.infer.util import log_density

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    write_archive(inputs / "UBVRIplus.txz", bc_v2_files())
    for feh, source in zip(FEHS, track_archives(version="2.5", feh=FEHS)):
        write_archive(inputs / source.filename, {
            f"{round(mass*100):05d}M.track.eep": structure_track(mass, feh) for mass in MASSES})
    options = dict(version="2.5", feh=FEHS, local_dir=inputs,
                   raw_cache_dir=tmp_path / "raw", offline=True)
    base = create_tracks.create_track_grid(tmp_path / "base.npz", **options)
    expanded = create_tracks.create_track_grid(tmp_path / "expanded.npz", include_structure=True, **options)
    with np.load(base) as a, np.load(expanded) as b:
        assert len(v25.STRUCTURE_COLUMNS) == len(set(v25.STRUCTURE_COLUMNS)) == 66
        assert set(v25.STRUCTURE_COLUMNS) <= set(b.files)
        assert len(set(b.files)-set(a.files)) == 65  # star_mass already exists
        assert "surface_h1" not in b.files  # isotope abundances remain excluded
        for name in a.files:
            if name != "metadata_json":
                np.testing.assert_array_equal(a[name], b[name])
        assert np.isnan(b["e_thermal"][0, 0, 2])
        np.testing.assert_allclose(b["star_age"], b["age_gyr"]*1e9)
        for mass_index, mass in enumerate(MASSES):
            for feh_index, feh in enumerate(FEHS):
                expected, _, _ = v25.read_track(io.BytesIO(structure_track(mass, feh).encode()))
                for name in v25.STRUCTURE_COLUMNS:
                    np.testing.assert_array_equal(b[name][mass_index, feh_index], expected[name])
    with x64_context():
        models = [MistModel.from_tracks(path) for path in (base, expanded)]
        assert models[1].metadata["native_columns"] == list(v25.STRUCTURE_COLUMNS)
        point = dict(mass_init=.9, feh_init=0., eep=3., distance=.1)
        result = jax.jit(lambda m, p: m(p, keys=("conv_env_turnover_time_g", "e_thermal")))(models[1], point)
        np.testing.assert_allclose(result["conv_env_turnover_time_g"], 3.9*86400.)
        assert np.isneginf(result["e_thermal"])
        observations = {"teff": (5700., 100.), "kmag": (8., .03), "parallax": (10., .1)}
        parameters = dict(mass=.9, feh_init=0., eep=3., distance=.1)
        outputs = []
        for model in models:
            fit = MistFit(model=model, observations=observations)
            density = lambda p: log_density(fit.model, (), {"age_max": None}, p)[0]
            value, gradient = jax.jit(jax.value_and_grad(density))(parameters)
            outputs.append((value, gradient))
        for old, new in zip(jax.tree_util.tree_leaves(outputs[0]), jax.tree_util.tree_leaves(outputs[1])):
            assert np.isfinite(new).all()
            np.testing.assert_array_equal(old, new)
    with pytest.raises(ValueError, match="Nonfinite selected"):
        create_tracks.create_track_grid(tmp_path / "strict.npz", include_structure=True,
                                       allow_missing=False, **options)


def test_structure_selection_rejects_wrong_release_or_missing_columns(tmp_path):
    with pytest.raises(ValueError, match="requires MIST v2.5"):
        create_tracks.create_track_grid(tmp_path / "wrong.npz", include_structure=True)
    with pytest.raises(TypeError, match="include_structure must be a bool"):
        create_tracks.create_track_grid(tmp_path / "wrong.npz", include_structure="yes")
    path = tmp_path / "tracks.txz"
    write_archive(path, {"00100M.track.eep": native_track(1., 0.)})
    with pytest.raises(ValueError, match="Missing requested native columns"):
        create_tracks._read_archive(path, 0., .4, 0., None, None, None, None, True,
                                   v25, ("total_moment_of_inertia",))
